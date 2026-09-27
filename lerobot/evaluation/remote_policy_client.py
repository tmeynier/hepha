"""Hepha adapter for LeRobot's asynchronous gRPC policy-server protocol."""

from __future__ import annotations

import pickle  # nosec: only used through a loopback SSH tunnel to a trusted server
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def require_loopback_server(address: str) -> tuple[str, int]:
    """Require an SSH-tunnel endpoint instead of exposing unsafe gRPC publicly."""
    host, separator, port_text = address.rpartition(":")
    host = host.strip("[]")
    if not separator or not host or not port_text:
        raise ValueError("--remote-server must use HOST:PORT format")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("--remote-server port must be an integer") from exc
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            "Remote policy transport must connect through a loopback SSH tunnel, for "
            "example --remote-server 127.0.0.1:8080. LeRobot async inference uses "
            "unauthenticated gRPC and pickle and must not be exposed publicly."
        )
    if not 1 <= port <= 65535:
        raise ValueError("--remote-server port must be between 1 and 65535")
    return host, port


def build_remote_lerobot_features(
    joints: tuple[str, ...],
    *,
    camera_name: str,
    width: int,
    height: int,
) -> dict[str, dict]:
    hardware_features: dict[str, type | tuple[int, int, int]] = {joint: float for joint in joints}
    hardware_features[camera_name] = (height, width, 3)
    return hw_to_dataset_features(hardware_features, OBS_STR, use_video=False)


class HephaRemotePolicyClient:
    """Chunked client compatible with ``lerobot.async_inference.policy_server``.

    The GPU server predicts chunks in a background request while the physical
    control loop consumes previously received actions. Overlapping chunk entries
    use LeRobot's ``latest_only`` convention.
    """

    def __init__(
        self,
        *,
        server_address: str,
        policy_type: str,
        policy_path: str,
        policy_device: str,
        actions_per_chunk: int,
        prefetch_threshold: float,
        camera_name: str,
        width: int,
        height: int,
        joints: tuple[str, ...],
        connect_timeout: float,
        load_timeout: float,
        request_timeout: float,
    ) -> None:
        require_loopback_server(server_address)
        if actions_per_chunk <= 0:
            raise ValueError("Remote actions_per_chunk must be positive")
        if not 0.0 <= prefetch_threshold <= 1.0:
            raise ValueError("--remote-prefetch-threshold must be between 0 and 1")
        if connect_timeout <= 0 or load_timeout <= 0 or request_timeout <= 0:
            raise ValueError("Remote timeouts must be positive")

        self.server_address = server_address
        self.policy_type = policy_type
        self.policy_path = policy_path
        self.policy_device = policy_device
        self.actions_per_chunk = actions_per_chunk
        self.prefetch_threshold = prefetch_threshold
        self.camera_name = camera_name
        self.joints = joints
        self.connect_timeout = connect_timeout
        self.load_timeout = load_timeout
        self.request_timeout = request_timeout
        self.lerobot_features = build_remote_lerobot_features(
            joints,
            camera_name=camera_name,
            width=width,
            height=height,
        )

        self._grpc: Any = None
        self._services_pb2: Any = None
        self._send_bytes_in_chunks: Any = None
        self._timed_observation_class: Any = None
        self._channel: Any = None
        self._stub: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hepha-remote-policy")
        self._future: Future | None = None
        self._queued_actions: dict[int, torch.Tensor] = {}
        self._last_executed_timestep = -1

    def connect(self) -> float:
        """Connect and ask the server to load the requested policy."""
        try:
            import grpc
            from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
            from lerobot.transport import services_pb2, services_pb2_grpc
            from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Remote inference dependencies are missing. Install them with "
                '`.venv/bin/python -m pip install -e ".[remote]"`.'
            ) from exc

        self._grpc = grpc
        self._services_pb2 = services_pb2
        self._send_bytes_in_chunks = send_bytes_in_chunks
        self._timed_observation_class = TimedObservation
        self._channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff="0.2s"),
        )
        started = time.perf_counter()
        try:
            grpc.channel_ready_future(self._channel).result(timeout=self.connect_timeout)
            self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
            self._stub.Ready(services_pb2.Empty(), timeout=self.connect_timeout)
            policy_specs = RemotePolicyConfig(
                policy_type=self.policy_type,
                pretrained_name_or_path=self.policy_path,
                lerobot_features=self.lerobot_features,
                actions_per_chunk=self.actions_per_chunk,
                device=self.policy_device,
                # The transport camera already uses the model's expected name.
                rename_map={},
            )
            self._stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(policy_specs)),  # nosec
                timeout=self.load_timeout,
            )
        except Exception:
            self.close()
            raise
        return time.perf_counter() - started

    def reset(self) -> None:
        """Clear local/server queues while leaving the remotely loaded model resident."""
        if self._stub is None:
            raise RuntimeError("Remote policy client is not connected")
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._queued_actions.clear()
        self._last_executed_timestep = -1
        self._stub.Ready(
            self._services_pb2.Empty(),
            timeout=self.connect_timeout,
        )

    def _raw_observation(
        self,
        positions: dict[str, float],
        rgb: np.ndarray,
        task: str,
    ) -> dict[str, Any]:
        missing = sorted(set(self.joints) - set(positions))
        if missing:
            raise ValueError(f"Follower observation is missing joints: {missing}")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Remote camera observation must be HWC uint8 RGB")
        return {
            **{joint: float(positions[joint]) for joint in self.joints},
            self.camera_name: np.array(rgb, copy=True),
            "task": task,
        }

    def _request_action_chunk(
        self,
        positions: dict[str, float],
        rgb: np.ndarray,
        task: str,
        timestep: int,
    ) -> list[Any]:
        observation = self._timed_observation_class(
            timestamp=time.time(),
            timestep=timestep,
            observation=self._raw_observation(positions, rgb, task),
            must_go=True,
        )
        observation_bytes = pickle.dumps(observation)  # nosec
        chunks = self._send_bytes_in_chunks(
            observation_bytes,
            self._services_pb2.Observation,
            log_prefix="[HEPHA] Observation",
            silent=True,
        )
        self._stub.SendObservations(chunks, timeout=self.request_timeout)
        response = self._stub.GetActions(
            self._services_pb2.Empty(),
            timeout=self.request_timeout,
        )
        if not response.data:
            raise RuntimeError("Remote policy server returned an empty action chunk")
        actions = pickle.loads(response.data)  # nosec: trusted server through SSH tunnel
        if not actions:
            raise RuntimeError("Remote policy server returned no actions")
        return actions

    def _merge_actions(self, actions: list[Any]) -> None:
        for timed_action in actions:
            timestep = int(timed_action.get_timestep())
            if timestep <= self._last_executed_timestep:
                continue
            action = timed_action.get_action().detach().cpu()
            self._queued_actions[timestep] = action

    def _collect_prefetch(self, *, block: bool) -> float:
        if self._future is None or (not block and not self._future.done()):
            return 0.0
        started = time.perf_counter()
        actions = self._future.result(timeout=self.request_timeout if block else None)
        self._future = None
        self._merge_actions(actions)
        return time.perf_counter() - started

    def predict_once(
        self,
        positions: dict[str, float],
        rgb: np.ndarray,
        task: str,
    ) -> tuple[torch.Tensor, float]:
        started = time.perf_counter()
        actions = self._request_action_chunk(positions, rgb, task, 0)
        action = actions[0].get_action().detach().cpu()
        return action, time.perf_counter() - started

    def next_action(
        self,
        positions: dict[str, float],
        rgb: np.ndarray,
        task: str,
    ) -> tuple[torch.Tensor, float, int]:
        """Return one action and asynchronously prefetch the next overlapping chunk."""
        waited = self._collect_prefetch(block=False)
        if not self._queued_actions:
            if self._future is None:
                timestep = max(self._last_executed_timestep, 0)
                self._future = self._executor.submit(
                    self._request_action_chunk,
                    positions.copy(),
                    np.array(rgb, copy=True),
                    task,
                    timestep,
                )
            waited += self._collect_prefetch(block=True)
        if not self._queued_actions:
            raise RuntimeError("Remote policy action queue is empty")

        timestep = min(self._queued_actions)
        action = self._queued_actions.pop(timestep)
        self._last_executed_timestep = timestep

        threshold = int(self.actions_per_chunk * self.prefetch_threshold)
        if len(self._queued_actions) <= threshold and self._future is None:
            self._future = self._executor.submit(
                self._request_action_chunk,
                positions.copy(),
                np.array(rgb, copy=True),
                task,
                max(self._last_executed_timestep, 0),
            )
        return action, waited, len(self._queued_actions)

    def close(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        self._stub = None
        self._executor.shutdown(wait=False, cancel_futures=True)
