from __future__ import annotations

import pickle
from concurrent import futures
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hepha_lerobot.evaluation import physical_rollout
from hepha_lerobot.evaluation.remote_policy_client import (
    HephaRemotePolicyClient,
    build_remote_lerobot_features,
    require_loopback_server,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _policy_config(
    *,
    state_shape=(12,),
    image_shape=(3, 256, 256),
    action_shape=(12,),
    policy_type="act",
    camera_name="head_camera",
):
    return SimpleNamespace(
        type=policy_type,
        max_state_dim=32,
        input_features={
            OBS_STATE: SimpleNamespace(shape=state_shape),
            f"{OBS_IMAGES}.{camera_name}": SimpleNamespace(shape=image_shape),
        },
        output_features={ACTION: SimpleNamespace(shape=action_shape)},
    )


def test_physical_policy_schema_matches_recorded_dataset() -> None:
    physical_rollout.validate_physical_policy_features(
        _policy_config(),
        camera_name="head_camera",
        width=256,
        height=256,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_policy_config(state_shape=(15,)), "observation.state"),
        (_policy_config(image_shape=(3, 480, 640)), "head_camera"),
        (_policy_config(action_shape=(15,)), "action"),
        (_policy_config(policy_type="diffusion"), "ACT and PI0"),
    ],
)
def test_physical_policy_schema_rejects_incompatible_checkpoint(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        physical_rollout.validate_physical_policy_features(
            config,
            camera_name="head_camera",
            width=256,
            height=256,
        )


def test_pi0_schema_accepts_padded_state_and_saved_camera_rename() -> None:
    config = _policy_config(
        state_shape=(32,),
        image_shape=(3, 224, 224),
        policy_type="pi0",
        camera_name="base_0_rgb",
    )
    preprocessor = SimpleNamespace(
        steps=[
            SimpleNamespace(
                rename_map={
                    f"{OBS_IMAGES}.head_camera": f"{OBS_IMAGES}.base_0_rgb",
                }
            )
        ]
    )

    physical_rollout.validate_physical_policy_features(
        config,
        preprocessor=preprocessor,
        camera_name="head_camera",
        width=256,
        height=256,
    )


def test_pi0_schema_rejects_missing_saved_camera_rename() -> None:
    config = _policy_config(
        state_shape=(32,),
        image_shape=(3, 224, 224),
        policy_type="pi0",
        camera_name="base_0_rgb",
    )

    with pytest.raises(ValueError, match="absent from the policy inputs"):
        physical_rollout.validate_physical_policy_features(
            config,
            camera_name="head_camera",
            width=256,
            height=256,
        )


def test_physical_observation_uses_canonical_joint_order_and_rgb() -> None:
    positions = {
        joint: float(index) for index, joint in enumerate(physical_rollout.CANONICAL_JOINTS)
    }
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    observation = physical_rollout.build_physical_policy_observation(
        positions,
        rgb,
        camera_name="head_camera",
    )

    assert observation[OBS_STATE].dtype == np.float32
    assert observation[OBS_STATE].tolist() == list(range(12))
    assert observation[f"{OBS_IMAGES}.head_camera"] is rgb


def test_physical_action_maps_canonical_joint_targets() -> None:
    values = torch.arange(12, dtype=torch.float32).unsqueeze(0)

    targets = physical_rollout.action_targets(values)

    assert list(targets) == list(physical_rollout.CANONICAL_JOINTS)
    assert targets["shoulder_r"] == 0.0
    assert targets["finger_l"] == 11.0


def test_physical_action_rejects_nonfinite_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="action shape"):
        physical_rollout.action_targets(np.zeros(11, dtype=np.float32))
    invalid = np.zeros(12, dtype=np.float32)
    invalid[3] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        physical_rollout.action_targets(invalid)


def test_remote_transport_uses_pi0_camera_and_canonical_joint_order() -> None:
    features = build_remote_lerobot_features(
        physical_rollout.CANONICAL_JOINTS,
        camera_name="base_0_rgb",
        width=256,
        height=256,
    )

    assert features[OBS_STATE]["shape"] == (12,)
    assert features[OBS_STATE]["names"] == list(physical_rollout.CANONICAL_JOINTS)
    assert features[f"{OBS_IMAGES}.base_0_rgb"]["shape"] == (256, 256, 3)


def test_remote_transport_requires_ssh_loopback_tunnel() -> None:
    assert require_loopback_server("127.0.0.1:8080") == ("127.0.0.1", 8080)
    assert require_loopback_server("[::1]:8080") == ("::1", 8080)

    with pytest.raises(ValueError, match="SSH tunnel"):
        require_loopback_server("203.0.113.10:8080")


def test_remote_raw_observation_uses_server_camera_name() -> None:
    client = HephaRemotePolicyClient(
        server_address="127.0.0.1:8080",
        policy_type="pi0",
        policy_path="model",
        policy_device="cuda",
        actions_per_chunk=10,
        prefetch_threshold=0.5,
        camera_name="base_0_rgb",
        width=256,
        height=256,
        joints=physical_rollout.CANONICAL_JOINTS,
        connect_timeout=1.0,
        load_timeout=1.0,
        request_timeout=1.0,
    )
    positions = {joint: float(index) for index, joint in enumerate(client.joints)}
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    observation = client._raw_observation(positions, rgb, "Put the cube in the bowl")

    assert observation["base_0_rgb"].shape == (256, 256, 3)
    assert observation["task"] == "Put the cube in the bowl"
    assert observation["shoulder_r"] == 0.0
    client.close()


def test_remote_client_uses_lerobot_grpc_protocol() -> None:
    grpc = pytest.importorskip("grpc")
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction
    from lerobot.transport import services_pb2, services_pb2_grpc

    class FakePolicyServer(services_pb2_grpc.AsyncInferenceServicer):
        def __init__(self) -> None:
            self.policy_specs = None
            self.observation = None

        def Ready(self, request, context):
            return services_pb2.Empty()

        def SendPolicyInstructions(self, request, context):
            self.policy_specs = pickle.loads(request.data)  # nosec: test fixture
            return services_pb2.Empty()

        def SendObservations(self, request_iterator, context):
            payload = b"".join(request.data for request in request_iterator)
            self.observation = pickle.loads(payload)  # nosec: test fixture
            return services_pb2.Empty()

        def GetActions(self, request, context):
            timestep = self.observation.get_timestep()
            actions = [
                TimedAction(
                    timestamp=0.0,
                    timestep=timestep + index,
                    action=torch.arange(12, dtype=torch.float32) + index,
                )
                for index in range(3)
            ]
            return services_pb2.Actions(data=pickle.dumps(actions))  # nosec: test fixture

    fake = FakePolicyServer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(fake, server)
    try:
        port = server.add_insecure_port("127.0.0.1:0")
    except RuntimeError as exc:
        pytest.skip(f"Loopback sockets are unavailable in this sandbox: {exc}")
    server.start()
    client = HephaRemotePolicyClient(
        server_address=f"127.0.0.1:{port}",
        policy_type="pi0",
        policy_path="remote-model",
        policy_device="cuda",
        actions_per_chunk=3,
        prefetch_threshold=0.0,
        camera_name="base_0_rgb",
        width=32,
        height=32,
        joints=physical_rollout.CANONICAL_JOINTS,
        connect_timeout=2.0,
        load_timeout=2.0,
        request_timeout=2.0,
    )
    try:
        client.connect()
        positions = {joint: 0.0 for joint in client.joints}
        action, _ = client.predict_once(
            positions,
            np.zeros((32, 32, 3), dtype=np.uint8),
            "Put the cube in the bowl",
        )

        assert isinstance(fake.policy_specs, RemotePolicyConfig)
        assert fake.policy_specs.pretrained_name_or_path == "remote-model"
        assert fake.policy_specs.rename_map == {}
        assert fake.observation.get_observation()["task"] == "Put the cube in the bowl"
        assert action.tolist() == list(range(12))
    finally:
        client.close()
        server.stop(grace=0).wait()
