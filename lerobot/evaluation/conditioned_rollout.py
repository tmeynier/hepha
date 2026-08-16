"""Run a LeRobot policy with an explicit requested-drawer condition."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from hepha_lerobot.conditioning import drawer_condition, drawer_task
from hepha_lerobot.training.train import resolve_device
from lerobot.configs import PreTrainedConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)

from lerobot.policies import get_policy_class, make_pre_post_processors
from simulation import SimulationConfig, create_backend
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import initialize_task_episode
from simulation.base import parse_backend_options
from simulation.view import _parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path", type=Path)
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--task", required=True)
    parser.add_argument("--drawer-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--viewer", type=_parse_bool, default=True)
    parser.add_argument("--debug", type=_parse_bool, default=False)
    parser.add_argument("--display-data", type=_parse_bool, default=False)
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser.parse_args()


def _apply_inference_overrides(
    policy_config,
    *,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
) -> None:
    if n_action_steps is not None:
        if n_action_steps <= 0:
            raise ValueError("--n-action-steps must be positive")
        if not hasattr(policy_config, "n_action_steps"):
            raise ValueError(
                f"Policy {policy_config.type!r} does not support --n-action-steps"
            )
        chunk_size = getattr(policy_config, "chunk_size", None)
        if chunk_size is not None and n_action_steps > chunk_size:
            raise ValueError(
                f"--n-action-steps ({n_action_steps}) cannot exceed the policy "
                f"chunk size ({chunk_size})"
            )
        policy_config.n_action_steps = n_action_steps

    if temporal_ensemble_coeff is None:
        return
    if not math.isfinite(temporal_ensemble_coeff) or temporal_ensemble_coeff < 0.0:
        raise ValueError("--temporal-ensemble-coeff must be finite and non-negative")
    if policy_config.type != "act" or not hasattr(
        policy_config, "temporal_ensemble_coeff"
    ):
        raise ValueError("--temporal-ensemble-coeff is only supported for ACT")
    effective_action_steps = getattr(policy_config, "n_action_steps", None)
    if effective_action_steps != 1:
        raise ValueError("Temporal ensembling requires --n-action-steps 1")
    policy_config.temporal_ensemble_coeff = temporal_ensemble_coeff


def _load_policy(
    policy_path: Path,
    device: str,
    *,
    n_action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
):
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.device = device
    policy_config.pretrained_path = policy_path
    _apply_inference_overrides(
        policy_config,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(policy_path, config=policy_config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        pretrained_revision=policy_config.pretrained_revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, policy_config, preprocessor, postprocessor


def _policy_observation(
    raw_observation: dict,
    *,
    state_names: tuple[str, ...],
    camera: str,
    drawer_index: int,
) -> dict[str, np.ndarray]:
    return {
        OBS_STATE: np.asarray(
            [raw_observation[name] for name in state_names], dtype=np.float32
        ),
        OBS_ENV_STATE: drawer_condition(drawer_index),
        f"{OBS_IMAGES}.{camera}": np.asarray(raw_observation[camera]),
    }


def run(args: argparse.Namespace) -> None:
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("--duration and --fps must be positive")
    device = resolve_device(args.device)
    task = drawer_task(args.task, args.drawer_index)
    policy, policy_config, preprocessor, postprocessor = _load_policy(
        args.policy_path,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    if OBS_ENV_STATE not in (policy_config.input_features or {}):
        raise ValueError(
            "The policy does not expect observation.environment_state. Train it on "
            "a drawer-conditioned Hepha dataset before using this evaluator."
        )
    if hasattr(policy_config, "n_action_steps"):
        print(
            "Policy inference: "
            f"n_action_steps={policy_config.n_action_steps}, "
            "temporal_ensemble_coeff="
            f"{getattr(policy_config, 'temporal_ensemble_coeff', None)}",
            flush=True,
        )

    simulation_config = SimulationConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=True,
        viewer=args.viewer,
        debug=args.debug,
        options=parse_backend_options(args.backend_option),
    )
    visualizing = bool(args.display_data)
    if visualizing:
        init_visualization("rerun", session_name="hepha_conditioned_rollout")

    try:
        with create_backend(args.backend, simulation_config) as backend:
            if isinstance(backend, MujocoBackend):
                # Match the exact pre-demonstration state used by the recorder,
                # including seeded cube placement. No IK controller is created:
                # every action after this initialization comes from the policy.
                initialize_task_episode(backend, seed=args.seed)
            else:
                backend.reset(seed=args.seed)
            state_names = tuple(
                name
                for name, feature_type in backend.observation_features.items()
                if feature_type is float
            )
            action_names = tuple(backend.action_features)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            deadline = time.perf_counter() + args.duration
            control_interval = 1.0 / args.fps

            while time.perf_counter() < deadline and backend.is_open:
                started = time.perf_counter()
                raw_observation = backend.get_observation()
                observation = _policy_observation(
                    raw_observation,
                    state_names=state_names,
                    camera=args.camera,
                    drawer_index=args.drawer_index,
                )
                batch = prepare_observation_for_inference(
                    observation,
                    torch.device(device),
                    task=task,
                    robot_type=f"hepha_{backend.name}",
                )
                with torch.inference_mode():
                    batch = preprocessor(batch)
                    action = postprocessor(policy.select_action(batch))
                values = action.squeeze(0).detach().cpu().numpy()
                if values.shape != (len(action_names),):
                    raise ValueError(
                        f"Policy returned action shape {values.shape}; expected "
                        f"({len(action_names)},)"
                    )
                action_dict = dict(zip(action_names, values.tolist(), strict=True))
                backend.send_action(action_dict)
                if visualizing:
                    log_visualization_data(
                        "rerun", observation=raw_observation, action=action_dict
                    )
                remaining = control_interval - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if visualizing:
            shutdown_visualization("rerun")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
