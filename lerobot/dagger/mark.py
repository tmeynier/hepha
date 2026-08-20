"""Watch policy rollouts and save intervention states by pressing Space."""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import suppress
from pathlib import Path

import torch
from hepha_lerobot.conditioning import DEFAULT_TASK, drawer_task, validate_task_phase
from hepha_lerobot.dagger.snapshot import InterventionMetadata, save_snapshot
from hepha_lerobot.evaluation.conditioned_rollout import (
    _load_policy,
    _policy_observation,
)
from hepha_lerobot.evaluation.phase_control import PhaseTransitionState
from hepha_lerobot.training.train import resolve_device
from lerobot.policies.utils import prepare_observation_for_inference

from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import initialize_task_episode
from simulation.view import _ensure_mjpython_on_macos

SPACE_KEY = 32
NEXT_KEY = ord("N")
QUIT_KEY = ord("Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dagger/interventions"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=100.0)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--task-phase", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    parser.add_argument("--domain-randomization-scale", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.policy_path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")
    if args.episodes <= 0 or args.episode_seconds <= 0 or args.fps <= 0:
        raise ValueError("--episodes, --episode-seconds, and --fps must be positive")
    if args.domain_randomization_scale < 0.0:
        raise ValueError("--domain-randomization-scale must be non-negative")
    validate_task_phase(args.task_phase)


def run(args: argparse.Namespace) -> list[Path]:
    _validate_args(args)
    device = resolve_device(args.device)
    policy, policy_config, preprocessor, postprocessor = _load_policy(
        args.policy_path,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    if policy_config.type == "hepha_act_phase" and policy_config.n_action_steps != 1:
        raise ValueError("Phase-aware intervention marking requires --n-action-steps 1")

    config = SimulationConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=True,
        viewer=False,
        debug=args.debug,
        options={"domain_randomization_scale": args.domain_randomization_scale},
    )
    command_lock = threading.Lock()
    pending_command: list[str | None] = [None]
    saved: list[Path] = []

    def on_key(keycode: int) -> None:
        command = {
            SPACE_KEY: "mark",
            NEXT_KEY: "skip",
            QUIT_KEY: "quit",
        }.get(keycode)
        if command is not None:
            with command_lock:
                pending_command[0] = command

    def read_command() -> str | None:
        with command_lock:
            return pending_command[0]

    def clear_command() -> None:
        with command_lock:
            pending_command[0] = None

    with MujocoBackend(config) as backend:
        backend.open_viewer(debug=args.debug, key_callback=on_key)
        state_names = tuple(
            name
            for name, feature_type in backend.observation_features.items()
            if feature_type is float
        )
        max_steps = round(args.episode_seconds * args.fps)
        interval = 1.0 / args.fps
        print(
            "Controls: Space=mark state, N=skip good episode, Q=finish. "
            "Closing the viewer also finishes.",
            flush=True,
        )

        for episode_index in range(args.episodes):
            if not backend.viewer_is_running():
                break
            seed = args.seed_start + episode_index
            rng = initialize_task_episode(backend, seed=seed)
            drawer_index = int(rng.integers(1, 10))
            task = drawer_task(args.task, drawer_index)
            phase_state = (
                PhaseTransitionState(current_phase=args.task_phase)
                if policy_config.type == "hepha_act_phase"
                else None
            )
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            clear_command()
            marked_step: int | None = None
            print(
                f"Episode {episode_index + 1}/{args.episodes}: seed={seed}, "
                f"drawer={drawer_index}",
                flush=True,
            )

            for step in range(max_steps):
                started = time.perf_counter()
                command = read_command()
                if command is not None or not backend.viewer_is_running():
                    marked_step = step if command == "mark" else None
                    break
                raw_observation = backend.get_observation()
                command = read_command()
                if command is not None:
                    marked_step = step if command == "mark" else None
                    break
                current_phase = (
                    phase_state.current_phase if phase_state is not None else args.task_phase
                )
                observation = _policy_observation(
                    raw_observation,
                    state_names=state_names,
                    camera=args.camera,
                    drawer_index=drawer_index,
                    current_phase=(current_phase if phase_state is not None else None),
                )
                batch = prepare_observation_for_inference(
                    observation,
                    torch.device(device),
                    task=task,
                    robot_type="hepha_mujoco",
                )
                with torch.inference_mode():
                    batch = preprocessor(batch)
                    if phase_state is not None:
                        action, phase_logits = policy.select_action_with_phase(batch)
                        phase_state.observe(int(phase_logits.argmax(dim=-1).item()) + 1)
                    else:
                        action = policy.select_action(batch)
                    action = postprocessor(action)
                backend.send_action(action.squeeze(0).detach().cpu().numpy())
                remaining = interval - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

            command = read_command()
            if command == "quit" or not backend.viewer_is_running():
                print("Finished intervention marking.", flush=True)
                break
            if command == "skip":
                print(f"Skipped episode seed={seed} without saving.", flush=True)
                continue
            if marked_step is None:
                print(f"Episode seed={seed} ended without an intervention mark.", flush=True)
                continue
            current_phase = (
                phase_state.current_phase if phase_state is not None else args.task_phase
            )
            metadata = InterventionMetadata.create(
                seed=seed,
                source_episode=episode_index,
                source_step=marked_step,
                drawer_index=drawer_index,
                current_task_phase=current_phase,
                next_task_phase=current_phase,
                fps=args.fps,
                camera=args.camera,
                task=task,
                policy_path=str(args.policy_path.resolve()),
                policy_type=policy_config.type,
                domain_randomization_scale=args.domain_randomization_scale,
            )
            directory = save_snapshot(args.output_dir, backend=backend, metadata=metadata)
            saved.append(directory)
            print(f"Saved intervention: {directory}", flush=True)

    print(f"Saved {len(saved)} intervention state(s) in {args.output_dir}")
    return saved


def main() -> None:
    args = parse_args()
    _ensure_mjpython_on_macos("hepha_lerobot.dagger.mark")
    with suppress(KeyboardInterrupt):
        run(args)


if __name__ == "__main__":
    main()
