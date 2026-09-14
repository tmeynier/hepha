"""Record MuJoCo episodes interactively from a calibrated Feetech leader."""

from __future__ import annotations

import argparse
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from hepha_lerobot.conditioning import DEFAULT_TASK, drawer_task, validate_drawer_index
from hepha_lerobot.datasets import add_robot_frame, create_dataset

from hardware.feetech_leader import FeetechLeader, configure_cnc_start_pose
from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.backends.mujoco.episode import initialize_task_episode
from simulation.base import SimulationConfig
from simulation.view import _ensure_mjpython_on_macos

EPISODE_DECISIONS = ("SAVE", "DISCARD", "RETRY", "QUIT")


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def smoothing_factor(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("smoothing must be greater than 0 and at most 1")
    return parsed


def drawer_index(value: str) -> int:
    try:
        return validate_drawer_index(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="hepha/mujoco_feetech_teleop")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/hepha_mujoco_feetech_teleop"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("hardware/feetech_calibration.json"),
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        help="Calibrated leader IDs (default: every ID in the calibration file).",
    )
    parser.add_argument("--port", help="Override the serial port stored in calibration.")
    parser.add_argument("--episodes", type=positive_integer, default=20)
    parser.add_argument("--episode-seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=positive_integer, default=30)
    parser.add_argument("--width", type=positive_integer, default=256)
    parser.add_argument("--height", type=positive_integer, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--drawer-index",
        type=drawer_index,
        help="Fixed requested drawer 1-9 (default: deterministically sampled each attempt).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--countdown", type=nonnegative_float, default=3.0)
    parser.add_argument("--smoothing", type=smoothing_factor, default=0.25)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the native MuJoCo viewer (default: true).",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episode_seconds <= 0:
        raise ValueError("--episode-seconds must be greater than zero")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    if args.debug and not args.viewer:
        raise ValueError("--debug requires --viewer")
    if args.root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Dataset root already exists: {args.root}. Use --overwrite to replace it."
        )


def prepare_dataset_root(root: Path, *, overwrite: bool) -> None:
    """Create a clean recording target only after the operator has confirmed."""
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset root already exists: {root}.")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)


def parse_episode_decision(value: str) -> str:
    decision = value.strip().upper()
    if decision not in EPISODE_DECISIONS:
        choices = ", ".join(EPISODE_DECISIONS)
        raise ValueError(f"Choose one of: {choices}")
    return decision


def prompt_episode_decision() -> str:
    while True:
        try:
            return parse_episode_decision(
                input("Save this attempt? Type SAVE, DISCARD, RETRY, or QUIT: ")
            )
        except ValueError as exc:
            print(exc)


def _episode_buffer_has_frames(dataset: Any) -> bool:
    buffer = getattr(getattr(dataset, "writer", None), "episode_buffer", None)
    return isinstance(buffer, dict) and int(buffer.get("size", 0)) > 0


def clear_unsaved_episode(dataset: Any) -> None:
    if _episode_buffer_has_frames(dataset):
        dataset.clear_episode_buffer()


def base_teleoperation_action(backend: MujocoBackend) -> np.ndarray:
    action = backend.home_action()
    return configure_cnc_start_pose(action, backend.control_low, ACTUATOR_NAMES)


def initialize_from_leader(
    backend: MujocoBackend,
    leader: FeetechLeader,
    *,
    seed: int,
    fixed_drawer_index: int | None,
) -> tuple[np.ndarray, int]:
    """Reset the task and put controlled joints directly at the physical pose."""
    rng = initialize_task_episode(backend, seed=seed)
    selected_drawer = (
        validate_drawer_index(fixed_drawer_index)
        if fixed_drawer_index is not None
        else int(rng.integers(1, 10))
    )
    base_action = base_teleoperation_action(backend)
    leader.reset_filter()
    initial_action = leader.read_action(base_action, ACTUATOR_NAMES)
    backend.data.qpos[backend.qpos_ids] = initial_action
    dof_ids = backend.model.jnt_dofadr[backend.joint_ids].astype(int)
    backend.data.qvel[dof_ids] = 0.0
    backend.send_action(initial_action)
    mujoco.mj_forward(backend.model, backend.data)
    backend.sync_viewer()
    return base_action, selected_drawer


def countdown(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print(f"\rRecording starts in {remaining:4.1f} s", end="", flush=True)
        time.sleep(min(0.1, remaining))
    if seconds:
        print("\rRecording started.                 ")


def record_attempt(
    *,
    backend: MujocoBackend,
    leader: FeetechLeader,
    dataset: Any,
    base_action: np.ndarray,
    drawer: int,
    task: str,
    fps: int,
    episode_seconds: float,
    viewer_required: bool,
) -> int:
    """Record one fixed-duration attempt and return its frame count."""
    frame_count = max(1, round(episode_seconds * fps))
    period = 1.0 / fps
    episode_task = drawer_task(task, drawer)
    previous_status_width = 0
    for frame_index in range(frame_count):
        if viewer_required and not backend.viewer_is_running():
            raise KeyboardInterrupt("MuJoCo viewer was closed")
        started = time.perf_counter()
        observation = backend.get_observation(advance=False)
        requested_action = leader.read_action(base_action, ACTUATOR_NAMES)
        sent_action = backend.send_action(requested_action)
        add_robot_frame(
            dataset,
            observation=observation,
            action=sent_action,
            task=episode_task,
            drawer_index=drawer,
            current_task_phase=None,
            next_task_phase=None,
        )
        backend.step()

        status = (
            f"frame {frame_index + 1:>5}/{frame_count}  "
            f"elapsed={(frame_index + 1) / fps:6.1f}/{episode_seconds:.1f} s"
        )
        print(f"\r{status:<{previous_status_width}}", end="", flush=True)
        previous_status_width = max(previous_status_width, len(status))
        time.sleep(max(0.0, period - (time.perf_counter() - started)))
    print()
    return frame_count


def record_dataset(args: argparse.Namespace) -> Path:
    validate_args(args)
    simulation_config = SimulationConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=True,
        viewer=False,
        debug=args.debug,
    )
    leader = FeetechLeader(
        calibration_path=args.calibration,
        servo_ids=args.ids,
        port=args.port,
        retries=args.retries,
        smoothing=args.smoothing,
    )

    with MujocoBackend(simulation_config) as backend:
        leader.validate_mujoco_ranges(
            ACTUATOR_NAMES,
            backend.control_low,
            backend.control_high,
        )
        with leader:
            print("\nPhysical Feetech leader -> MuJoCo episode recording")
            for servo in leader.servos:
                print(f"ID {servo.servo_id}: {servo.label} -> {servo.actuator}")
            print(f"Port: {leader.port}   Baud: {leader.baudrate:,}")
            print("The physical servos are read only; no position commands are sent to them.")
            print("MuJoCo CNC pose: cnc_x=0 m, cnc_y=0 m, head_z=minimum.")
            input("Support all selected axes, then press ENTER to disable their torque: ")
            leader.disable_torque()
            prepare_dataset_root(args.root, overwrite=args.overwrite)

            dataset = create_dataset(
                backend=backend,
                repo_id=args.repo_id,
                root=args.root,
                fps=args.fps,
                use_videos=not args.no_video,
                include_task_phase=False,
            )
            saved_episodes = 0
            attempt_index = 0
            try:
                while saved_episodes < args.episodes:
                    attempt_seed = args.seed + attempt_index
                    base_action, selected_drawer = initialize_from_leader(
                        backend,
                        leader,
                        seed=attempt_seed,
                        fixed_drawer_index=args.drawer_index,
                    )
                    if args.viewer and not backend.viewer_is_running():
                        backend.open_viewer(debug=args.debug)
                    print(
                        f"\nEpisode {saved_episodes + 1}/{args.episodes}; "
                        f"attempt seed={attempt_seed}; requested drawer={selected_drawer}"
                    )
                    input("Arrange the leader at the desired start pose, then press ENTER: ")
                    # Re-read after the operator has positioned the leader.
                    base_action, selected_drawer = initialize_from_leader(
                        backend,
                        leader,
                        seed=attempt_seed,
                        fixed_drawer_index=selected_drawer,
                    )
                    countdown(args.countdown)
                    frames = record_attempt(
                        backend=backend,
                        leader=leader,
                        dataset=dataset,
                        base_action=base_action,
                        drawer=selected_drawer,
                        task=args.task,
                        fps=args.fps,
                        episode_seconds=args.episode_seconds,
                        viewer_required=args.viewer,
                    )
                    decision = prompt_episode_decision()
                    if decision == "SAVE":
                        dataset.save_episode()
                        saved_episodes += 1
                        attempt_index += 1
                        print(f"Saved episode {saved_episodes} ({frames} frames).")
                    else:
                        clear_unsaved_episode(dataset)
                        if decision == "QUIT":
                            break
                        if decision == "DISCARD":
                            attempt_index += 1
                            print("Discarded attempt; advancing to a new randomized task.")
                        else:
                            print("Discarded attempt; retrying the same randomized task.")
            finally:
                active_error = sys.exception()
                try:
                    clear_unsaved_episode(dataset)
                    dataset.finalize()
                except Exception:
                    if active_error is None:
                        raise

            if args.push_to_hub and saved_episodes:
                dataset.push_to_hub(tags=["hepha", "mujoco", "feetech", "teleoperation"])
            print(f"Recorded {saved_episodes} episode(s) at {args.root}.")
            return args.root


def main() -> None:
    args = parse_args()
    if args.viewer:
        _ensure_mjpython_on_macos("hepha_lerobot.recording.teleop")
    try:
        dataset_root = record_dataset(args)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nStopped. The current unsaved attempt was discarded.")
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Teleoperation recording failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"LeRobot teleoperation dataset ready at {dataset_root}")


if __name__ == "__main__":
    main()
