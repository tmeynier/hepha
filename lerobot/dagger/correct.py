"""Record human DAgger corrections from saved MuJoCo intervention states."""

from __future__ import annotations

import argparse
import shutil
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hepha_lerobot.conditioning import validate_task_phase
from hepha_lerobot.dagger.snapshot import (
    InterventionSnapshot,
    discover_snapshots,
    restore_snapshot,
)
from hepha_lerobot.datasets import add_robot_frame, create_dataset
from scipy.signal import savgol_filter

from lerobot.datasets import LeRobotDataset, merge_datasets
from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.view import _ensure_mjpython_on_macos

SPACE_KEY = 32
NEXT_KEY = ord("N")
RETRY_KEY = ord("R")
QUIT_KEY = ord("Q")


@dataclass(frozen=True)
class ManualFrame:
    action: np.ndarray
    task_phase: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intervention_dir", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--smoothing-order", type=int, default=2)
    parser.add_argument("--current-phase", type=int, default=None)
    parser.add_argument("--next-phase", type=int, default=None)
    parser.add_argument(
        "--dataset-schema",
        choices=("phase", "drawer"),
        default="phase",
        help="Use 'drawer' when extending an original ACT dataset without phase fields",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse independently finalized correction shards from an interrupted run",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0 or args.max_seconds <= 0:
        raise ValueError("--fps and --max-seconds must be positive")
    if args.smoothing_window <= 0 or args.smoothing_order < 0:
        raise ValueError("Smoothing window must be positive and order non-negative")
    if args.current_phase is not None:
        validate_task_phase(args.current_phase)
    if args.next_phase is not None:
        validate_task_phase(args.next_phase)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")


def smooth_action_sequence(
    actions: np.ndarray,
    *,
    window: int,
    order: int,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Smooth uniformly sampled joint targets without changing frame count."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected a [frames, actions] array, got {actions.shape}")
    usable_window = min(window, actions.shape[0])
    if usable_window % 2 == 0:
        usable_window -= 1
    if usable_window <= order or usable_window < 3:
        return np.clip(actions, low, high)
    smoothed = savgol_filter(
        actions,
        window_length=usable_window,
        polyorder=order,
        axis=0,
        mode="interp",
    )
    return np.clip(smoothed, low, high)


def _shard_root(dataset_root: Path) -> Path:
    return dataset_root.parent / f".{dataset_root.name}.dagger-shards"


def _prepare_storage(args: argparse.Namespace, shard_root: Path) -> None:
    if args.resume:
        if not shard_root.is_dir():
            raise FileNotFoundError(
                f"No interrupted correction session found at {shard_root}"
            )
        if args.root.exists():
            shutil.rmtree(args.root)
        return

    existing = [path for path in (args.root, shard_root) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"DAgger output already exists: {paths}. Use --overwrite to restart or "
            "--resume to keep completed corrections."
        )
    if args.overwrite:
        for path in existing:
            shutil.rmtree(path)
    shard_root.mkdir(parents=True, exist_ok=True)


def _load_completed_shards(
    shard_root: Path,
) -> dict[str, LeRobotDataset]:
    completed: dict[str, LeRobotDataset] = {}
    for directory in sorted(shard_root.glob("mark-*")):
        if not directory.is_dir():
            continue
        repo_id_path = directory / "dagger_repo_id.txt"
        try:
            repo_id = repo_id_path.read_text(encoding="utf-8").strip()
            dataset = LeRobotDataset(repo_id=repo_id, root=directory)
            if dataset.meta.total_episodes != 1:
                raise ValueError("A correction shard must contain exactly one episode")
        except Exception as error:
            print(f"Removing incomplete correction shard {directory}: {error}")
            shutil.rmtree(directory)
            continue
        completed[directory.name] = dataset
    return completed


def _save_correction_shard(
    *,
    backend: MujocoBackend,
    snapshot: InterventionSnapshot,
    frames: list[ManualFrame],
    actions: np.ndarray,
    args: argparse.Namespace,
    shard_root: Path,
) -> LeRobotDataset:
    directory = shard_root / snapshot.directory.name
    repo_id = f"{args.repo_id}-correction-{snapshot.directory.name}"
    if directory.exists():
        raise FileExistsError(f"Correction shard already exists: {directory}")
    dataset = create_dataset(
        backend=backend,
        repo_id=repo_id,
        root=directory,
        fps=args.fps,
        use_videos=not args.no_video,
        include_task_phase=args.dataset_schema == "phase",
    )
    try:
        restore_snapshot(backend, snapshot)
        print(
            f"Replaying and recording smoothed correction {snapshot.directory.name}...",
            flush=True,
        )
        for frame_index, (frame, action) in enumerate(
            zip(frames, actions, strict=True)
        ):
            next_phase = (
                validate_task_phase(args.next_phase)
                if args.next_phase is not None
                else frames[min(frame_index + 1, len(frames) - 1)].task_phase
            )
            observation = backend.get_observation(advance=False)
            expert_action = backend.send_action(action)
            add_robot_frame(
                dataset,
                observation=observation,
                action=expert_action,
                task=snapshot.metadata.task,
                drawer_index=snapshot.metadata.drawer_index,
                current_task_phase=frame.task_phase,
                next_task_phase=next_phase,
            )
            backend.step()
        dataset.save_episode()
        dataset.finalize()
        (directory / "dagger_repo_id.txt").write_text(repo_id + "\n", encoding="utf-8")
    except BaseException:
        with suppress(Exception):
            episode_buffer = getattr(dataset.writer, "episode_buffer", None)
            if isinstance(episode_buffer, dict) and episode_buffer.get("size", 0) > 0:
                dataset.clear_episode_buffer()
            dataset.finalize()
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return LeRobotDataset(repo_id=repo_id, root=directory)


def run(args: argparse.Namespace) -> Path | None:
    _validate_args(args)
    snapshots = discover_snapshots(args.intervention_dir)
    randomization_scales = {
        snapshot.metadata.domain_randomization_scale for snapshot in snapshots
    }
    cameras = {snapshot.metadata.camera for snapshot in snapshots}
    if len(randomization_scales) != 1 or len(cameras) != 1:
        raise ValueError(
            "All snapshots in one correction dataset must share camera and "
            "domain-randomization settings"
        )
    shard_root = _shard_root(args.root)
    _prepare_storage(args, shard_root)
    completed = _load_completed_shards(shard_root)
    camera = next(iter(cameras))
    config = SimulationConfig(
        camera=camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=True,
        viewer=False,
        debug=args.debug,
        options={"domain_randomization_scale": next(iter(randomization_scales))},
    )
    command_lock = threading.Lock()
    pending_command: list[str | None] = [None]
    selected_phase = [1]

    def on_key(keycode: int) -> None:
        command = {
            SPACE_KEY: "save",
            NEXT_KEY: "skip",
            RETRY_KEY: "retry",
            QUIT_KEY: "quit",
        }.get(keycode)
        if command is not None:
            with command_lock:
                pending_command[0] = command
        elif ord("1") <= keycode <= ord("5"):
            selected_phase[0] = keycode - ord("0")
            print(f"Manual task phase: {selected_phase[0]}", flush=True)

    def read_command() -> str | None:
        with command_lock:
            return pending_command[0]

    def clear_command() -> None:
        with command_lock:
            pending_command[0] = None

    stop_requested = False
    with MujocoBackend(config) as backend:
        backend.open_viewer(debug=args.debug, key_callback=on_key)
        print(
            "Controls: Space=save correction, N=skip state, R=retry state, "
            "Q=finish. Use the native Control panel for joints; keys 1-5 change "
            "the semantic phase.",
            flush=True,
        )
        for correction_index, snapshot in enumerate(snapshots):
            if snapshot.directory.name in completed:
                print(
                    f"Correction {correction_index + 1}/{len(snapshots)} already saved: "
                    f"{snapshot.directory.name}",
                    flush=True,
                )
                continue
            if not backend.viewer_is_running():
                break

            while backend.viewer_is_running():
                restore_snapshot(backend, snapshot)
                clear_command()
                selected_phase[0] = validate_task_phase(
                    args.current_phase
                    if args.current_phase is not None
                    else snapshot.metadata.current_task_phase
                )
                frames: list[ManualFrame] = []
                interval = 1.0 / args.fps
                max_steps = round(args.max_seconds * args.fps)
                print(
                    f"Correction {correction_index + 1}/{len(snapshots)}: "
                    f"{snapshot.directory.name}, seed={snapshot.metadata.seed}, "
                    f"drawer={snapshot.metadata.drawer_index}",
                    flush=True,
                )

                for _ in range(max_steps):
                    started = time.perf_counter()
                    if read_command() is not None or not backend.viewer_is_running():
                        break
                    action = backend.data.ctrl[backend.actuator_ids].astype(
                        np.float64, copy=True
                    )
                    frames.append(
                        ManualFrame(action=action, task_phase=selected_phase[0])
                    )
                    backend.step()
                    remaining = interval - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)

                command = read_command()
                if not backend.viewer_is_running() or command == "quit":
                    stop_requested = True
                    print("Finishing correction collection.", flush=True)
                    break
                if command == "skip":
                    print(
                        f"Skipped {snapshot.directory.name} without saving.", flush=True
                    )
                    break
                if command == "retry":
                    print(f"Retrying {snapshot.directory.name} from its saved state.")
                    continue
                if command != "save":
                    print(
                        f"Correction {snapshot.directory.name} timed out; retrying it. "
                        "Press N to skip or Q to finish.",
                        flush=True,
                    )
                    continue
                if len(frames) < 2:
                    print(
                        f"Correction {snapshot.directory.name} was too short; retrying.",
                        flush=True,
                    )
                    continue

                actions = smooth_action_sequence(
                    np.stack([frame.action for frame in frames]),
                    window=args.smoothing_window,
                    order=args.smoothing_order,
                    low=backend.control_low,
                    high=backend.control_high,
                )
                dataset = _save_correction_shard(
                    backend=backend,
                    snapshot=snapshot,
                    frames=frames,
                    actions=actions,
                    args=args,
                    shard_root=shard_root,
                )
                completed[snapshot.directory.name] = dataset
                print(
                    f"Durably saved {snapshot.directory.name}: {len(frames)} frames",
                    flush=True,
                )
                break

            if stop_requested:
                break

    completed = _load_completed_shards(shard_root)
    if not completed:
        print("No completed human correction episodes were saved.")
        return None
    if args.root.exists():
        shutil.rmtree(args.root)
    merged = merge_datasets(
        list(completed.values()),
        output_repo_id=args.repo_id,
        output_dir=args.root,
        concatenate_videos=False,
        concatenate_data=False,
    )
    if args.push_to_hub:
        merged.push_to_hub(tags=["hepha", "mujoco", "dagger", "human-correction"])
    print(
        f"LeRobot DAgger dataset ready at {args.root} "
        f"({merged.meta.total_episodes} episodes)"
    )
    print(f"Independent correction shards retained at {shard_root}")
    return args.root


def main() -> None:
    args = parse_args()
    _ensure_mjpython_on_macos("hepha_lerobot.dagger.correct")
    with suppress(KeyboardInterrupt):
        run(args)


if __name__ == "__main__":
    main()
