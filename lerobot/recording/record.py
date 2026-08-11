"""Record a native LeRobot dataset from any registered simulation backend."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from hepha_lerobot.datasets import add_robot_frame, create_dataset

from simulation import SimulationConfig, create_backend
from simulation.base import parse_backend_options
from simulation.view import _ensure_mjpython_on_macos, _parse_bool

from .controllers import available_controllers, create_controller

DEFAULT_TASK = (
    "Pick up the cube, open the selected drawer, place the cube inside, "
    "and close the drawer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--controller", default="ik")
    parser.add_argument("--repo-id", default="hepha/simulation_ik")
    parser.add_argument("--root", type=Path, default=Path("datasets/hepha_simulation_ik"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=90.0,
        help="Maximum duration of each physical task attempt",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum attempts used to collect successful episodes (default: 3 per episode)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--viewer",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Open the native simulation viewer while recording",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Show collisions, camera axes, and IK target coordinate systems",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Backend-specific option; may be repeated and accepts JSON values",
    )
    parser.add_argument("--list-controllers", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.episode_seconds <= 0 or args.fps <= 0:
        raise ValueError("--episodes, --episode-seconds, and --fps must be positive")
    if args.max_attempts is not None and args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {args.root}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.root)


def record_dataset(args: argparse.Namespace) -> Path:
    _validate_args(args)
    config = SimulationConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=True,
        viewer=args.viewer,
        debug=args.debug,
        options=parse_backend_options(args.backend_option),
    )
    use_videos = not args.no_video
    with create_backend(args.backend, config) as backend:
        controller = create_controller(args.controller, backend=backend, seed=args.seed)
        dataset = create_dataset(
            backend=backend,
            repo_id=args.repo_id,
            root=args.root,
            fps=args.fps,
            use_videos=use_videos,
        )
        frames_per_attempt = max(2, round(args.episode_seconds * args.fps))
        maximum_attempts = args.max_attempts or args.episodes * 3
        try:
            saved_episodes = 0
            for attempt in range(maximum_attempts):
                if saved_episodes >= args.episodes:
                    break
                attempt_seed = args.seed + attempt
                controller.reset(episode_seed=attempt)
                status = getattr(controller, "status", "controller reset")
                print(
                    f"Starting attempt {attempt + 1}/{maximum_attempts} "
                    f"with seed {attempt_seed}: {status}"
                )
                recorded_frames = 0
                for frame_index in range(frames_per_attempt):
                    progress = frame_index / (frames_per_attempt - 1)
                    observation = backend.get_observation(advance=False)
                    action = backend.send_action(controller.action(progress))
                    add_robot_frame(
                        dataset,
                        observation=observation,
                        action=action,
                        task=args.task,
                    )
                    backend.step()
                    recorded_frames += 1
                    if getattr(controller, "done", False):
                        break

                has_terminal_check = hasattr(controller, "successful")
                attempt_succeeded = (
                    bool(getattr(controller, "done", False))
                    and bool(getattr(controller, "successful", False))
                    if has_terminal_check
                    else True
                )
                if not attempt_succeeded:
                    dataset.clear_episode_buffer()
                    status = getattr(controller, "status", "attempt timed out")
                    print(
                        f"Discarded attempt {attempt + 1}/{maximum_attempts}: {status} "
                        f"({recorded_frames} frames)"
                    )
                    continue

                dataset.save_episode()
                saved_episodes += 1
                print(
                    f"Saved episode {saved_episodes}/{args.episodes} "
                    f"with {args.backend}/{args.controller} ({recorded_frames} frames)"
                )

            if saved_episodes < args.episodes:
                raise RuntimeError(
                    f"Collected {saved_episodes}/{args.episodes} successful episodes after "
                    f"{maximum_attempts} attempts; increase --max-attempts or inspect "
                    "the simulation."
                )
        finally:
            dataset.finalize()
        if args.push_to_hub:
            dataset.push_to_hub(tags=["hepha", args.backend, args.controller])
        return dataset.root


def main() -> None:
    args = parse_args()
    if args.list_controllers:
        print("\n".join(available_controllers(args.backend)))
        return
    if args.viewer and args.backend == "mujoco":
        _ensure_mjpython_on_macos("hepha_lerobot.recording.record")
    dataset_root = record_dataset(args)
    print(f"LeRobot dataset ready at {dataset_root}")


if __name__ == "__main__":
    main()
