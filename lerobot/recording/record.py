"""Record a native LeRobot dataset from any registered simulation backend."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hepha_lerobot.conditioning import (
    DEFAULT_TASK,
    drawer_task,
    validate_task_phase,
)
from hepha_lerobot.datasets import add_robot_frame, create_dataset

from lerobot.datasets import LeRobotDataset, merge_datasets
from simulation import SimulationConfig, create_backend
from simulation.base import parse_backend_options
from simulation.view import _ensure_mjpython_on_macos, _parse_bool

from .controllers import available_controllers, create_controller

CUBE_QUADRANTS = (
    "upper_left",
    "upper_right",
    "bottom_left",
    "bottom_right",
)


@dataclass(frozen=True)
class _WorkerConfig:
    backend: str
    controller: str
    repo_id: str
    shard_root: Path
    episode_seconds: float
    fps: int
    width: int
    height: int
    camera: str
    task: str
    base_seed: int
    target_episodes: int
    maximum_attempts: int
    use_videos: bool
    deferred_rendering: bool
    viewer: bool
    debug: bool
    backend_options: dict[str, Any]
    image_writer_processes: int
    image_writer_threads: int
    streaming_encoding: bool
    encoder_threads: int | None


@dataclass(frozen=True)
class _AttemptResult:
    attempt_index: int
    seed: int
    succeeded: bool
    saved: bool
    frames: int
    status: str
    drawer_index: int | None
    cube_quadrant: str | None


@dataclass(frozen=True)
class _DeferredFrame:
    observation: dict[str, Any]
    render_state: Any
    action: dict[str, Any]
    current_task_phase: int
    next_task_phase: int


@dataclass(frozen=True)
class _WorkerResult:
    worker_id: int
    repo_id: str
    root: Path
    saved_episodes: int
    attempts: tuple[_AttemptResult, ...]


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
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Independent episode-generation processes (default: 2)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Absolute seed of the first attempted episode",
    )
    parser.add_argument(
        "--viewer",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Open the native viewer; only supported with --workers 1",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Show collisions, camera axes, and IK targets in the optional viewer",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--deferred-rendering",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        metavar="BOOL",
        help="Render and write camera frames only after an attempt succeeds (default: true)",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="LeRobot background image-writer processes per recording worker",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=2,
        help="LeRobot background image-writer threads per process (default: 2)",
    )
    parser.add_argument(
        "--video-encoding-batch-size",
        type=int,
        default=1,
        help=(
            "Retained for command compatibility; pinned LeRobot 0.6.1 worker "
            "shards always use the safe value 1"
        ),
    )
    parser.add_argument(
        "--streaming-encoding",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Use LeRobot streaming video encoding instead of temporary frame images",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=2,
        help="Video encoder threads per recording worker (default: 2)",
    )
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
    positive_values = {
        "--episodes": args.episodes,
        "--episode-seconds": args.episode_seconds,
        "--fps": args.fps,
        "--workers": args.workers,
        "--image-writer-threads": args.image_writer_threads,
        "--video-encoding-batch-size": args.video_encoding_batch_size,
        "--encoder-threads": args.encoder_threads,
    }
    invalid = [
        name
        for name, value in positive_values.items()
        if value is not None and value <= 0
    ]
    if invalid:
        raise ValueError(f"The following arguments must be positive: {', '.join(invalid)}")
    if args.max_attempts is not None and args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.max_attempts is not None and args.max_attempts < args.episodes:
        raise ValueError("--max-attempts cannot be smaller than --episodes")
    if args.image_writer_processes < 0:
        raise ValueError("--image-writer-processes must be non-negative")
    effective_workers = min(args.workers, args.episodes)
    if args.viewer and effective_workers > 1:
        raise ValueError("The MuJoCo viewer is only supported with --workers 1")
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {args.root}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.root)


def _claim_attempt(
    *,
    next_attempt: Any,
    saved_episodes: Any,
    shared_lock: Any,
    stop_event: Any,
    target_episodes: int,
    maximum_attempts: int,
) -> int | None:
    with shared_lock:
        if (
            stop_event.is_set()
            or saved_episodes.value >= target_episodes
            or next_attempt.value >= maximum_attempts
        ):
            return None
        attempt_index = int(next_attempt.value)
        next_attempt.value += 1
        return attempt_index


def _reserve_episode(
    *,
    saved_episodes: Any,
    shared_lock: Any,
    stop_event: Any,
    target_episodes: int,
) -> bool:
    with shared_lock:
        if saved_episodes.value >= target_episodes:
            return False
        saved_episodes.value += 1
        if saved_episodes.value >= target_episodes:
            stop_event.set()
        return True


def _recording_metadata(controller: Any) -> tuple[int | None, str | None]:
    metadata = getattr(controller, "recording_metadata", {})
    drawer = metadata.get("drawer_index") if isinstance(metadata, dict) else None
    quadrant = metadata.get("cube_quadrant") if isinstance(metadata, dict) else None
    return (
        int(drawer) if drawer is not None else None,
        str(quadrant) if quadrant is not None else None,
    )


def _recording_task_phases(controller: Any) -> tuple[int, int]:
    """Read the semantic input and transition target for the generated action."""

    current = getattr(controller, "recording_task_phase", None)
    next_phase = getattr(controller, "recording_next_task_phase", None)
    if current is None or next_phase is None:
        raise RuntimeError(
            f"Controller {type(controller).__name__!r} must expose "
            "recording_task_phase and recording_next_task_phase"
        )
    return validate_task_phase(current), validate_task_phase(next_phase)


def _record_worker(
    worker_id: int,
    config: _WorkerConfig,
    next_attempt: Any,
    saved_episodes: Any,
    shared_lock: Any,
    stop_event: Any,
) -> _WorkerResult:
    shard_repo_id = f"{config.repo_id}-shard-{worker_id:03d}"
    shard_root = config.shard_root / f"worker-{worker_id:03d}"
    simulation_config = SimulationConfig(
        camera=config.camera,
        width=config.width,
        height=config.height,
        fps=config.fps,
        render=True,
        viewer=config.viewer,
        debug=config.debug,
        options=config.backend_options,
    )
    attempt_results: list[_AttemptResult] = []
    worker_saved_episodes = 0

    try:
        with create_backend(config.backend, simulation_config) as backend:
            dataset = create_dataset(
                backend=backend,
                repo_id=shard_repo_id,
                root=shard_root,
                fps=config.fps,
                use_videos=config.use_videos,
                image_writer_processes=config.image_writer_processes,
                image_writer_threads=config.image_writer_threads,
                video_encoding_batch_size=1,
                streaming_encoding=config.streaming_encoding,
                encoder_threads=config.encoder_threads,
            )
            frames_per_attempt = max(2, round(config.episode_seconds * config.fps))
            capture_deferred = getattr(backend, "capture_deferred_observation", None)
            materialize_deferred = getattr(
                backend, "materialize_deferred_observation", None
            )
            use_deferred_rendering = (
                config.deferred_rendering
                and not config.viewer
                and callable(capture_deferred)
                and callable(materialize_deferred)
            )
            try:
                while True:
                    attempt_index = _claim_attempt(
                        next_attempt=next_attempt,
                        saved_episodes=saved_episodes,
                        shared_lock=shared_lock,
                        stop_event=stop_event,
                        target_episodes=config.target_episodes,
                        maximum_attempts=config.maximum_attempts,
                    )
                    if attempt_index is None:
                        break

                    attempt_seed = config.base_seed + attempt_index
                    # Recreate the controller with the absolute attempt seed. Resetting
                    # at local episode zero makes every controller RNG derive from the
                    # same printed seed, independently of worker assignment.
                    controller = create_controller(
                        config.controller,
                        backend=backend,
                        seed=attempt_seed,
                    )
                    controller.reset(episode_seed=0)
                    drawer_index, cube_quadrant = _recording_metadata(controller)
                    if drawer_index is None:
                        raise RuntimeError(
                            f"Controller {config.controller!r} did not provide the "
                            "drawer_index required for drawer-conditioned recording"
                        )
                    episode_task = drawer_task(config.task, drawer_index)
                    initial_status = getattr(controller, "status", "controller reset")
                    print(
                        f"[worker {worker_id}] Starting attempt {attempt_index + 1}/"
                        f"{config.maximum_attempts} with seed {attempt_seed}: {initial_status}",
                        flush=True,
                    )

                    recorded_frames = 0
                    deferred_frames: list[_DeferredFrame] = []
                    for frame_index in range(frames_per_attempt):
                        progress = frame_index / (frames_per_attempt - 1)
                        if use_deferred_rendering:
                            observation, render_state = capture_deferred()
                        else:
                            observation = backend.get_observation(advance=False)
                            render_state = None
                        action = backend.send_action(controller.action(progress))
                        expert_action = getattr(controller, "recording_action", action)
                        current_task_phase, next_task_phase = _recording_task_phases(
                            controller
                        )
                        if use_deferred_rendering:
                            deferred_frames.append(
                                _DeferredFrame(
                                    observation=observation,
                                    render_state=render_state,
                                    action=expert_action,
                                    current_task_phase=current_task_phase,
                                    next_task_phase=next_task_phase,
                                )
                            )
                        else:
                            add_robot_frame(
                                dataset,
                                observation=observation,
                                action=expert_action,
                                task=episode_task,
                                drawer_index=drawer_index,
                                current_task_phase=current_task_phase,
                                next_task_phase=next_task_phase,
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
                    status = (
                        str(getattr(controller, "status", "episode completed"))
                        if getattr(controller, "done", False)
                        else "attempt timed out"
                    )
                    keep_episode = attempt_succeeded and _reserve_episode(
                        saved_episodes=saved_episodes,
                        shared_lock=shared_lock,
                        stop_event=stop_event,
                        target_episodes=config.target_episodes,
                    )

                    if keep_episode:
                        if use_deferred_rendering:
                            for deferred_frame in deferred_frames:
                                observation = materialize_deferred(
                                    deferred_frame.observation,
                                    deferred_frame.render_state,
                                )
                                add_robot_frame(
                                    dataset,
                                    observation=observation,
                                    action=deferred_frame.action,
                                    task=episode_task,
                                    drawer_index=drawer_index,
                                    current_task_phase=(
                                        deferred_frame.current_task_phase
                                    ),
                                    next_task_phase=deferred_frame.next_task_phase,
                                )
                        dataset.save_episode()
                        worker_saved_episodes += 1
                        print(
                            f"[worker {worker_id}] Saved seed {attempt_seed} "
                            f"({recorded_frames} frames)",
                            flush=True,
                        )
                    else:
                        if not use_deferred_rendering:
                            dataset.clear_episode_buffer()
                        disposition = (
                            "successful surplus attempt" if attempt_succeeded else status
                        )
                        print(
                            f"[worker {worker_id}] Discarded seed {attempt_seed}: "
                            f"{disposition} ({recorded_frames} frames)",
                            flush=True,
                        )

                    attempt_results.append(
                        _AttemptResult(
                            attempt_index=attempt_index,
                            seed=attempt_seed,
                            succeeded=attempt_succeeded,
                            saved=keep_episode,
                            frames=recorded_frames,
                            status=status,
                            drawer_index=drawer_index,
                            cube_quadrant=cube_quadrant,
                        )
                    )
            finally:
                active_error = sys.exception()
                try:
                    episode_buffer = getattr(dataset.writer, "episode_buffer", None)
                    if (
                        isinstance(episode_buffer, dict)
                        and episode_buffer.get("size", 0) > 0
                    ):
                        dataset.clear_episode_buffer()
                    dataset.finalize()
                except Exception:
                    if active_error is None:
                        raise
    except Exception:
        stop_event.set()
        raise

    return _WorkerResult(
        worker_id=worker_id,
        repo_id=shard_repo_id,
        root=shard_root,
        saved_episodes=worker_saved_episodes,
        attempts=tuple(attempt_results),
    )


def _percentage(count: int, total: int) -> str:
    return "0.0%" if total == 0 else f"{100.0 * count / total:.1f}%"


def _failure_category(status: str) -> str:
    return status.partition(":")[0]


def _print_recording_summary(results: list[_AttemptResult]) -> None:
    ordered = sorted(results, key=lambda result: result.attempt_index)
    successful = [result for result in ordered if result.succeeded]
    saved = [result for result in ordered if result.saved]
    print("\nRecording summary")
    print(f"  Attempts completed: {len(ordered)}")
    print(f"  Physically successful: {len(successful)}")
    print(f"  Saved episodes: {len(saved)}")
    print(f"  Failed attempts: {len(ordered) - len(successful)}")
    print(f"  Overall success rate: {_percentage(len(successful), len(ordered))}")

    failure_counts = Counter(
        _failure_category(result.status) for result in ordered if not result.succeeded
    )
    if failure_counts:
        print("\nFailure reasons")
        for reason, count in failure_counts.most_common():
            print(f"  {reason}: {count}")

    attempted_drawers = Counter(result.drawer_index for result in ordered)
    successful_drawers = Counter(result.drawer_index for result in successful)
    saved_drawers = Counter(result.drawer_index for result in saved)
    print("\nSaved dataset distribution by drawer")
    for drawer_index in range(1, 10):
        attempted = attempted_drawers[drawer_index]
        succeeded = successful_drawers[drawer_index]
        count = saved_drawers[drawer_index]
        print(
            f"  Drawer {drawer_index}: {count}/{len(saved)} "
            f"({_percentage(count, len(saved))}); attempt success "
            f"{succeeded}/{attempted} ({_percentage(succeeded, attempted)})"
        )

    attempted_quadrants = Counter(result.cube_quadrant for result in ordered)
    successful_quadrants = Counter(result.cube_quadrant for result in successful)
    saved_quadrants = Counter(result.cube_quadrant for result in saved)
    print("\nSaved dataset distribution by cube quadrant")
    for quadrant in CUBE_QUADRANTS:
        attempted = attempted_quadrants[quadrant]
        succeeded = successful_quadrants[quadrant]
        count = saved_quadrants[quadrant]
        print(
            f"  {quadrant}: {count}/{len(saved)} ({_percentage(count, len(saved))}); "
            f"attempt success {succeeded}/{attempted} "
            f"({_percentage(succeeded, attempted)})"
        )


def _merge_worker_datasets(
    *,
    worker_results: list[_WorkerResult],
    repo_id: str,
    root: Path,
) -> LeRobotDataset | None:
    populated = sorted(
        (result for result in worker_results if result.saved_episodes > 0),
        key=lambda result: result.worker_id,
    )
    if not populated:
        return None
    shards = [
        LeRobotDataset(repo_id=result.repo_id, root=result.root)
        for result in populated
    ]
    return merge_datasets(
        shards,
        output_repo_id=repo_id,
        output_dir=root,
        concatenate_videos=False,
        concatenate_data=False,
    )


def record_dataset(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if args.video_encoding_batch_size != 1:
        print(
            "Warning: LeRobot 0.6.1 batch video encoding is incompatible with "
            "new worker shards; using --video-encoding-batch-size 1."
        )
    args.root.parent.mkdir(parents=True, exist_ok=True)
    maximum_attempts = args.max_attempts or args.episodes * 3
    worker_count = min(args.workers, args.episodes, maximum_attempts)
    shard_root = Path(
        tempfile.mkdtemp(
            prefix=f".{args.root.name}.shards-",
            dir=args.root.parent,
        )
    )
    config = _WorkerConfig(
        backend=args.backend,
        controller=args.controller,
        repo_id=args.repo_id,
        shard_root=shard_root,
        episode_seconds=args.episode_seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera=args.camera,
        task=args.task,
        base_seed=args.seed,
        target_episodes=args.episodes,
        maximum_attempts=maximum_attempts,
        use_videos=not args.no_video,
        deferred_rendering=args.deferred_rendering,
        viewer=args.viewer,
        debug=args.debug and args.viewer,
        backend_options=parse_backend_options(args.backend_option),
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        streaming_encoding=args.streaming_encoding,
        encoder_threads=args.encoder_threads,
    )
    print(
        f"Recording {args.episodes} successful episodes with {worker_count} "
        f"{'headless' if not args.viewer else 'interactive'} worker"
        f"{'s' if worker_count != 1 else ''}; deferred rendering="
        f"{args.deferred_rendering and not args.viewer}"
    )

    context = multiprocessing.get_context("spawn")
    worker_results: list[_WorkerResult] = []
    try:
        with context.Manager() as manager:
            next_attempt = manager.Value("i", 0)
            saved_episodes = manager.Value("i", 0)
            shared_lock = manager.Lock()
            stop_event = manager.Event()
            if worker_count == 1:
                worker_results.append(
                    _record_worker(
                        0,
                        config,
                        next_attempt,
                        saved_episodes,
                        shared_lock,
                        stop_event,
                    )
                )
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=context,
                ) as executor:
                    futures = [
                        executor.submit(
                            _record_worker,
                            worker_id,
                            config,
                            next_attempt,
                            saved_episodes,
                            shared_lock,
                            stop_event,
                        )
                        for worker_id in range(worker_count)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        worker_results.append(future.result())

        attempts = [
            attempt
            for worker_result in worker_results
            for attempt in worker_result.attempts
        ]
        _print_recording_summary(attempts)
        saved_count = sum(result.saved_episodes for result in worker_results)
        if saved_count < args.episodes:
            raise RuntimeError(
                f"Collected {saved_count}/{args.episodes} successful episodes after "
                f"{len(attempts)} completed attempts; increase --max-attempts or inspect "
                "the per-drawer and per-quadrant success rates above."
            )
        merged_dataset = _merge_worker_datasets(
            worker_results=worker_results,
            repo_id=args.repo_id,
            root=args.root,
        )
        if merged_dataset is None:
            raise RuntimeError("No populated worker datasets were available to merge")
        if args.push_to_hub:
            merged_dataset.push_to_hub(tags=["hepha", args.backend, args.controller])
        shutil.rmtree(shard_root)
        return args.root
    except Exception:
        print(f"Worker shards retained for diagnosis at: {shard_root}")
        raise


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
