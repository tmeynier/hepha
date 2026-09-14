"""Evaluate complete IK-controller tasks over uninterrupted MuJoCo episodes."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import cube_position
from simulation.backends.mujoco.ik import (
    CUBE_HAND_DISTANCE_M,
    CUBE_LIFT_CHECK_M,
    MujocoIKController,
    _cube_center_inside_drawer,
    hand_pose,
)
from simulation.view import _ensure_mjpython_on_macos

from .task_metrics import (
    DRAWER_CLOSED_THRESHOLD_M,
    TaskMilestones,
    drawer_opening,
)


@dataclass(frozen=True)
class IKTaskResult:
    seed: int
    drawer_index: int
    cube_quadrant: str
    handoff_required: bool
    drawer_opened: bool
    cube_grasped: bool
    handoff_completed: bool
    cube_in_drawer: bool
    drawer_closed_after_insertion: bool
    final_cube_inside_drawer: bool
    final_drawer_closed: bool
    successful: bool
    completed: bool
    elapsed_steps: int
    status: str

    @property
    def handoff_requirement_satisfied(self) -> bool:
        return not self.handoff_required or self.handoff_completed


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=360.0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stable-grasp-frames", type=int, default=5)
    parser.add_argument("--trajectory-randomization-scale", type=float, default=1.0)
    parser.add_argument(
        "--viewer",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
    )
    parser.add_argument(
        "--early-failures",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Enable task-quality early aborts (default: false)",
    )
    parser.add_argument(
        "--cube-grasp-lock",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help=(
            "Optionally lock a successfully approached cube to its grasping hand "
            "until handoff or release (default: false)"
        ),
    )
    parser.add_argument(
        "--cube-drop-assist",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        metavar="BOOL",
        help=(
            "Guide a released cube downward inside its selected drawer without "
            "lateral or upward bounce (default: true)"
        ),
    )
    parser.add_argument(
        "--realtime",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        metavar="BOOL",
        help="Pace the native viewer at --fps (default: true)",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_seconds <= 0.0 or args.fps <= 0:
        raise ValueError("--episode-seconds and --fps must be positive")
    if args.stable_grasp_frames <= 0:
        raise ValueError("--stable-grasp-frames must be positive")
    if args.trajectory_randomization_scale < 0.0:
        raise ValueError("--trajectory-randomization-scale must be non-negative")
    if args.debug and not args.viewer:
        raise ValueError("--debug requires --viewer true")


def _nearest_hand_distance(backend: MujocoBackend) -> float:
    cube = cube_position(backend.model, backend.data)
    return min(
        float(np.linalg.norm(cube - hand_pose(backend.model, backend.data, side)[0]))
        for side in ("l", "r")
    )


def _sample_milestones(
    *,
    backend: MujocoBackend,
    controller: MujocoIKController,
    milestones: TaskMilestones,
    initial_cube_z: float,
    stable_grasp_count: int,
    stable_grasp_frames: int,
) -> tuple[int, bool]:
    lift = float(cube_position(backend.model, backend.data)[2]) - initial_cube_z
    physically_grasped = (
        lift >= CUBE_LIFT_CHECK_M
        and _nearest_hand_distance(backend) <= CUBE_HAND_DISTANCE_M
    )
    stable_grasp_count = stable_grasp_count + 1 if physically_grasped else 0
    opening = drawer_opening(backend, controller.drawer_index)
    inside = _cube_center_inside_drawer(
        backend.model,
        backend.data,
        controller.drawer_index,
    )
    milestones.update(
        drawer_opening_m=opening,
        stable_cube_grasp=stable_grasp_count >= stable_grasp_frames,
        cube_inside_drawer=inside,
    )
    handoff_completed = (
        controller.handoff_donor is not None
        and controller.cube_hand == controller.placement_hand
        and controller.phase == "handoff_donor_rest"
    )
    return stable_grasp_count, handoff_completed


def _run_episode(
    *,
    backend: MujocoBackend,
    seed: int,
    max_steps: int,
    stable_grasp_frames: int,
    trajectory_randomization_scale: float,
    early_failures: bool,
    cube_grasp_lock: bool,
    cube_drop_assist: bool,
    realtime: bool,
) -> IKTaskResult:
    controller = MujocoIKController(
        backend,
        seed=seed,
        trajectory_randomization_scale=trajectory_randomization_scale,
        early_failures=early_failures,
        cube_grasp_lock=cube_grasp_lock,
        cube_drop_assist=cube_drop_assist,
    )
    controller.reset(episode_seed=0)
    initial_cube_z = float(cube_position(backend.model, backend.data)[2])
    handoff_required = controller.cube_hand != controller.placement_hand
    milestones = TaskMilestones()
    stable_grasp_count = 0
    handoff_completed = False
    elapsed_steps = max_steps
    period = 1.0 / backend.config.fps

    for step in range(1, max_steps + 1):
        frame_started = time.perf_counter()
        progress = (step - 1) / max(1, max_steps - 1)
        backend.send_action(controller.action(progress))
        backend.step()
        stable_grasp_count, just_completed_handoff = _sample_milestones(
            backend=backend,
            controller=controller,
            milestones=milestones,
            initial_cube_z=initial_cube_z,
            stable_grasp_count=stable_grasp_count,
            stable_grasp_frames=stable_grasp_frames,
        )
        handoff_completed |= just_completed_handoff
        if realtime and backend.config.viewer:
            time.sleep(max(0.0, period - (time.perf_counter() - frame_started)))
        if controller.done:
            elapsed_steps = step
            break

    final_opening = drawer_opening(backend, controller.drawer_index)
    final_inside = _cube_center_inside_drawer(
        backend.model,
        backend.data,
        controller.drawer_index,
    )
    final_closed = final_opening <= DRAWER_CLOSED_THRESHOLD_M
    milestones.update(
        drawer_opening_m=final_opening,
        stable_cube_grasp=False,
        cube_inside_drawer=final_inside,
    )
    return IKTaskResult(
        seed=seed,
        drawer_index=controller.drawer_index,
        cube_quadrant=controller.cube_quadrant,
        handoff_required=handoff_required,
        drawer_opened=milestones.drawer_opened,
        cube_grasped=milestones.cube_grasped,
        handoff_completed=handoff_completed,
        cube_in_drawer=milestones.cube_in_drawer,
        drawer_closed_after_insertion=milestones.drawer_closed_after_insertion,
        final_cube_inside_drawer=final_inside,
        final_drawer_closed=final_closed,
        successful=final_inside and final_closed,
        completed=controller.done,
        elapsed_steps=elapsed_steps,
        status=(controller.status if controller.done else "episode timed out"),
    )


def _rate(results: list[IKTaskResult], attribute: str) -> tuple[int, float]:
    count = sum(bool(getattr(result, attribute)) for result in results)
    return count, count / len(results)


def run(args: argparse.Namespace) -> list[IKTaskResult]:
    _validate_args(args)
    config = SimulationConfig(
        fps=args.fps,
        render=False,
        viewer=args.viewer,
        debug=args.debug,
    )
    max_steps = round(args.episode_seconds * args.fps)
    results: list[IKTaskResult] = []
    started = time.perf_counter()

    with MujocoBackend(config) as backend:
        for episode_index in range(args.episodes):
            seed = args.seed_start + episode_index
            result = _run_episode(
                backend=backend,
                seed=seed,
                max_steps=max_steps,
                stable_grasp_frames=args.stable_grasp_frames,
                trajectory_randomization_scale=args.trajectory_randomization_scale,
                early_failures=args.early_failures,
                cube_grasp_lock=args.cube_grasp_lock,
                cube_drop_assist=args.cube_drop_assist,
                realtime=args.realtime,
            )
            results.append(result)
            handoff = (
                str(result.handoff_completed) if result.handoff_required else "n/a"
            )
            print(
                f"[{episode_index + 1:02d}/{args.episodes:02d}] "
                f"{'PASS' if result.successful else 'FAIL'} seed={result.seed} "
                f"drawer={result.drawer_index} quadrant={result.cube_quadrant} "
                f"opened={result.drawer_opened} grasped={result.cube_grasped} "
                f"handoff={handoff} inside={result.cube_in_drawer} "
                f"closed={result.drawer_closed_after_insertion} "
                f"final_success={result.successful} status={result.status}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print("\nIK closed-loop task evaluation")
    print(f"Episodes completed: {len(results)}")
    for label, attribute in (
        ("Drawer opened", "drawer_opened"),
        ("Cube grasped", "cube_grasped"),
        ("Handoff requirement satisfied", "handoff_requirement_satisfied"),
        ("Cube entered selected drawer", "cube_in_drawer"),
        ("Drawer closed after insertion", "drawer_closed_after_insertion"),
        ("Final cube inside drawer", "final_cube_inside_drawer"),
        ("Final drawer closed", "final_drawer_closed"),
        ("Final closed-loop success", "successful"),
    ):
        count, rate = _rate(results, attribute)
        print(f"{label}: {count}/{len(results)} ({rate:.1%})")

    required = [result for result in results if result.handoff_required]
    completed_handoffs = sum(result.handoff_completed for result in required)
    if required:
        print(
            "Physical handoff when required: "
            f"{completed_handoffs}/{len(required)} "
            f"({completed_handoffs / len(required):.1%})"
        )
    else:
        print("Physical handoff when required: n/a (no episode required handoff)")
    print(f"Wall time: {elapsed:.1f} s")
    print(f"Successful seeds: {[result.seed for result in results if result.successful]}")
    print(f"Failed seeds: {[result.seed for result in results if not result.successful]}")
    return results


def main() -> None:
    args = parse_args()
    if args.viewer:
        _ensure_mjpython_on_macos("hepha_lerobot.evaluation.ik_sweep")
    run(args)


if __name__ == "__main__":
    main()
