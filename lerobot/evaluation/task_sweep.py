"""Measure complete closed-loop task performance over headless MuJoCo episodes."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch
from hepha_lerobot.conditioning import drawer_task
from hepha_lerobot.evaluation.conditioned_rollout import (
    _load_policy,
    _policy_observation,
)
from hepha_lerobot.evaluation.grasp_sweep import _nearest_hand_distance
from hepha_lerobot.evaluation.phase_control import PhaseTransitionState
from hepha_lerobot.training.train import resolve_device
from lerobot.policies.utils import prepare_observation_for_inference

from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import (
    cube_position,
    cube_spawn_quadrant,
    initialize_task_episode,
)
from simulation.backends.mujoco.ik import (
    CUBE_HAND_DISTANCE_M,
    CUBE_LIFT_CHECK_M,
    _cube_center_inside_drawer,
)

DRAWER_OPEN_THRESHOLD_M = 0.040
DRAWER_CLOSED_THRESHOLD_M = 0.005


@dataclass
class TaskMilestones:
    drawer_opened: bool = False
    cube_grasped: bool = False
    cube_in_drawer: bool = False
    drawer_closed_after_insertion: bool = False

    def update(
        self,
        *,
        drawer_opening_m: float,
        stable_cube_grasp: bool,
        cube_inside_drawer: bool,
    ) -> None:
        self.drawer_opened |= drawer_opening_m >= DRAWER_OPEN_THRESHOLD_M
        self.cube_grasped |= stable_cube_grasp
        self.cube_in_drawer |= cube_inside_drawer
        self.drawer_closed_after_insertion |= (
            self.drawer_opened
            and self.cube_in_drawer
            and drawer_opening_m <= DRAWER_CLOSED_THRESHOLD_M
        )


@dataclass(frozen=True)
class TaskResult:
    seed: int
    drawer_index: int
    cube_quadrant: str
    drawer_opened: bool
    cube_grasped: bool
    cube_in_drawer: bool
    drawer_closed_after_insertion: bool
    final_cube_inside_drawer: bool
    final_drawer_closed: bool
    successful: bool
    final_phase: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seconds", type=float, default=100.0)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    parser.add_argument("--domain-randomization-scale", type=float, default=0.0)
    parser.add_argument("--stable-grasp-frames", type=int, default=5)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_seconds <= 0 or args.fps <= 0:
        raise ValueError("--episode-seconds and --fps must be positive")
    if args.stable_grasp_frames <= 0:
        raise ValueError("--stable-grasp-frames must be positive")
    if args.domain_randomization_scale < 0.0:
        raise ValueError("--domain-randomization-scale must be non-negative")
    if not args.policy_path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")


def _drawer_opening(backend: MujocoBackend, drawer_index: int) -> float:
    joint_name = f"base_link_base_drawer_{drawer_index}_joint"
    joint_id = mujoco.mj_name2id(
        backend.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    if joint_id < 0:
        raise RuntimeError(f"MuJoCo joint not found: {joint_name}")
    qpos_id = int(backend.model.jnt_qposadr[joint_id])
    return max(
        0.0,
        float(backend.model.jnt_range[joint_id, 1] - backend.data.qpos[qpos_id]),
    )


def _run_episode(
    *,
    backend: MujocoBackend,
    policy,
    preprocessor,
    postprocessor,
    device: str,
    policy_type: str,
    seed: int,
    max_steps: int,
    stable_grasp_frames: int,
    camera: str,
) -> TaskResult:
    rng = initialize_task_episode(backend, seed=seed)
    drawer_index = int(rng.integers(1, 10))
    quadrant = cube_spawn_quadrant(backend.model, backend.data)
    initial_cube_z = float(cube_position(backend.model, backend.data)[2])
    task = drawer_task(
        "Open drawer {drawer_index}, pick up the cube, place it inside that drawer, "
        "and close the drawer.",
        drawer_index,
    )
    state_names = tuple(
        name
        for name, feature_type in backend.observation_features.items()
        if feature_type is float
    )
    phase_state = PhaseTransitionState() if policy_type == "hepha_act_phase" else None
    milestones = TaskMilestones()
    stable_grasp_count = 0
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    for _ in range(max_steps):
        raw_observation = backend.get_observation()
        lift = float(cube_position(backend.model, backend.data)[2]) - initial_cube_z
        hand_distance = _nearest_hand_distance(backend)
        physically_grasped = (
            lift >= CUBE_LIFT_CHECK_M and hand_distance <= CUBE_HAND_DISTANCE_M
        )
        stable_grasp_count = stable_grasp_count + 1 if physically_grasped else 0
        opening = _drawer_opening(backend, drawer_index)
        inside = _cube_center_inside_drawer(
            backend.model, backend.data, drawer_index
        )
        milestones.update(
            drawer_opening_m=opening,
            stable_cube_grasp=stable_grasp_count >= stable_grasp_frames,
            cube_inside_drawer=inside,
        )

        observation = _policy_observation(
            raw_observation,
            state_names=state_names,
            camera=camera,
            drawer_index=drawer_index,
            current_phase=(phase_state.current_phase if phase_state is not None else None),
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

    # Advance the final command through physics before evaluating the terminal state.
    backend.step()
    final_opening = _drawer_opening(backend, drawer_index)
    final_inside = _cube_center_inside_drawer(
        backend.model, backend.data, drawer_index
    )
    final_closed = final_opening <= DRAWER_CLOSED_THRESHOLD_M
    milestones.update(
        drawer_opening_m=final_opening,
        stable_cube_grasp=False,
        cube_inside_drawer=final_inside,
    )
    return TaskResult(
        seed=seed,
        drawer_index=drawer_index,
        cube_quadrant=quadrant,
        drawer_opened=milestones.drawer_opened,
        cube_grasped=milestones.cube_grasped,
        cube_in_drawer=milestones.cube_in_drawer,
        drawer_closed_after_insertion=milestones.drawer_closed_after_insertion,
        final_cube_inside_drawer=final_inside,
        final_drawer_closed=final_closed,
        successful=final_inside and final_closed,
        final_phase=(phase_state.current_phase if phase_state is not None else 0),
    )


def _rate(results: list[TaskResult], attribute: str) -> tuple[int, float]:
    count = sum(bool(getattr(result, attribute)) for result in results)
    return count, count / len(results)


def run(args: argparse.Namespace) -> list[TaskResult]:
    _validate_args(args)
    device = resolve_device(args.device)
    policy, policy_config, preprocessor, postprocessor = _load_policy(
        args.policy_path,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    simulation_config = SimulationConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=True,
        viewer=False,
        debug=False,
        options={"domain_randomization_scale": args.domain_randomization_scale},
    )
    max_steps = round(args.episode_seconds * args.fps)
    results: list[TaskResult] = []
    started = time.perf_counter()
    with MujocoBackend(simulation_config) as backend:
        for episode_index in range(args.episodes):
            result = _run_episode(
                backend=backend,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                policy_type=policy_config.type,
                seed=args.seed_start + episode_index,
                max_steps=max_steps,
                stable_grasp_frames=args.stable_grasp_frames,
                camera=args.camera,
            )
            results.append(result)
            print(
                f"[{episode_index + 1:03d}/{args.episodes:03d}] "
                f"{'PASS' if result.successful else 'FAIL'} seed={result.seed} "
                f"drawer={result.drawer_index} quadrant={result.cube_quadrant} "
                f"opened={result.drawer_opened} grasped={result.cube_grasped} "
                f"inside={result.cube_in_drawer} "
                f"closed={result.drawer_closed_after_insertion} "
                f"final_success={result.successful}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print("\nClosed-loop task sweep summary")
    print(f"Episodes: {len(results)}")
    for label, attribute in (
        ("Drawer opened", "drawer_opened"),
        ("Cube grasped", "cube_grasped"),
        ("Cube entered selected drawer", "cube_in_drawer"),
        ("Drawer closed after insertion", "drawer_closed_after_insertion"),
        ("Final cube inside drawer", "final_cube_inside_drawer"),
        ("Final drawer closed", "final_drawer_closed"),
        ("Final closed-loop success", "successful"),
    ):
        count, rate = _rate(results, attribute)
        print(f"{label}: {count}/{len(results)} ({rate:.2%})")
    print(f"Wall time: {elapsed:.1f} s")
    print(f"Successful seeds: {[result.seed for result in results if result.successful]}")
    print(f"Failed seeds: {[result.seed for result in results if not result.successful]}")
    return results


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
