"""Measure cube-grasp success over reproducible headless MuJoCo rollouts."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from hepha_lerobot.conditioning import drawer_task
from hepha_lerobot.evaluation.conditioned_rollout import (
    _load_policy,
    _policy_observation,
)
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
from simulation.backends.mujoco.ik import CUBE_HAND_DISTANCE_M, CUBE_LIFT_CHECK_M, hand_pose


@dataclass(frozen=True)
class GraspResult:
    seed: int
    drawer_index: int
    cube_quadrant: str
    finger_contact: bool
    successful: bool
    first_contact_step: int | None
    elapsed_steps: int
    max_lift_m: float
    closest_hand_distance_m: float
    final_phase: int


_GRASP_FINGER_GEOMS = tuple(
    f"{link}_{side}_link_collision_box_{box}_geom"
    for side in ("l", "r")
    for link in ("hand", "finger")
    for box in ("02", "03")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    parser.add_argument("--domain-randomization-scale", type=float, default=0.0)
    parser.add_argument("--stable-frames", type=int, default=5)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_seconds <= 0 or args.fps <= 0:
        raise ValueError("--episode-seconds and --fps must be positive")
    if args.stable_frames <= 0:
        raise ValueError("--stable-frames must be positive")
    if args.domain_randomization_scale < 0.0:
        raise ValueError("--domain-randomization-scale must be non-negative")
    if not args.policy_path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")


def _nearest_hand_distance(backend: MujocoBackend) -> float:
    cube = cube_position(backend.model, backend.data)
    return min(
        float(np.linalg.norm(cube - hand_pose(backend.model, backend.data, side)[0]))
        for side in ("l", "r")
    )


def _grasp_contact_ids(model: mujoco.MjModel) -> tuple[int, frozenset[int]]:
    cube_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube_link_collision_box_01_geom"
    )
    finger_ids = frozenset(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in _GRASP_FINGER_GEOMS
    )
    if cube_id < 0 or -1 in finger_ids:
        raise RuntimeError("MuJoCo model is missing cube or fingertip collision geometries")
    return cube_id, finger_ids


def _has_cube_finger_contact(
    data: mujoco.MjData,
    *,
    cube_geom_id: int,
    finger_geom_ids: frozenset[int],
) -> bool:
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if (geom1 == cube_geom_id and geom2 in finger_geom_ids) or (
            geom2 == cube_geom_id and geom1 in finger_geom_ids
        ):
            return True
    return False


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
    stable_frames: int,
    camera: str,
    cube_geom_id: int,
    finger_geom_ids: frozenset[int],
) -> GraspResult:
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
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    stable_count = 0
    max_lift = 0.0
    closest_hand_distance = float("inf")
    successful = False
    finger_contact = False
    first_contact_step: int | None = None
    elapsed_steps = max_steps

    for step in range(1, max_steps + 1):
        raw_observation = backend.get_observation()
        if _has_cube_finger_contact(
            backend.data,
            cube_geom_id=cube_geom_id,
            finger_geom_ids=finger_geom_ids,
        ):
            finger_contact = True
            if first_contact_step is None:
                first_contact_step = step
        lift = float(cube_position(backend.model, backend.data)[2]) - initial_cube_z
        hand_distance = _nearest_hand_distance(backend)
        max_lift = max(max_lift, lift)
        closest_hand_distance = min(closest_hand_distance, hand_distance)
        physically_grasped = (
            lift >= CUBE_LIFT_CHECK_M and hand_distance <= CUBE_HAND_DISTANCE_M
        )
        stable_count = stable_count + 1 if physically_grasped else 0
        if stable_count >= stable_frames:
            successful = True
            elapsed_steps = step
            break

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

    return GraspResult(
        seed=seed,
        drawer_index=drawer_index,
        cube_quadrant=quadrant,
        finger_contact=finger_contact,
        successful=successful,
        first_contact_step=first_contact_step,
        elapsed_steps=elapsed_steps,
        max_lift_m=max_lift,
        closest_hand_distance_m=closest_hand_distance,
        final_phase=(phase_state.current_phase if phase_state is not None else 0),
    )


def run(args: argparse.Namespace) -> list[GraspResult]:
    _validate_args(args)
    device = resolve_device(args.device)
    policy, policy_config, preprocessor, postprocessor = _load_policy(
        args.policy_path,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    if policy_config.type == "hepha_act_phase" and policy_config.n_action_steps != 1:
        raise ValueError("Phase-aware grasp evaluation requires --n-action-steps 1")

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
    results: list[GraspResult] = []
    started = time.perf_counter()
    with MujocoBackend(simulation_config) as backend:
        cube_geom_id, finger_geom_ids = _grasp_contact_ids(backend.model)
        for episode_index in range(args.episodes):
            seed = args.seed_start + episode_index
            result = _run_episode(
                backend=backend,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                policy_type=policy_config.type,
                seed=seed,
                max_steps=max_steps,
                stable_frames=args.stable_frames,
                camera=args.camera,
                cube_geom_id=cube_geom_id,
                finger_geom_ids=finger_geom_ids,
            )
            results.append(result)
            successes = sum(item.successful for item in results)
            outcome = "PASS" if result.successful else "FAIL"
            print(
                f"[{episode_index + 1:03d}/{args.episodes:03d}] {outcome} "
                f"seed={seed} drawer={result.drawer_index} "
                f"quadrant={result.cube_quadrant} contact={result.finger_contact} "
                f"phase={result.final_phase} "
                f"running_success={successes / len(results):.1%}",
                flush=True,
            )

    successful = [result for result in results if result.successful]
    contacted = [result for result in results if result.finger_contact]
    failed = [result for result in results if not result.successful]
    elapsed = time.perf_counter() - started
    print("\nCube-grasp sweep summary")
    print(f"Episodes: {len(results)}")
    print(f"Episodes with cube/finger contact: {len(contacted)}")
    print(f"Cube/finger contact rate: {len(contacted) / len(results):.2%}")
    print(f"Successful grasps: {len(successful)}")
    print(f"Failed grasps: {len(failed)}")
    print(f"Success rate: {len(successful) / len(results):.2%}")
    print(f"Wall time: {elapsed:.1f} s")
    print(
        "Contact without stable grasp seeds: "
        f"{[result.seed for result in contacted if not result.successful]}"
    )
    print(
        "No cube/finger contact seeds: "
        f"{[result.seed for result in results if not result.finger_contact]}"
    )
    print(f"Failed seeds: {[result.seed for result in failed]}")
    return results


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
