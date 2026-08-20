"""Collect rewarded closed-loop ACT rollouts as a native LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from hepha_lerobot.awr.rewards import AWRRewardTracker
from hepha_lerobot.conditioning import DEFAULT_TASK, drawer_task
from hepha_lerobot.datasets.builder import add_robot_frame, create_dataset
from hepha_lerobot.evaluation.conditioned_rollout import _load_policy, _policy_observation
from hepha_lerobot.training.train import resolve_device
from lerobot.policies.utils import prepare_observation_for_inference

from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import (
    cube_spawn_quadrant,
    initialize_task_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seconds", type=float, default=100.0)
    parser.add_argument("--seed-start", type=int, default=30_000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    parser.add_argument("--domain-randomization-scale", type=float, default=0.0)
    parser.add_argument("--stable-grasp-frames", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if args.episodes <= 0 or args.episode_seconds <= 0 or args.fps <= 0:
        raise ValueError("episodes, episode-seconds and fps must be positive")
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset root exists: {args.root}")
        shutil.rmtree(args.root)

    device = resolve_device(args.device)
    policy, policy_config, preprocessor, postprocessor = _load_policy(
        args.policy_path,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    if policy_config.type not in {"act", "hepha_act_awr"}:
        raise ValueError("AWR rollout collection currently supports original ACT-derived policies")

    config = SimulationConfig(
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
    success_count = 0
    drawer_counts: Counter[int] = Counter()
    quadrant_counts: Counter[str] = Counter()
    returns: list[float] = []
    event_counts: Counter[str] = Counter()
    started = time.perf_counter()

    with MujocoBackend(config) as backend:
        dataset = create_dataset(
            backend=backend,
            repo_id=args.repo_id,
            root=args.root,
            fps=args.fps,
            use_videos=True,
            include_task_phase=False,
            include_rewards=True,
        )
        state_names = tuple(
            name
            for name, feature_type in backend.observation_features.items()
            if feature_type is float
        )
        action_names = tuple(backend.action_features)
        try:
            for episode in range(args.episodes):
                seed = args.seed_start + episode
                rng = initialize_task_episode(backend, seed=seed)
                drawer_index = int(rng.integers(1, 10))
                quadrant = cube_spawn_quadrant(backend.model, backend.data)
                task = drawer_task(DEFAULT_TASK, drawer_index)
                tracker = AWRRewardTracker(
                    backend,
                    drawer_index=drawer_index,
                    stable_grasp_frames=args.stable_grasp_frames,
                )
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                episode_return = 0.0

                for step in range(max_steps):
                    raw_observation = backend.get_observation(advance=False)
                    observation = _policy_observation(
                        raw_observation,
                        state_names=state_names,
                        camera=args.camera,
                        drawer_index=drawer_index,
                    )
                    batch = prepare_observation_for_inference(
                        observation,
                        torch.device(device),
                        task=task,
                        robot_type="hepha_mujoco",
                    )
                    with torch.inference_mode():
                        batch = preprocessor(batch)
                        action = postprocessor(policy.select_action(batch))
                    values = action.squeeze(0).detach().cpu().numpy()
                    action_dict = dict(zip(action_names, values.tolist(), strict=True))
                    backend.send_action(action_dict)
                    backend.step()
                    final_step = step == max_steps - 1
                    reward = tracker.evaluate(final_step=final_step)
                    for key, value in vars(reward).items():
                        if key != "step_cost" and value != 0.0:
                            event_counts[key] += 1
                    episode_return += reward.total
                    add_robot_frame(
                        dataset,
                        observation=raw_observation,
                        action=action_dict,
                        task=task,
                        drawer_index=drawer_index,
                        current_task_phase=None,
                        next_task_phase=None,
                        reward=reward.total,
                        done=final_step,
                    )
                dataset.save_episode()
                successful = bool(reward.final_success)
                success_count += int(successful)
                drawer_counts[drawer_index] += 1
                quadrant_counts[quadrant] += 1
                returns.append(episode_return)
                print(
                    f"[{episode + 1:03d}/{args.episodes:03d}] "
                    f"{'PASS' if successful else 'FAIL'} seed={seed} drawer={drawer_index} "
                    f"quadrant={quadrant} return={episode_return:.3f}",
                    flush=True,
                )
        finally:
            dataset.finalize()

    if args.push_to_hub:
        dataset.push_to_hub(tags=["hepha", "mujoco", "awr", "act-rollout"])
    elapsed = time.perf_counter() - started
    print("\nAWR rollout summary")
    print(f"Episodes: {args.episodes}")
    print(f"Final success: {success_count}/{args.episodes} ({success_count / args.episodes:.2%})")
    print(f"Mean undiscounted return: {sum(returns) / len(returns):.3f}")
    print(f"Drawer distribution: {dict(sorted(drawer_counts.items()))}")
    print(f"Cube quadrant distribution: {dict(sorted(quadrant_counts.items()))}")
    print(f"Reward event counts: {dict(sorted(event_counts.items()))}")
    print(f"Wall time: {elapsed:.1f} s")
    print(f"LeRobot AWR dataset ready at {args.root}")
    return args.root


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
