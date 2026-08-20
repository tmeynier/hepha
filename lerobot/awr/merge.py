"""Append new policy rollouts to the cumulative LeRobot AWR replay dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset, merge_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-repo-id", required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--new-repo-id", required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    output = args.root.expanduser().resolve()
    inputs = {args.replay_root.expanduser().resolve(), args.new_root.expanduser().resolve()}
    if output in inputs:
        raise ValueError("Merged output root must differ from both input roots")
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Replay output exists: {args.root}")
        shutil.rmtree(args.root)

    replay = LeRobotDataset(args.replay_repo_id, root=args.replay_root)
    new_rollouts = LeRobotDataset(args.new_repo_id, root=args.new_root)
    if replay.fps != new_rollouts.fps or replay.features != new_rollouts.features:
        raise ValueError("Replay and new-rollout dataset schemas or FPS do not match")
    merged = merge_datasets(
        [replay, new_rollouts],
        output_repo_id=args.repo_id,
        output_dir=args.root,
        concatenate_videos=False,
        concatenate_data=False,
    )
    if args.push_to_hub:
        merged.push_to_hub(tags=["hepha", "mujoco", "awr", "replay-buffer"])
    print(
        f"AWR replay dataset ready at {args.root}: "
        f"{merged.meta.total_episodes} episodes"
    )
    return args.root


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
