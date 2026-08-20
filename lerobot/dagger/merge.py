"""Merge a human-correction dataset into an existing LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset, merge_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo-id", required=True)
    parser.add_argument("--base-root", type=Path, default=None)
    parser.add_argument("--correction-repo-id", required=True)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Merged output repository ID")
    parser.add_argument("--root", type=Path, required=True, help="Merged output directory")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _same_path(left: Path | None, right: Path) -> bool:
    return left is not None and left.expanduser().resolve() == right.expanduser().resolve()


def run(args: argparse.Namespace) -> Path:
    if _same_path(args.base_root, args.root) or _same_path(args.correction_root, args.root):
        raise ValueError("Merged output --root must differ from both input dataset roots")
    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Merged dataset root already exists: {args.root}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.root)

    base = LeRobotDataset(repo_id=args.base_repo_id, root=args.base_root)
    corrections = LeRobotDataset(
        repo_id=args.correction_repo_id,
        root=args.correction_root,
    )
    if base.fps != corrections.fps:
        raise ValueError(
            f"Dataset FPS mismatch: base={base.fps}, corrections={corrections.fps}"
        )
    if base.features != corrections.features:
        raise ValueError(
            "Dataset feature schemas do not match. Record corrections with the same "
            "camera, video mode, drawer condition, and phase condition as the base dataset."
        )

    merged = merge_datasets(
        [base, corrections],
        output_repo_id=args.repo_id,
        output_dir=args.root,
        concatenate_videos=False,
        concatenate_data=False,
    )
    if args.push_to_hub:
        merged.push_to_hub(tags=["hepha", "mujoco", "dagger", "human-correction"])
    print(
        f"Merged LeRobot dataset ready at {args.root}: "
        f"{merged.meta.total_episodes} episodes"
    )
    return args.root


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
