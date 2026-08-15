"""Deploy any LeRobot policy through any registered Hepha backend."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hepha_lerobot.cli import find_environment_executable
from hepha_lerobot.conditioning import DEFAULT_TASK, validate_drawer_index
from hepha_lerobot.training.train import resolve_device

from simulation.view import _parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_path", type=Path)
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--drawer-index", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--debug", nargs="?", const=True, default=False, type=_parse_bool
    )
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser.parse_args()


def build_rollout_command(args: argparse.Namespace) -> list[str]:
    mjpython = find_environment_executable("mjpython")
    executable = (
        mjpython
        if sys.platform == "darwin"
        and not args.headless
        and args.backend == "mujoco"
        and Path(mjpython).is_file()
        else sys.executable
    )
    command = [
        executable,
        "-m",
        "hepha_lerobot.evaluation.conditioned_rollout",
        str(args.policy_path),
    ]
    command.extend(
        [
            f"--backend={args.backend}",
            f"--duration={args.duration}",
            f"--fps={args.fps}",
            f"--width={args.width}",
            f"--height={args.height}",
            f"--camera={args.camera}",
            f"--task={args.task}",
            f"--drawer-index={args.drawer_index}",
            f"--seed={getattr(args, 'seed', 0)}",
            f"--device={resolve_device(args.device)}",
            f"--viewer={str(not args.headless).lower()}",
            f"--debug={str(getattr(args, 'debug', False)).lower()}",
            f"--display-data={str(args.display_data).lower()}",
        ]
    )
    for backend_option in args.backend_option:
        command.append(f"--backend-option={backend_option}")
    return command


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("--duration and --fps must be positive")
    validate_drawer_index(args.drawer_index)
    if not args.policy_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")
    command = build_rollout_command(args)
    print(" ".join(map(str, command)))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
