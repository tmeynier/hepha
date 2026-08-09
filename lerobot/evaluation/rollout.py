"""Deploy any LeRobot policy through any registered Hepha backend."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hepha_lerobot.cli import find_environment_executable, passthrough_arguments
from hepha_lerobot.recording.record import DEFAULT_TASK
from hepha_lerobot.training.train import resolve_device

from simulation.base import parse_backend_options


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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    args, lerobot_args = parser.parse_known_args()
    args.lerobot_args = lerobot_args
    return args


def build_rollout_command(args: argparse.Namespace) -> list[str]:
    mjpython = find_environment_executable("mjpython")
    module_mode = (
        sys.platform == "darwin"
        and not args.headless
        and args.backend == "mujoco"
        and Path(mjpython).is_file()
    )
    command = (
        [mjpython, "-m", "lerobot.scripts.lerobot_rollout"]
        if module_mode
        else [find_environment_executable("lerobot-rollout")]
    )
    backend_options = parse_backend_options(args.backend_option)
    command.extend(
        [
            "--strategy.type=base",
            f"--policy.path={args.policy_path}",
            f"--policy.device={resolve_device(args.device)}",
            "--robot.type=hepha_simulation",
            f"--robot.backend={args.backend}",
            f"--robot.backend_options={json.dumps(backend_options)}",
            f"--robot.camera={args.camera}",
            f"--robot.width={args.width}",
            f"--robot.height={args.height}",
            f"--robot.fps={args.fps}",
            f"--robot.viewer={str(not args.headless).lower()}",
            f"--fps={args.fps}",
            f"--duration={args.duration}",
            f"--task={args.task}",
            f"--display_data={str(args.display_data).lower()}",
        ]
    )
    return [*command, *passthrough_arguments(args.lerobot_args)]


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("--duration and --fps must be positive")
    if not args.policy_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")
    command = build_rollout_command(args)
    print(" ".join(map(str, command)))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
