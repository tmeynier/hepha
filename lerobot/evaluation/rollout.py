"""Deploy any LeRobot policy through any registered Hepha backend."""

from __future__ import annotations

import argparse
import math
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
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help=(
            "Override how many predicted actions are executed before replanning; "
            "use 1 for closed-loop replanning every frame"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help=(
            "Enable ACT temporal ensembling at inference (original ACT uses 0.01); "
            "requires --n-action-steps 1"
        ),
    )
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
    n_action_steps = getattr(args, "n_action_steps", None)
    temporal_ensemble_coeff = getattr(args, "temporal_ensemble_coeff", None)
    if n_action_steps is not None:
        command.append(f"--n-action-steps={n_action_steps}")
    if temporal_ensemble_coeff is not None:
        command.append(f"--temporal-ensemble-coeff={temporal_ensemble_coeff}")
    return command


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("--duration and --fps must be positive")
    if args.n_action_steps is not None and args.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    if args.temporal_ensemble_coeff is not None:
        if not math.isfinite(args.temporal_ensemble_coeff):
            raise ValueError("--temporal-ensemble-coeff must be finite")
        if args.temporal_ensemble_coeff < 0.0:
            raise ValueError("--temporal-ensemble-coeff must be non-negative")
        if args.n_action_steps != 1:
            raise ValueError(
                "Temporal ensembling requires --n-action-steps 1"
            )
    validate_drawer_index(args.drawer_index)
    if not args.policy_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")
    command = build_rollout_command(args)
    print(" ".join(map(str, command)))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
