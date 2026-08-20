"""Launch official LeRobot training for any registered policy type."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hepha_lerobot.cli import find_environment_executable, passthrough_arguments
from hepha_lerobot.policies import available_policy_types
from lerobot.utils.device_utils import auto_select_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="hepha/simulation_ik")
    parser.add_argument("--root", type=Path, default=Path("datasets/hepha_simulation_ik"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--policy-type", default="act")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Initialize training from a checkpoint of --policy-type",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--policy-repo-id", default=None)
    parser.add_argument("--list-policies", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, lerobot_args = parser.parse_known_args()
    args.lerobot_args = lerobot_args
    return args


def resolve_device(requested: str) -> str:
    return auto_select_torch_device().type if requested == "auto" else requested


def build_train_command(args: argparse.Namespace) -> list[str]:
    policy_type = args.policy_type
    policy_path = getattr(args, "policy_path", None)
    output_dir = args.output_dir or Path("outputs") / policy_type
    job_name = args.job_name or f"hepha_{policy_type}"
    if policy_type == "hepha_act_phase":
        executable = [sys.executable, "-m", "hepha_lerobot.training.phase_train"]
    elif policy_type == "hepha_act_awr":
        executable = [sys.executable, "-m", "hepha_lerobot.training.awr_train"]
    else:
        executable = [find_environment_executable("lerobot-train")]
    command = [
        *executable,
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.root}",
        (
            f"--policy.path={policy_path}"
            if policy_path is not None
            else f"--policy.type={policy_type}"
        ),
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={resolve_device(args.device)}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--persistent_workers={str(args.num_workers > 0).lower()}",
        f"--wandb.enable={str(args.wandb).lower()}",
        f"--policy.push_to_hub={str(bool(args.policy_repo_id)).lower()}",
    ]
    if args.policy_repo_id:
        command.append(f"--policy.repo_id={args.policy_repo_id}")
    return [*command, *passthrough_arguments(args.lerobot_args)]


def main() -> None:
    args = parse_args()
    policies = available_policy_types()
    if args.list_policies:
        print("\n".join(policies))
        return
    if args.policy_type not in policies:
        raise ValueError(
            f"Policy {args.policy_type!r} is not registered by LeRobot; "
            f"available: {', '.join(policies)}"
        )
    if args.steps <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError(
            "--steps and --batch-size must be positive; --num-workers cannot be negative"
        )
    if not args.root.is_dir() and not args.dry_run:
        raise FileNotFoundError(f"LeRobot dataset root not found: {args.root}")
    if args.policy_path is not None and not args.policy_path.is_dir() and not args.dry_run:
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")
    if args.policy_path is not None and args.policy_path.is_dir():
        from lerobot.configs import PreTrainedConfig
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        checkpoint_type = PreTrainedConfig.from_pretrained(args.policy_path).type
        if checkpoint_type != args.policy_type:
            raise ValueError(
                f"--policy-path contains {checkpoint_type!r}, but --policy-type is "
                f"{args.policy_type!r}. Convert the checkpoint first if needed."
            )
    command = build_train_command(args)
    print(" ".join(map(str, command)))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
