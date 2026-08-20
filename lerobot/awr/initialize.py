"""Convert an original ACT checkpoint into an AWR actor-critic checkpoint."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import fields
from pathlib import Path

from hepha_lerobot.policies.hepha_act_awr import HephaActAWRConfig, HephaActAWRPolicy
from lerobot.configs import PreTrainedConfig
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot.policies import get_policy_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original LeRobot ACT checkpoint")
    parser.add_argument("destination", type=Path, help="New hepha_act_awr checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if not args.source.is_dir():
        raise FileNotFoundError(f"ACT checkpoint not found: {args.source}")
    if args.destination.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination exists: {args.destination}; pass --overwrite to replace it"
            )
        shutil.rmtree(args.destination)

    register_third_party_plugins()
    source_config = PreTrainedConfig.from_pretrained(args.source)
    if source_config.type != "act":
        raise ValueError(f"Expected an original ACT checkpoint, got {source_config.type!r}")
    source_policy = get_policy_class("act").from_pretrained(
        args.source, config=source_config
    )
    kwargs = {
        field.name: getattr(source_config, field.name)
        for field in fields(HephaActAWRConfig)
        if field.init and hasattr(source_config, field.name)
    }
    target_config = HephaActAWRConfig(**kwargs)
    target_config.pretrained_path = None
    target_policy = HephaActAWRPolicy(target_config)
    incompatible = target_policy.load_state_dict(source_policy.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected ACT weights: {incompatible.unexpected_keys}")
    missing_non_value = [
        key for key in incompatible.missing_keys if not key.startswith("value_head.")
    ]
    if missing_non_value:
        raise RuntimeError(f"Missing non-value ACT weights: {missing_non_value}")
    target_policy.save_pretrained(args.destination)
    for pattern in ("policy_preprocessor*", "policy_postprocessor*"):
        for source_file in args.source.glob(pattern):
            if source_file.is_file():
                shutil.copy2(source_file, args.destination / source_file.name)
    print(
        f"Created {args.destination} from {args.source}; ACT weights copied and "
        "value head initialized randomly."
    )
    return args.destination


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
