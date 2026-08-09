"""List every policy registered by the installed LeRobot distribution."""

from __future__ import annotations


def available_policy_types() -> tuple[str, ...]:
    # Importing this public package registers every built-in policy config. Plugin
    # discovery then adds externally installed lerobot_policy_* distributions.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    import lerobot.policies  # noqa: F401

    register_third_party_plugins()
    return tuple(sorted(PreTrainedConfig.get_known_choices()))


def main() -> None:
    print("\n".join(available_policy_types()))


if __name__ == "__main__":
    main()
