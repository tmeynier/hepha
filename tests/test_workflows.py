from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from hepha_lerobot.evaluation.rollout import (
    build_rollout_command,
)
from hepha_lerobot.evaluation.rollout import (
    parse_args as parse_rollout_args,
)
from hepha_lerobot.policies import available_policy_types
from hepha_lerobot.training.train import build_train_command
from hepha_lerobot.training.train import parse_args as parse_train_args


def test_train_command_forwards_any_registered_policy_to_lerobot() -> None:
    args = Namespace(
        repo_id="hepha/test",
        root=Path("datasets/test"),
        output_dir=Path("outputs/test"),
        job_name="test",
        policy_type="diffusion",
        device="cpu",
        steps=3,
        batch_size=2,
        num_workers=0,
        wandb=False,
        policy_repo_id=None,
        lerobot_args=["--", "--policy.n_action_steps=8"],
    )
    command = build_train_command(args)

    assert "act" in available_policy_types()
    assert "diffusion" in available_policy_types()
    assert "--policy.type=diffusion" in command
    assert "--dataset.repo_id=hepha/test" in command
    assert "--dataset.root=datasets/test" in command
    assert "--policy.push_to_hub=false" in command
    assert "--policy.n_action_steps=8" in command


def test_evaluate_command_uses_plugin_and_rollout(monkeypatch) -> None:
    monkeypatch.setattr("hepha_lerobot.evaluation.rollout.sys.platform", "linux")
    args = Namespace(
        policy_path=Path("outputs/test/checkpoint"),
        backend="mujoco",
        backend_option=["model_path=custom.xml"],
        duration=1.0,
        fps=30,
        width=256,
        height=256,
        camera="head_camera",
        task="move",
        device="cpu",
        headless=True,
        display_data=False,
        lerobot_args=[],
    )
    command = build_rollout_command(args)

    assert "--strategy.type=base" in command
    assert "--robot.type=hepha_simulation" in command
    assert "--robot.backend=mujoco" in command
    assert '--robot.backend_options={"model_path": "custom.xml"}' in command
    assert "--robot.viewer=false" in command


def test_rollout_options_after_policy_path_are_not_forwarded(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "hepha-evaluate",
            "checkpoint",
            "--backend",
            "mujoco",
            "--duration",
            "2",
            "--headless",
            "--inference.type=sync",
        ],
    )
    args = parse_rollout_args()

    assert args.backend == "mujoco"
    assert args.duration == 2
    assert args.headless
    assert args.lerobot_args == ["--inference.type=sync"]


def test_training_forwards_only_unknown_lerobot_options(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "hepha-train",
            "--policy-type",
            "act",
            "--steps",
            "2",
            "--",
            "--policy.chunk_size=16",
        ],
    )
    args = parse_train_args()

    assert args.policy_type == "act"
    assert args.steps == 2
    assert args.lerobot_args == ["--", "--policy.chunk_size=16"]
