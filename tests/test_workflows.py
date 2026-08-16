from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from hepha_lerobot.evaluation.conditioned_rollout import (
    _apply_inference_overrides,
)
from hepha_lerobot.evaluation.rollout import (
    build_rollout_command,
)
from hepha_lerobot.evaluation.rollout import (
    parse_args as parse_rollout_args,
)
from hepha_lerobot.policies import available_policy_types
from hepha_lerobot.recording.record import parse_args as parse_record_args
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


def test_evaluate_command_uses_conditioned_rollout(monkeypatch) -> None:
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
        drawer_index=4,
        seed=7,
        device="cpu",
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
        headless=True,
        debug=False,
        display_data=False,
    )
    command = build_rollout_command(args)

    assert "hepha_lerobot.evaluation.conditioned_rollout" in command
    assert "--backend=mujoco" in command
    assert "--backend-option=model_path=custom.xml" in command
    assert "--drawer-index=4" in command
    assert "--task=move" in command
    assert "--viewer=false" in command
    assert "--n-action-steps=1" in command
    assert "--temporal-ensemble-coeff=0.01" in command


def test_act_inference_overrides_enable_per_frame_temporal_ensembling() -> None:
    config = Namespace(
        type="act",
        chunk_size=100,
        n_action_steps=10,
        temporal_ensemble_coeff=None,
    )

    _apply_inference_overrides(
        config,
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
    )

    assert config.n_action_steps == 1
    assert config.temporal_ensemble_coeff == 0.01


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
            "--drawer-index",
            "7",
        ],
    )
    args = parse_rollout_args()

    assert args.backend == "mujoco"
    assert args.duration == 2
    assert args.headless
    assert args.drawer_index == 7


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


def test_record_viewer_accepts_flag_or_explicit_boolean(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["hepha-record", "--viewer"])
    assert parse_record_args().viewer is True

    monkeypatch.setattr("sys.argv", ["hepha-record", "--viewer", "false"])
    assert parse_record_args().viewer is False

    monkeypatch.setattr("sys.argv", ["hepha-record", "--debug", "true"])
    assert parse_record_args().debug is True
