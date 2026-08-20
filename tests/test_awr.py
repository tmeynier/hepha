"""Unit tests for AWR return construction and configuration."""

from __future__ import annotations

import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hepha_lerobot.awr.dataset import AWRAdvantageDataset, AWRReturnDataset
from hepha_lerobot.datasets.builder import create_dataset
from hepha_lerobot.policies.hepha_act_awr.configuration_hepha_act_awr import (
    HephaActAWRConfig,
)
from hepha_lerobot.policies.hepha_act_awr.modeling_hepha_act_awr import (
    HephaActAWRPolicy,
)
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, DONE, OBS_ENV_STATE, OBS_STATE, REWARD


class _RawColumns(dict):
    def select_columns(self, _columns):
        return self


class _TinyRewardDataset:
    def __init__(self) -> None:
        self.repo_id = "hepha/test"
        self.hf_dataset = _RawColumns(
            {
                "index": [10, 11, 20, 21],
                "episode_index": [0, 0, 1, 1],
                "next.reward": [[1.0], [2.0], [-1.0], [4.0]],
            }
        )

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {"index": torch.tensor([10, 11, 20, 21][index])}


def test_discounted_returns_reset_at_episode_boundaries() -> None:
    dataset = AWRReturnDataset(_TinyRewardDataset(), discount=0.5)
    returns = [float(dataset[index]["awr.return"].item()) for index in range(4)]
    assert returns == pytest.approx([2.0, 2.0, 1.0, 4.0])


def test_awr_dataset_can_be_serialized_for_spawn_workers() -> None:
    dataset = AWRReturnDataset(_TinyRewardDataset(), discount=0.5)

    restored = pickle.loads(pickle.dumps(dataset))

    assert len(restored) == 4
    assert restored[0]["awr.return"].item() == pytest.approx(2.0)


def test_awr_config_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="awr_beta"):
        HephaActAWRConfig(awr_beta=0.0)


def test_actor_config_requires_materialized_advantages() -> None:
    with pytest.raises(ValueError, match="awr_advantage_path"):
        HephaActAWRConfig(awr_stage="actor")


def test_actor_dataset_loads_fixed_advantage_weights(tmp_path) -> None:
    path = tmp_path / "advantages.parquet"
    table = pa.table(
        {
            "index": [10, 11, 20, 21],
            "advantage": [1.0, -1.0, 2.0, -2.0],
            "weight": [2.0, 0.5, 3.0, 0.25],
        }
    )
    pq.write_table(table, path)
    dataset = AWRAdvantageDataset(_TinyRewardDataset(), advantage_path=str(path))

    assert dataset[0]["awr.advantage"].item() == pytest.approx(1.0)
    assert dataset[0]["awr.weight"].item() == pytest.approx(2.0)


def _small_awr_policy(*, stage: str) -> HephaActAWRPolicy:
    return HephaActAWRPolicy(
        HephaActAWRConfig(
            device="cpu",
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
                OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            },
            chunk_size=3,
            n_action_steps=1,
            dim_model=16,
            n_heads=4,
            dim_feedforward=32,
            n_encoder_layers=1,
            n_decoder_layers=1,
            latent_dim=4,
            n_vae_encoder_layers=1,
            value_hidden_dim=8,
            value_head_dropout=0.0,
            awr_stage=stage,
            awr_advantage_path="fixed.parquet" if stage == "actor" else None,
        )
    )


def _small_awr_batch() -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(2, 4),
        OBS_ENV_STATE: torch.randn(2, 3),
        ACTION: torch.randn(2, 3, 4),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }


def test_value_stage_updates_only_value_head() -> None:
    policy = _small_awr_policy(stage="value").train()
    batch = {**_small_awr_batch(), "awr.return": torch.tensor([[1.0], [-1.0]])}

    loss, metrics = policy(batch)
    loss.backward()

    assert metrics.keys() >= {"value_loss", "value_mean", "return_mean"}
    assert all(parameter.grad is None for parameter in policy.model.parameters())
    assert any(parameter.grad is not None for parameter in policy.value_head.parameters())
    assert not policy.model.training
    assert all(
        parameter.requires_grad
        for group in policy.get_optim_params()
        for parameter in group["params"]
    )


def test_actor_stage_uses_fixed_weights_and_freezes_value_head() -> None:
    policy = _small_awr_policy(stage="actor").train()
    batch = {
        **_small_awr_batch(),
        "awr.advantage": torch.tensor([[0.5], [-0.5]]),
        "awr.weight": torch.tensor([[2.0], [0.25]]),
    }

    loss, metrics = policy(batch)
    loss.backward()

    assert metrics["awr_weight_mean"] == pytest.approx(1.125)
    assert any(parameter.grad is not None for parameter in policy.model.parameters())
    assert all(parameter.grad is None for parameter in policy.value_head.parameters())
    assert not policy.value_head.training


def test_reward_dataset_features_have_scalar_names(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "hepha_lerobot.datasets.builder.LeRobotDataset.create", fake_create
    )
    backend = type(
        "Backend",
        (),
        {
            "name": "test",
            "action_features": {"joint": float},
            "observation_features": {"joint": float},
        },
    )()
    create_dataset(
        backend=backend,
        repo_id="hepha/test",
        root=tmp_path,
        fps=30,
        use_videos=False,
        include_task_phase=False,
        include_rewards=True,
    )

    assert captured["features"][REWARD]["names"] == ["reward"]
    assert captured["features"][DONE]["names"] == ["done"]
