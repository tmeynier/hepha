"""Unit tests for AWR return construction and configuration."""

from __future__ import annotations

import pytest
import torch
from hepha_lerobot.awr.dataset import AWRReturnDataset
from hepha_lerobot.datasets.builder import create_dataset
from hepha_lerobot.policies.hepha_act_awr.configuration_hepha_act_awr import (
    HephaActAWRConfig,
)
from lerobot.utils.constants import DONE, REWARD


class _RawColumns(dict):
    def select_columns(self, _columns):
        return self


class _TinyRewardDataset:
    def __init__(self) -> None:
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


def test_awr_config_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="awr_beta"):
        HephaActAWRConfig(awr_beta=0.0)


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
