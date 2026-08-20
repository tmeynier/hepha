"""Dataset adapters for the two ordered AWR regression stages."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from hepha_lerobot.policies.hepha_act_awr.processor_hepha_act_awr import (
    AWR_ADVANTAGE,
    AWR_RETURN,
    AWR_WEIGHT,
)
from lerobot.utils.constants import REWARD
from torch.utils.data import Dataset


def discounted_return_map(dataset: Dataset, *, discount: float) -> dict[int, float]:
    """Compute a Monte-Carlo return for every absolute LeRobot frame index."""
    raw = dataset.hf_dataset.select_columns(["index", "episode_index", REWARD])
    indices = np.asarray(raw["index"], dtype=np.int64)
    episodes = np.asarray(raw["episode_index"], dtype=np.int64)
    rewards = np.asarray(raw[REWARD], dtype=np.float32).reshape(-1)
    returns = np.empty_like(rewards)
    running = 0.0
    previous_episode: int | None = None
    for row in range(len(rewards) - 1, -1, -1):
        episode = int(episodes[row])
        if previous_episode is None or episode != previous_episode:
            running = 0.0
        running = float(rewards[row]) + discount * running
        returns[row] = running
        previous_episode = episode
    return dict(zip(indices.tolist(), returns.tolist(), strict=True))


class _AWRDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


class AWRReturnDataset(_AWRDataset):
    """Attach discounted returns during the value-regression stage."""

    def __init__(self, dataset: Dataset, *, discount: float) -> None:
        super().__init__(dataset)
        self._returns = discounted_return_map(dataset, discount=discount)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        absolute_index = int(item["index"])
        item[AWR_RETURN] = torch.tensor(
            [self._returns[absolute_index]], dtype=torch.float32
        )
        return item


class AWRAdvantageDataset(_AWRDataset):
    """Attach immutable critic advantages and weights during actor regression."""

    def __init__(self, dataset: Dataset, *, advantage_path: str) -> None:
        super().__init__(dataset)
        table = pq.read_table(advantage_path, columns=["index", "advantage", "weight"])
        metadata = table.schema.metadata or {}
        awr_metadata = json.loads(metadata.get(b"hepha_awr", b"{}").decode())
        expected_repo_id = awr_metadata.get("repo_id")
        if expected_repo_id and expected_repo_id != dataset.repo_id:
            raise ValueError(
                f"Advantage file belongs to {expected_repo_id!r}, not "
                f"{dataset.repo_id!r}"
            )
        indices = np.asarray(table["index"], dtype=np.int64)
        advantages = np.asarray(table["advantage"], dtype=np.float32)
        weights = np.asarray(table["weight"], dtype=np.float32)
        self._advantages = dict(
            zip(indices.tolist(), advantages.tolist(), strict=True)
        )
        self._weights = dict(zip(indices.tolist(), weights.tolist(), strict=True))
        dataset_indices = set(
            np.asarray(dataset.hf_dataset["index"], dtype=np.int64).tolist()
        )
        missing = dataset_indices - self._weights.keys()
        if missing:
            raise ValueError(
                f"Advantage file is missing {len(missing)} dataset frame(s); "
                "recompute it for this exact replay dataset"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        absolute_index = int(item["index"])
        item[AWR_ADVANTAGE] = torch.tensor(
            [self._advantages[absolute_index]], dtype=torch.float32
        )
        item[AWR_WEIGHT] = torch.tensor(
            [self._weights[absolute_index]], dtype=torch.float32
        )
        return item
