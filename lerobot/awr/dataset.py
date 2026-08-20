"""Compute Monte-Carlo return targets from native LeRobot reward columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from hepha_lerobot.policies.hepha_act_awr.processor_hepha_act_awr import AWR_RETURN
from lerobot.utils.constants import REWARD
from torch.utils.data import Dataset


class AWRReturnDataset(Dataset):
    """Transparent dataset wrapper adding a discounted ``awr.return`` tensor."""

    def __init__(self, dataset: Dataset, *, discount: float) -> None:
        self.dataset = dataset
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
        self._returns = dict(zip(indices.tolist(), returns.tolist(), strict=True))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        absolute_index = int(item["index"])
        item[AWR_RETURN] = torch.tensor(
            [self._returns[absolute_index]], dtype=torch.float32
        )
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)
