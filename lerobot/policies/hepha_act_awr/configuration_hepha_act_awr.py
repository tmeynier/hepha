"""Configuration for ACT with a shared value head and AWR objective."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("hepha_act_awr")
@dataclass
class HephaActAWRConfig(ACTConfig):
    """Upstream ACT plus a small observation-value head."""

    awr_discount: float = 0.999
    awr_beta: float = 1.0
    awr_max_weight: float = 20.0
    awr_normalize_advantage: bool = True
    value_loss_weight: float = 0.5
    value_hidden_dim: int = 256
    value_head_dropout: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.awr_discount <= 1.0:
            raise ValueError("awr_discount must be in [0, 1]")
        if self.awr_beta <= 0.0:
            raise ValueError("awr_beta must be positive")
        if self.awr_max_weight < 1.0:
            raise ValueError("awr_max_weight must be at least 1")
        if self.value_loss_weight < 0.0 or self.value_hidden_dim <= 0:
            raise ValueError("value loss weight must be non-negative and hidden dim positive")
        if not 0.0 <= self.value_head_dropout < 1.0:
            raise ValueError("value_head_dropout must be in [0, 1)")
