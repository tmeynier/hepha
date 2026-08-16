"""Configuration for the phase-aware Hepha ACT policy."""

from __future__ import annotations

from dataclasses import dataclass

from hepha_lerobot.conditioning import TASK_PHASE_COUNT
from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("hepha_act_phase")
@dataclass
class HephaActPhaseConfig(ACTConfig):
    """Standard ACT with an auxiliary categorical task-phase head."""

    phase_count: int = TASK_PHASE_COUNT
    phase_loss_weight: float = 0.1
    phase_head_dropout: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.phase_count != TASK_PHASE_COUNT:
            raise ValueError(
                f"Hepha recordings define {TASK_PHASE_COUNT} task phases; "
                f"got phase_count={self.phase_count}"
            )
        if self.phase_loss_weight < 0.0:
            raise ValueError("phase_loss_weight must be non-negative")
        if not 0.0 <= self.phase_head_dropout < 1.0:
            raise ValueError("phase_head_dropout must be in [0, 1)")
