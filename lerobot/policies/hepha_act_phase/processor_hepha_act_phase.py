"""Reuse LeRobot's official ACT preprocessing and postprocessing."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_hepha_act_phase import HephaActPhaseConfig


def make_hepha_act_phase_pre_post_processors(
    config: HephaActPhaseConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the same processors used by upstream ACT."""

    return make_act_pre_post_processors(config, dataset_stats)
