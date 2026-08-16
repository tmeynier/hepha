"""Reuse LeRobot's official ACT preprocessing and postprocessing."""

from __future__ import annotations

from typing import Any

import torch
from hepha_lerobot.conditioning import NEXT_TASK_PHASE
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, TransitionKey
from lerobot.processor.converters import batch_to_transition

from .configuration_hepha_act_phase import HephaActPhaseConfig


def _batch_to_transition_with_phase(batch: dict[str, Any]):
    """Use LeRobot's converter while retaining Hepha's auxiliary phase label."""

    transition = batch_to_transition(batch)
    if NEXT_TASK_PHASE not in batch:
        return transition

    complementary_data = dict(
        transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    )
    complementary_data[NEXT_TASK_PHASE] = batch[NEXT_TASK_PHASE]
    transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
    return transition


def make_hepha_act_phase_pre_post_processors(
    config: HephaActPhaseConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build upstream ACT processors while preserving phase supervision."""

    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats)
    preprocessor.to_transition = _batch_to_transition_with_phase
    return preprocessor, postprocessor
