"""Official ACT processors with the scalar AWR return retained."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, TransitionKey
from lerobot.processor.converters import batch_to_transition

from .configuration_hepha_act_awr import HephaActAWRConfig

AWR_RETURN = "awr.return"
AWR_ADVANTAGE = "awr.advantage"
AWR_WEIGHT = "awr.weight"


def _batch_to_transition_with_awr(batch: dict[str, Any]):
    transition = batch_to_transition(batch)
    awr_data = {
        key: batch[key]
        for key in (AWR_RETURN, AWR_ADVANTAGE, AWR_WEIGHT)
        if key in batch
    }
    if awr_data:
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary.update(awr_data)
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
    return transition


def make_hepha_act_awr_pre_post_processors(
    config: HephaActAWRConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats)
    preprocessor.to_transition = _batch_to_transition_with_awr
    return preprocessor, postprocessor


# Kept for checkpoints or external code created by the first Hepha AWR version.
_batch_to_transition_with_return = _batch_to_transition_with_awr
