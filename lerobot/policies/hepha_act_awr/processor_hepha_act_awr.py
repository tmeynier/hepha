"""Official ACT processors with the scalar AWR return retained."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, TransitionKey
from lerobot.processor.converters import batch_to_transition

from .configuration_hepha_act_awr import HephaActAWRConfig

AWR_RETURN = "awr.return"


def _batch_to_transition_with_return(batch: dict[str, Any]):
    transition = batch_to_transition(batch)
    if AWR_RETURN in batch:
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary[AWR_RETURN] = batch[AWR_RETURN]
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
    preprocessor.to_transition = _batch_to_transition_with_return
    return preprocessor, postprocessor
