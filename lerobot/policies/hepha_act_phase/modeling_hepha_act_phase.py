"""Phase-aware ACT implemented as a minimal extension of upstream LeRobot ACT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from hepha_lerobot.conditioning import (
    NEXT_TASK_PHASE,
    TASK_PHASE_CONDITION_NAMES,
    TASK_PHASE_COUNT,
)
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_ENV_STATE
from torch import Tensor, nn

from .configuration_hepha_act_phase import HephaActPhaseConfig
from .evaluation_metrics import (
    accumulate_phase_eval_metrics,
    reset_phase_eval_metrics,
)


class HephaActPhasePolicy(ACTPolicy):
    """ACT action prediction plus a shared-decoder next-phase classifier.

    The upstream ACT model and its action head are left unchanged. A forward
    pre-hook observes the exact decoder features passed into ACT's action head;
    their temporal mean is passed through a small classification head. This
    keeps the complete ACT action path and checkpoint behavior native to
    LeRobot while allowing gradients from phase classification to update the
    shared visual backbone and transformer.
    """

    config_class = HephaActPhaseConfig
    name = "hepha_act_phase"

    def __init__(
        self,
        config: HephaActPhaseConfig,
        *,
        dataset_meta: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.config = config
        self.phase_head = nn.Sequential(
            nn.LayerNorm(config.dim_model),
            nn.Dropout(config.phase_head_dropout),
            nn.Linear(config.dim_model, config.phase_count),
        )
        self._phase_logits: Tensor | None = None
        self._phase_feature_hook = self.model.action_head.register_forward_pre_hook(
            self._capture_phase_logits
        )
        if dataset_meta is not None:
            self._validate_phase_dataset(dataset_meta)

    def _validate_phase_dataset(self, dataset_meta: Any) -> None:
        features = getattr(dataset_meta, "features", {})
        if NEXT_TASK_PHASE not in features:
            raise ValueError(
                f"Dataset {getattr(dataset_meta, 'repo_id', '<unknown>')!r} has no "
                f"{NEXT_TASK_PHASE!r} supervision feature"
            )
        environment_feature = features.get(OBS_ENV_STATE, {})
        names = tuple(environment_feature.get("names", ()))
        if names[-TASK_PHASE_COUNT:] != TASK_PHASE_CONDITION_NAMES:
            raise ValueError(
                f"The final {TASK_PHASE_COUNT} {OBS_ENV_STATE} values must be the "
                "current-task-phase one-hot condition"
            )

    def _capture_phase_logits(
        self,
        _module: nn.Module,
        inputs: tuple[Tensor, ...],
    ) -> None:
        decoder_features = inputs[0]
        if decoder_features.ndim != 3:
            raise ValueError(
                "ACT decoder features must have shape (batch, chunk, hidden), "
                f"got {tuple(decoder_features.shape)}"
            )
        self._phase_logits = self.phase_head(decoder_features.mean(dim=1))

    def _require_phase_logits(self) -> Tensor:
        if self._phase_logits is None:
            raise RuntimeError("ACT did not produce decoder features for phase prediction")
        return self._phase_logits

    def _phase_metrics(
        self,
        batch: dict[str, Tensor],
        phase_logits: Tensor,
        phase_target: Tensor,
    ) -> dict[str, float]:
        prediction = phase_logits.argmax(dim=-1)
        correct = prediction.eq(phase_target)
        metrics = {"phase_accuracy": correct.float().mean().item()}

        environment_state = batch.get(OBS_ENV_STATE)
        if environment_state is not None:
            current_phase = environment_state[..., -TASK_PHASE_COUNT:].argmax(dim=-1)
            transition_mask = phase_target.ne(current_phase)
            hold_mask = ~transition_mask
            if transition_mask.any():
                metrics["phase_transition_accuracy"] = (
                    correct[transition_mask].float().mean().item()
                )
            if hold_mask.any():
                metrics["phase_hold_accuracy"] = correct[hold_mask].float().mean().item()

        for phase_index in range(self.config.phase_count):
            phase_mask = phase_target.eq(phase_index)
            if phase_mask.any():
                metrics[f"phase_{phase_index + 1}_accuracy"] = (
                    correct[phase_mask].float().mean().item()
                )
        return metrics

    def train(self, mode: bool = True) -> HephaActPhasePolicy:
        entering_evaluation = self.training and not mode
        policy = super().train(mode)
        if entering_evaluation:
            reset_phase_eval_metrics()
        return policy

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Compute the unmodified ACT objective plus phase cross-entropy."""

        if NEXT_TASK_PHASE not in batch:
            raise KeyError(
                f"Training {self.name!r} requires dataset feature {NEXT_TASK_PHASE!r}"
            )
        act_loss, act_metrics = super().forward(batch)
        phase_logits = self._require_phase_logits()
        phase_target = batch[NEXT_TASK_PHASE].long().reshape(-1)
        if phase_target.shape[0] != phase_logits.shape[0]:
            raise ValueError(
                f"Phase target batch has {phase_target.shape[0]} rows but logits have "
                f"{phase_logits.shape[0]}"
            )
        phase_loss = F.cross_entropy(phase_logits, phase_target)
        if not self.training:
            current_phase = batch[OBS_ENV_STATE][
                ..., -TASK_PHASE_COUNT:
            ].argmax(dim=-1)
            accumulate_phase_eval_metrics(
                phase_loss_sum=F.cross_entropy(
                    phase_logits, phase_target, reduction="sum"
                ),
                prediction=phase_logits.argmax(dim=-1),
                target=phase_target,
                current_phase=current_phase,
            )
        loss = act_loss + self.config.phase_loss_weight * phase_loss
        metrics = {
            **act_metrics,
            "act_loss": act_loss.item(),
            "phase_loss": phase_loss.item(),
            "weighted_phase_loss": (
                self.config.phase_loss_weight * phase_loss.detach()
            ).item(),
            **self._phase_metrics(batch, phase_logits, phase_target),
        }
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk_with_phase(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Predict an ACT action chunk and five next-phase logits in one pass."""

        actions = super().predict_action_chunk(batch)
        return actions, self._require_phase_logits()

    @torch.no_grad()
    def select_action_with_phase(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Select one ACT action and return the most recently computed phase logits.

        For a fresh phase prediction every control frame, use ACT with
        ``n_action_steps=1``. With a longer action queue, logits remain associated
        with the observation that generated that queue.
        """

        action = super().select_action(batch)
        return action, self._require_phase_logits()
