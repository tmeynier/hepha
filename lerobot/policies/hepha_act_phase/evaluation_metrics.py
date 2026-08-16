"""Accumulate held-out phase metrics for LeRobot's W&B evaluation event."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from hepha_lerobot.conditioning import TASK_PHASE_COUNT
from torch import Tensor


@dataclass
class _PhaseEvalAccumulator:
    values: list[float] = field(
        default_factory=lambda: [0.0] * (7 + 2 * TASK_PHASE_COUNT)
    )

    def reset(self) -> None:
        self.values = [0.0] * len(self.values)

    def update(
        self,
        *,
        phase_loss_sum: Tensor,
        prediction: Tensor,
        target: Tensor,
        current_phase: Tensor,
    ) -> None:
        correct = prediction.eq(target)
        transition = target.ne(current_phase)
        hold = ~transition
        statistics = torch.zeros(
            len(self.values), dtype=torch.float32, device=target.device
        )
        statistics[0] = phase_loss_sum.detach().float()
        statistics[1] = target.numel()
        statistics[2] = correct.sum()
        statistics[3] = transition.sum()
        statistics[4] = correct[transition].sum()
        statistics[5] = hold.sum()
        statistics[6] = correct[hold].sum()
        for phase_index in range(TASK_PHASE_COUNT):
            phase_mask = target.eq(phase_index)
            offset = 7 + 2 * phase_index
            statistics[offset] = phase_mask.sum()
            statistics[offset + 1] = correct[phase_mask].sum()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
        for index, value in enumerate(statistics.cpu().tolist()):
            self.values[index] += value

    def metrics(self) -> dict[str, float]:
        total = self.values[1]
        if total <= 0:
            return {}
        metrics = {
            "phase_loss": self.values[0] / total,
            "phase_accuracy": self.values[2] / total,
        }
        transition_total = self.values[3]
        if transition_total > 0:
            metrics["phase_transition_accuracy"] = (
                self.values[4] / transition_total
            )
        hold_total = self.values[5]
        if hold_total > 0:
            metrics["phase_hold_accuracy"] = self.values[6] / hold_total
        for phase_index in range(TASK_PHASE_COUNT):
            offset = 7 + 2 * phase_index
            phase_total = self.values[offset]
            if phase_total > 0:
                metrics[f"phase_{phase_index + 1}_accuracy"] = (
                    self.values[offset + 1] / phase_total
                )
        return metrics


_ACCUMULATOR = _PhaseEvalAccumulator()


def reset_phase_eval_metrics() -> None:
    _ACCUMULATOR.reset()


def accumulate_phase_eval_metrics(
    *,
    phase_loss_sum: Tensor,
    prediction: Tensor,
    target: Tensor,
    current_phase: Tensor,
) -> None:
    _ACCUMULATOR.update(
        phase_loss_sum=phase_loss_sum,
        prediction=prediction,
        target=target,
        current_phase=current_phase,
    )


def consume_phase_eval_metrics() -> dict[str, float]:
    metrics = _ACCUMULATOR.metrics()
    _ACCUMULATOR.reset()
    return metrics
