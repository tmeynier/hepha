"""Accumulate detailed AWR metrics for LeRobot's held-out W&B event."""

from __future__ import annotations

from dataclasses import dataclass, field

_METRIC_NAMES = (
    "actor_loss",
    "value_loss",
    "value_mean",
    "return_mean",
    "advantage_mean",
    "awr_weight_mean",
    "awr_weight_max",
)


@dataclass
class _Accumulator:
    sums: dict[str, float] = field(default_factory=dict)
    samples: int = 0
    maximum_weight: float = 0.0

    def reset(self) -> None:
        self.sums.clear()
        self.samples = 0
        self.maximum_weight = 0.0

    def update(self, metrics: dict[str, float], batch_size: int) -> None:
        for name in _METRIC_NAMES[:-1]:
            self.sums[name] = self.sums.get(name, 0.0) + metrics[name] * batch_size
        self.samples += batch_size
        self.maximum_weight = max(self.maximum_weight, metrics["awr_weight_max"])

    def consume(self) -> dict[str, float]:
        if self.samples == 0:
            return {}
        result = {
            name: total / self.samples for name, total in self.sums.items()
        }
        result["awr_weight_max"] = self.maximum_weight
        self.reset()
        return result


_ACCUMULATOR = _Accumulator()


def reset_awr_eval_metrics() -> None:
    _ACCUMULATOR.reset()


def accumulate_awr_eval_metrics(metrics: dict[str, float], batch_size: int) -> None:
    _ACCUMULATOR.update(metrics, batch_size)


def consume_awr_eval_metrics() -> dict[str, float]:
    return _ACCUMULATOR.consume()
