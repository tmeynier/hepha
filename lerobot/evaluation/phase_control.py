"""Monotonic task-phase control for phase-aware policy rollouts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from hepha_lerobot.conditioning import TASK_PHASE_COUNT, validate_task_phase


@dataclass
class PhaseTransitionState:
    """Advance phases using two matching votes among three recent predictions."""

    current_phase: int = 1
    vote_window: int = 3
    required_votes: int = 2
    _predictions: deque[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.current_phase = validate_task_phase(self.current_phase)
        if self.vote_window <= 0:
            raise ValueError("vote_window must be positive")
        if not 1 <= self.required_votes <= self.vote_window:
            raise ValueError("required_votes must be in [1, vote_window]")
        self._predictions = deque(maxlen=self.vote_window)

    @property
    def predictions(self) -> tuple[int, ...]:
        return tuple(self._predictions)

    def observe(self, predicted_phase: int) -> bool:
        """Record one prediction and advance only to the immediate next phase."""

        predicted_phase = validate_task_phase(predicted_phase)
        self._predictions.append(predicted_phase)
        if self.current_phase >= TASK_PHASE_COUNT:
            return False
        if len(self._predictions) < self.vote_window:
            return False

        requested_next_phase = self.current_phase + 1
        votes = sum(
            prediction == requested_next_phase for prediction in self._predictions
        )
        if votes < self.required_votes:
            return False

        self.current_phase = requested_next_phase
        self._predictions.clear()
        return True
