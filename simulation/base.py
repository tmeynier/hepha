"""Backend-neutral contracts for Hepha simulation environments."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Self

FeatureSpec = type | tuple[int, int, int]


@dataclass(frozen=True)
class SimulationConfig:
    """Settings shared by every simulation backend."""

    camera: str = "head_camera"
    width: int = 256
    height: int = 256
    fps: int = 30
    render: bool = True
    viewer: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Render width and height must be positive")
        if self.fps <= 0:
            raise ValueError("FPS must be positive")


def parse_backend_options(values: list[str]) -> dict[str, Any]:
    """Parse repeatable KEY=JSON options without knowing a backend's schema."""

    parsed: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError(f"Backend option must use KEY=VALUE syntax: {value!r}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


class SimulationBackend(abc.ABC):
    """Minimal robot-like interface implemented by MuJoCo, Isaac Sim, or real backends."""

    name: str

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    @classmethod
    @abc.abstractmethod
    def observation_features_for(cls, config: SimulationConfig) -> dict[str, FeatureSpec]:
        """Return LeRobot-compatible observation features without connecting."""

    @classmethod
    @abc.abstractmethod
    def action_features_for(cls, config: SimulationConfig) -> dict[str, type]:
        """Return LeRobot-compatible action features without connecting."""

    @property
    def observation_features(self) -> dict[str, FeatureSpec]:
        return self.observation_features_for(self.config)

    @property
    def action_features(self) -> dict[str, type]:
        return self.action_features_for(self.config)

    @property
    @abc.abstractmethod
    def is_open(self) -> bool:
        """Whether the backend can currently read observations and accept actions."""

    @abc.abstractmethod
    def reset(self, *, seed: int | None = None) -> None:
        """Reset the environment to its home state."""

    @abc.abstractmethod
    def get_observation(self, *, advance: bool = True) -> dict[str, Any]:
        """Return one observation in the backend's common robot schema."""

    @abc.abstractmethod
    def send_action(self, action: dict[str, Any] | Any) -> dict[str, Any]:
        """Apply and return a normalized action dictionary."""

    @abc.abstractmethod
    def step(self) -> None:
        """Advance by one control period."""

    def run_interactive(self, *, debug: bool = False) -> None:
        del debug
        raise NotImplementedError(f"Backend {self.name!r} has no interactive viewer")

    @abc.abstractmethod
    def close(self) -> None:
        """Release simulator and rendering resources."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
