"""LeRobot Robot adapter backed by a registered Hepha simulator."""

from __future__ import annotations

from typing import Any

from lerobot.robots import Robot

from simulation import (
    SimulationBackend,
    SimulationConfig,
    create_backend,
    get_backend_class,
)

from .config_hepha_simulation import HephaSimulationConfig


class HephaSimulation(Robot):
    config_class = HephaSimulationConfig
    name = "hepha_simulation"

    def __init__(self, config: HephaSimulationConfig):
        super().__init__(config)
        self.config = config
        self.backend: SimulationBackend | None = None
        self.cameras = {config.camera: None}

    def _simulation_config(self) -> SimulationConfig:
        return SimulationConfig(
            camera=self.config.camera,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            render=True,
            viewer=self.config.viewer,
            options=self.config.backend_options,
        )

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        backend_class = get_backend_class(self.config.backend)
        return backend_class.observation_features_for(self._simulation_config())

    @property
    def action_features(self) -> dict[str, type]:
        backend_class = get_backend_class(self.config.backend)
        return backend_class.action_features_for(self._simulation_config())

    @property
    def is_connected(self) -> bool:
        return self.backend is not None and self.backend.is_open

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise ConnectionError("Hepha simulation is already connected")
        self.backend = create_backend(self.config.backend, self._simulation_config())

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def reset(self, *, seed: int | None = None) -> None:
        self._require_backend().reset(seed=seed)

    def get_observation(self) -> dict[str, Any]:
        return self._require_backend().get_observation()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._require_backend().send_action(action)

    def disconnect(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def _require_backend(self) -> SimulationBackend:
        if not self.is_connected or self.backend is None:
            raise ConnectionError("Hepha simulation is not connected")
        return self.backend
