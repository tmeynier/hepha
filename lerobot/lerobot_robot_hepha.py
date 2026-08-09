"""Discovery shim required by LeRobot's third-party robot plugin convention."""

from hepha_lerobot.config_hepha_simulation import HephaSimulationConfig
from hepha_lerobot.hepha_simulation import HephaSimulation

__all__ = ["HephaSimulation", "HephaSimulationConfig"]
