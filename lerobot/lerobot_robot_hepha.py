"""Discovery shim required by LeRobot's third-party robot plugin convention."""

from hepha_lerobot.config_hepha_simulation import HephaSimulationConfig
from hepha_lerobot.hepha_simulation import HephaSimulation
from hepha_lerobot.policies.hepha_act_phase.configuration_hepha_act_phase import (
    HephaActPhaseConfig,
)
from hepha_lerobot.policies.hepha_act_awr.configuration_hepha_act_awr import (
    HephaActAWRConfig,
)

__all__ = [
    "HephaActAWRConfig",
    "HephaActPhaseConfig",
    "HephaSimulation",
    "HephaSimulationConfig",
]
