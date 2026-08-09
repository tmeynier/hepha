from __future__ import annotations

from pathlib import Path

from lerobot_robot_hepha import HephaSimulation, HephaSimulationConfig


def test_lerobot_plugin_features_are_available_before_connect(tmp_path: Path) -> None:
    config = HephaSimulationConfig(
        id="test",
        calibration_dir=tmp_path,
        backend="mujoco",
        width=160,
        height=120,
    )
    robot = HephaSimulation(config)

    assert robot.name == "hepha_simulation"
    assert robot.observation_features["head_camera"] == (120, 160, 3)
    assert tuple(robot.action_features) == tuple(
        key for key in robot.observation_features if key.endswith(".pos")
    )
    assert robot.is_calibrated
    assert not robot.is_connected
