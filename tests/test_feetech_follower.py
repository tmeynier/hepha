from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hardware.axes import AXES
from hardware.calibrate_feetech import (
    ALL_SERVO_IDS,
    RangeTracker,
    build_complete_calibration,
    commission_follower,
)
from hardware.calibration import (
    build_range_calibration,
    joint_to_raw_position,
    load_calibrated_axes,
    raw_to_joint_position,
)
from hardware.feetech_follower import FeetechFollower
from hardware.teleoperate_feetech import (
    JointSafetyLimiter,
    resolve_joint_names,
    start_pose_errors,
    startup_alignment_complete,
)


def _axis_calibration(
    servo_id: int = 2,
    *,
    joint_name: str = "shoulder_l",
) -> dict[str, object]:
    return {
        "servo_id": servo_id,
        "label": "shoulder left",
        "mujoco_actuator": joint_name,
        "q_min": -math.pi / 2,
        "q_home": 0.0,
        "q_max": math.pi / 2,
        "raw_min": 1024,
        "raw_home": 2048,
        "raw_max": 3072,
        "raw_min_delta": -1024,
        "raw_max_delta": 1024,
        "direction": 1,
        "mean_steps_per_radian": 2048 / math.pi,
        "sample_spread": {"min": 0, "home": 0, "max": 0},
        "homing_offset": 123,
        "hardware_min": 1024,
        "hardware_max": 3072,
        "operating_mode": 0,
    }


def _write_follower_calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "follower",
                "port": "/dev/cu.follower",
                "baudrate": 1_000_000,
                "axes": {"2": _axis_calibration()},
            }
        )
    )


def test_joint_to_raw_position_round_trips_sweep_calibration() -> None:
    calibration = build_range_calibration(
        3,
        AXES[3],
        encoder_low=1000,
        encoder_high=3000,
        sample_count=100,
    )

    for q in (-math.pi / 4, -0.2, math.pi / 4, 0.8, 3 * math.pi / 4):
        raw = joint_to_raw_position(q, calibration)
        assert math.isclose(raw_to_joint_position(raw, calibration), q, abs_tol=0.002)


def test_follower_sweep_commissioning_centers_all_servos_and_writes_limits() -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.values = {
                f"servo_{motor_id}": {"Homing_Offset": 0}
                for motor_id in ALL_SERVO_IDS
            }
            self.writes: list[tuple[str, str, int]] = []

        def write(self, name, motor, value, **_kwargs) -> None:
            self.values[motor][name] = value
            self.writes.append((name, motor, value))

        def read(self, name, motor, **_kwargs):
            return self.values[motor][name]

    trackers = {}
    for motor_id in ALL_SERVO_IDS:
        tracker = RangeTracker()
        tracker.observe(1000)
        tracker.observe(3000)
        trackers[motor_id] = tracker
    calibration = build_complete_calibration(
        role="follower",
        port="/dev/cu.follower",
        baudrate=1_000_000,
        trackers=trackers,
        minimum_span=100,
    )
    bus = FakeBus()
    commission_follower(bus, calibration, retries=2)

    assert len(bus.writes) == len(ALL_SERVO_IDS) * 6
    for motor_id in ALL_SERVO_IDS:
        axis = calibration["axes"][str(motor_id)]
        motor = f"servo_{motor_id}"
        assert axis["raw_home"] == 2048
        assert axis["hardware_min"] == 1048
        assert axis["hardware_max"] == 3048
        assert bus.values[motor]["Operating_Mode"] == 0
        assert bus.values[motor]["Homing_Offset"] == -48


def test_follower_requires_follower_role(tmp_path: Path) -> None:
    path = tmp_path / "leader.json"
    data = {
        "schema_version": 1,
        "port": "/dev/cu.leader",
        "baudrate": 1_000_000,
        "axes": {"2": _axis_calibration()},
    }
    path.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="not a follower calibration"):
        FeetechFollower(calibration_path=path)


def test_follower_seeds_before_enabling_and_writes_raw_goals(tmp_path: Path) -> None:
    path = tmp_path / "follower.json"
    _write_follower_calibration(path)
    follower = FeetechFollower(calibration_path=path)

    class FakeBus:
        is_connected = True

        def __init__(self) -> None:
            self.writes: list[tuple[str, dict[str, int]]] = []
            self.enabled = False

        @staticmethod
        def sync_read(*_args, **_kwargs):
            return {"servo_2": 2048}

        def sync_write(self, name, values, **_kwargs) -> None:
            self.writes.append((name, values.copy()))

        def enable_torque(self, *_args, **_kwargs) -> None:
            self.enabled = True

        def disable_torque(self, *_args, **_kwargs) -> None:
            self.enabled = False

        def disconnect(self, **_kwargs) -> None:
            self.is_connected = False

    bus = FakeBus()
    follower.bus = bus
    with pytest.raises(RuntimeError, match="Seed follower goals"):
        follower.enable_torque()

    assert follower.seed_goals_from_present_position() == {"shoulder_l": 0.0}
    follower.enable_torque()
    sent = follower.write_joint_positions({"shoulder_l": math.pi / 2})

    assert bus.enabled
    assert sent == {"shoulder_l": math.pi / 2}
    assert bus.writes == [
        ("Goal_Position", {"servo_2": 2048}),
        ("Goal_Position", {"servo_2": 3072}),
    ]

    follower.disconnect()
    assert not bus.enabled
    assert not bus.is_connected


def test_unarmed_follower_disconnect_does_not_change_torque(tmp_path: Path) -> None:
    path = tmp_path / "follower.json"
    _write_follower_calibration(path)
    follower = FeetechFollower(calibration_path=path)

    class FakeBus:
        is_connected = True

        def __init__(self) -> None:
            self.disable_calls = 0

        def disable_torque(self, *_args, **_kwargs) -> None:
            self.disable_calls += 1

        def disconnect(self, **_kwargs) -> None:
            self.is_connected = False

    bus = FakeBus()
    follower.bus = bus

    follower.disconnect()

    assert bus.disable_calls == 0
    assert not bus.is_connected


def test_joint_safety_limiter_clamps_range_and_velocity() -> None:
    limiter = JointSafetyLimiter(
        limits={"shoulder_l": (-1.0, 1.0)},
        max_velocity=0.5,
        previous={"shoulder_l": 0.0},
    )

    assert limiter.apply({"shoulder_l": 2.0}, 0.1) == {"shoulder_l": 0.05}
    assert limiter.apply({"shoulder_l": -2.0}, 0.1) == {"shoulder_l": 0.0}


def test_startup_alignment_requires_every_joint_within_tolerance() -> None:
    leader = {"shoulder_l": 0.5, "forearm_l": -0.2}

    assert startup_alignment_complete(
        leader,
        {"shoulder_l": 0.46, "forearm_l": -0.23},
        tolerance=0.05,
    )
    assert not startup_alignment_complete(
        leader,
        {"shoulder_l": 0.46, "forearm_l": -0.3},
        tolerance=0.05,
    )


def test_joint_selection_matches_semantic_names_not_ids(tmp_path: Path) -> None:
    leader_path = tmp_path / "leader.json"
    follower_path = tmp_path / "follower.json"
    leader_axis = _axis_calibration(2)
    follower_axis = _axis_calibration(42)
    leader_path.write_text(
        json.dumps({"schema_version": 1, "axes": {"2": leader_axis}})
    )
    follower_path.write_text(
        json.dumps({"schema_version": 1, "axes": {"42": follower_axis}})
    )
    _, leader_axes = load_calibrated_axes(leader_path)
    _, follower_axes = load_calibrated_axes(follower_path)

    assert resolve_joint_names(leader_axes, follower_axes, None) == ("shoulder_l",)
    assert start_pose_errors({"shoulder_l": 0.2}, {"shoulder_l": -0.1}) == {
        "shoulder_l": pytest.approx(0.3)
    }
