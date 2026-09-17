import math
from types import SimpleNamespace
from unittest.mock import patch

from hardware.axes import AXES
from hardware.calibrate_feetech import (
    ALL_SERVO_IDS,
    RangeTracker,
    build_complete_calibration,
    resolve_complete_bus,
)
from hardware.calibration import (
    build_range_calibration,
    load_calibration,
    raw_to_mujoco_position,
    wrapped_encoder_delta,
)
from hardware.read_feetech_positions import format_live_table, scan_for_bus
from hardware.scan_feetech_ids import _prefer_macos_callout_devices, find_candidate_ports
from hardware.teleoperate_mujoco_joint import configure_cnc_start_pose, selected_servo_ids


def test_macos_prefers_callout_device_for_same_adapter() -> None:
    devices = ["/dev/tty.usbmodem101", "/dev/cu.usbmodem101", "/dev/cu.usbserial-1"]

    with patch("hardware.scan_feetech_ids.platform.system", return_value="Darwin"):
        assert _prefer_macos_callout_devices(devices) == [
            "/dev/cu.usbmodem101",
            "/dev/cu.usbserial-1",
        ]


def test_candidate_ports_only_include_usb_serial_devices() -> None:
    ports = [
        SimpleNamespace(
            device="/dev/cu.Bluetooth-Incoming-Port", vid=None, description=None, hwid=None
        ),
        SimpleNamespace(
            device="/dev/cu.usbmodem101", vid=0x1A86, description="USB", hwid="USB VID:PID"
        ),
    ]

    with (
        patch("serial.tools.list_ports.comports", return_value=ports),
        patch("hardware.scan_feetech_ids.platform.system", return_value="Darwin"),
    ):
        assert find_candidate_ports() == ["/dev/cu.usbmodem101"]


def test_scan_for_bus_returns_the_only_responding_bus() -> None:
    class FakeBus:
        @staticmethod
        def scan_port(port: str) -> dict[int, list[int]]:
            return {1_000_000: [3, 1, 2]} if port == "/dev/cu.usbmodem101" else {}

    with patch(
        "hardware.read_feetech_positions.find_candidate_ports",
        return_value=["/dev/cu.usbmodem101", "/dev/cu.usbserial-other"],
    ):
        assert scan_for_bus(FakeBus, None) == (
            "/dev/cu.usbmodem101",
            1_000_000,
            [1, 2, 3],
        )


def test_explicit_port_skips_exhaustive_scan() -> None:
    with patch("hardware.calibrate_feetech.scan_for_bus") as exhaustive_scan:
        resolved = resolve_complete_bus(
            object,
            port="/dev/cu.usbmodem101",
            baudrate=1_000_000,
            retries=2,
        )

    assert resolved == ("/dev/cu.usbmodem101", 1_000_000)
    exhaustive_scan.assert_not_called()


def test_live_table_contains_robot_labels_and_positions() -> None:
    table = format_live_table(
        port="/dev/cu.usbmodem101",
        baudrate=1_000_000,
        elapsed=84.6,
        ids=[1, 7, 8, 9, 10, 11, 12, 42],
        raw_positions={
            1: 831,
            7: 355,
            8: 3993,
            9: 1163,
            10: 2065,
            11: 1778,
            12: 3581,
            42: 2048,
        },
    )

    assert "shoulder right" in table
    assert "wrist right" in table
    assert "wrist left" in table
    assert "finger right" in table
    assert "finger left" in table
    assert "unmapped servo" in table
    assert "73.04°" in table
    assert "180.00°" in table


def test_wrist_and_hand_servo_mapping() -> None:
    assert AXES[7].label == "wrist right"
    assert AXES[7].mujoco_actuator == "wrist_r"
    assert AXES[8].label == "wrist left"
    assert AXES[8].mujoco_actuator == "wrist_l"
    assert AXES[9].label == "hand right"
    assert AXES[9].mujoco_actuator == "hand_r"
    assert AXES[10].label == "hand left"
    assert AXES[10].mujoco_actuator == "hand_l"


def test_arm_servo_labels_match_mujoco_actuators() -> None:
    assert AXES[5].label == "arm right"
    assert AXES[5].mujoco_actuator == "arm_r"
    assert AXES[6].label == "arm left"
    assert AXES[6].mujoco_actuator == "arm_l"


def test_wrapped_encoder_delta_crosses_zero() -> None:
    assert wrapped_encoder_delta(100, 4000) == 196
    assert wrapped_encoder_delta(4000, 100) == -196


def test_range_tracker_unwraps_crossing_zero_and_keeps_extrema() -> None:
    tracker = RangeTracker()
    for raw in (3900, 4050, 50, 300, 100, 4000, 3800):
        tracker.observe(raw)

    assert tracker.samples == 7
    assert tracker.minimum == 3800
    assert tracker.maximum == 4396
    assert tracker.span == 596


def test_range_calibration_uses_midpoint_and_static_direction() -> None:
    calibration = build_range_calibration(
        3,
        AXES[3],
        encoder_low=1000,
        encoder_high=3000,
        sample_count=120,
    )

    assert calibration["direction"] == -1
    assert calibration["raw_min"] == 3000
    assert calibration["raw_home"] == 2000
    assert calibration["raw_max"] == 1000
    assert calibration["raw_min_delta"] == 1000
    assert calibration["raw_max_delta"] == -1000
    assert math.isclose(calibration["q_home"], math.pi / 4)
    assert math.isclose(calibration["mean_steps_per_radian"], 2000 / math.pi)


def test_complete_sweep_replaces_all_twelve_axes() -> None:
    trackers = {}
    for motor_id in ALL_SERVO_IDS:
        tracker = RangeTracker()
        tracker.observe(1000)
        tracker.observe(3000)
        trackers[motor_id] = tracker

    calibration = build_complete_calibration(
        role="leader",
        port="/dev/cu.usbmodem101",
        baudrate=1_000_000,
        trackers=trackers,
        minimum_span=100,
    )

    assert calibration["role"] == "leader"
    assert calibration["calibration_method"] == "simultaneous_range_sweep"
    assert set(calibration["axes"]) == {str(motor_id) for motor_id in ALL_SERVO_IDS}


def test_load_range_calibration(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text('{"schema_version": 1, "axes": {"1": {"raw_home": 2048}}}')

    calibration = load_calibration(path)

    assert calibration["axes"]["1"]["raw_home"] == 2048


def test_raw_to_mujoco_position_uses_all_three_calibration_points() -> None:
    calibration = {
        "raw_min_delta": -968,
        "raw_max_delta": 1026,
        "raw_home": 2213,
        "q_min": -math.pi / 2,
        "q_home": 0.0,
        "q_max": math.pi / 2,
    }

    assert math.isclose(raw_to_mujoco_position(1245, calibration), -math.pi / 2)
    assert raw_to_mujoco_position(2213, calibration) == 0.0
    assert math.isclose(raw_to_mujoco_position(3239, calibration), math.pi / 2)
    assert math.isclose(raw_to_mujoco_position(1729, calibration), -math.pi / 4)


def test_raw_to_mujoco_position_clamps_outside_calibrated_range() -> None:
    calibration = {
        "raw_min_delta": -500,
        "raw_max_delta": 500,
        "raw_home": 4000,
        "q_min": -1.0,
        "q_home": 0.0,
        "q_max": 1.0,
    }

    assert raw_to_mujoco_position(3000, calibration) == -1.0
    assert raw_to_mujoco_position(904, calibration) == 1.0


def test_teleoperation_resolves_single_and_multiple_ids() -> None:
    assert selected_servo_ids(None, None) == [2]
    assert selected_servo_ids(4, None) == [4]
    assert selected_servo_ids(None, [4, 2, 4]) == [2, 4]


def test_teleoperation_starts_cnc_at_requested_pose() -> None:
    actuator_names = ("cnc_x", "cnc_y", "head_z", "shoulder_l")
    action = [0.065, 0.0, 0.02, 0.5]
    control_low = [-0.1, -0.2, -0.1, -math.pi / 2]

    configured = configure_cnc_start_pose(action, control_low, actuator_names)

    assert configured == [0.0, 0.0, -0.1, 0.5]
