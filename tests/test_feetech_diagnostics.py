import math
from types import SimpleNamespace
from unittest.mock import patch

from hardware.calibrate_feetech_positions import (
    AXES,
    build_axis_calibration,
    calibration_checks,
    calibration_warnings,
    load_calibration,
    prepare_calibration,
    raw_to_mujoco_position,
    resolve_calibration_bus,
    resolve_endpoint_deltas,
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


def test_explicit_port_and_ids_skip_exhaustive_scan() -> None:
    probe = SimpleNamespace(is_connected=True)
    probe.disconnect = lambda **_kwargs: setattr(probe, "is_connected", False)

    with (
        patch(
            "hardware.calibrate_feetech_positions.create_feetech_bus",
            return_value=probe,
        ) as create_bus,
        patch("hardware.calibrate_feetech_positions.connect_and_ping") as quick_ping,
        patch("hardware.calibrate_feetech_positions.scan_for_bus") as exhaustive_scan,
    ):
        resolved = resolve_calibration_bus(
            object,
            requested_port="/dev/cu.usbmodem101",
            requested_ids=[3, 1, 3],
            baudrate=1_000_000,
            retries=2,
        )

    assert resolved == ("/dev/cu.usbmodem101", 1_000_000, [1, 3])
    create_bus.assert_called_once_with("/dev/cu.usbmodem101", [1, 3])
    quick_ping.assert_called_once_with(
        probe,
        baudrate=1_000_000,
        motor_ids=[1, 3],
        retries=2,
    )
    exhaustive_scan.assert_not_called()
    assert not probe.is_connected


def test_failed_quick_ping_falls_back_to_exhaustive_scan() -> None:
    probe = SimpleNamespace(is_connected=False)
    probe.disconnect = lambda **_kwargs: None
    discovered = ("/dev/cu.usbmodem101", 500_000, [1, 2, 3])

    with (
        patch(
            "hardware.calibrate_feetech_positions.create_feetech_bus",
            return_value=probe,
        ),
        patch(
            "hardware.calibrate_feetech_positions.connect_and_ping",
            side_effect=RuntimeError("ID 1 did not respond"),
        ),
        patch(
            "hardware.calibrate_feetech_positions.scan_for_bus",
            return_value=discovered,
        ) as exhaustive_scan,
    ):
        resolved = resolve_calibration_bus(
            object,
            requested_port="/dev/cu.usbmodem101",
            requested_ids=[1],
            baudrate=1_000_000,
            retries=2,
        )

    assert resolved == discovered
    exhaustive_scan.assert_called_once_with(object, "/dev/cu.usbmodem101")


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


def test_axis_calibration_detects_direction_and_scale() -> None:
    calibration = build_axis_calibration(
        1,
        AXES[1],
        raw_min=1024,
        raw_home=2048,
        raw_max=3072,
        spreads={"min": 1, "home": 0, "max": 1},
    )

    assert calibration["direction"] == 1
    assert calibration["raw_min_delta"] == -1024
    assert calibration["raw_max_delta"] == 1024
    assert math.isclose(calibration["mean_steps_per_radian"], 2048 / math.pi)


def test_axis_calibration_supports_inverted_servo() -> None:
    calibration = build_axis_calibration(
        3,
        AXES[3],
        raw_min=2560,
        raw_home=2048,
        raw_max=512,
        spreads={"min": 0, "home": 0, "max": 0},
    )

    assert calibration["direction"] == -1
    assert not calibration_warnings(
        AXES[3],
        raw_min=2560,
        raw_home=2048,
        raw_max=512,
        spreads={"min": 0, "home": 0, "max": 0},
    )


def test_axis_calibration_jointly_resolves_encoder_wrap() -> None:
    min_delta, max_delta = resolve_endpoint_deltas(
        AXES[5],
        raw_min=1415,
        raw_home=3456,
        raw_max=3010,
    )

    assert min_delta == 2055
    assert max_delta == -446

    calibration = build_axis_calibration(
        5,
        AXES[5],
        raw_min=1415,
        raw_home=3456,
        raw_max=3010,
        spreads={"min": 4, "home": 1, "max": 1},
    )
    assert calibration["direction"] == -1
    assert calibration["raw_min_delta"] == 2055
    assert calibration["raw_max_delta"] == -446
    assert math.isclose(raw_to_mujoco_position(1415, calibration), -3 * math.pi / 4)
    assert raw_to_mujoco_position(3456, calibration) == 0.0
    assert math.isclose(raw_to_mujoco_position(3010, calibration), math.pi / 4)


def test_scale_difference_is_advisory_only() -> None:
    problems, advisories = calibration_checks(
        AXES[5],
        raw_min=1415,
        raw_home=3456,
        raw_max=3010,
        spreads={"min": 4, "home": 1, "max": 1},
    )

    assert problems == []
    assert advisories == ["negative and positive encoder scales differ by more than 15%"]


def test_identical_min_home_and_max_requests_recapture_without_dividing_by_zero() -> None:
    problems, advisories = calibration_checks(
        AXES[9],
        raw_min=2071,
        raw_home=2071,
        raw_max=2071,
        spreads={"min": 0, "home": 0, "max": 0},
    )
    preview = build_axis_calibration(
        9,
        AXES[9],
        raw_min=2071,
        raw_home=2071,
        raw_max=2071,
        spreads={"min": 0, "home": 0, "max": 0},
    )

    assert "an endpoint is too close to home" in problems
    assert "minimum and maximum are not on opposite sides of home" in problems
    assert advisories == []
    assert preview["raw_min_delta"] == 0
    assert preview["raw_max_delta"] == 0
    assert preview["mean_steps_per_radian"] == 0


def test_finger_minimum_and_home_must_match() -> None:
    warnings = calibration_warnings(
        AXES[11],
        raw_min=1900,
        raw_home=2000,
        raw_max=2652,
        spreads={"min": 0, "home": 0, "max": 0},
    )

    assert "minimum and home represent q=0 but their readings differ" in warnings


def test_load_calibration_for_resume(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text('{"schema_version": 1, "axes": {"1": {"raw_home": 2048}}}')

    calibration = load_calibration(path)

    assert calibration["axes"]["1"]["raw_home"] == 2048


def test_separate_axis_calibration_preserves_existing_axes(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"schema_version": 1, "port": "old", "baudrate": 1000000, '
        '"axes": {"2": {"raw_home": 2213}}}'
    )

    calibration, selected_ids, replacing_ids = prepare_calibration(
        path,
        overwrite=False,
        resume=False,
        requested_ids=[4],
        selected_ids=[4],
        port="/dev/cu.usbmodem101",
        baudrate=1_000_000,
    )

    assert calibration["axes"]["2"]["raw_home"] == 2213
    assert selected_ids == [4]
    assert replacing_ids == []


def test_recalibrating_one_axis_replaces_only_that_axis(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"schema_version": 1, "axes": {"2": {"raw_home": 2213}, "4": {"raw_home": 2000}}}'
    )

    calibration, selected_ids, replacing_ids = prepare_calibration(
        path,
        overwrite=False,
        resume=False,
        requested_ids=[4],
        selected_ids=[4],
        port="/dev/cu.usbmodem101",
        baudrate=1_000_000,
    )

    assert set(calibration["axes"]) == {"2", "4"}
    assert selected_ids == [4]
    assert replacing_ids == [4]


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
