import json
import struct
from itertools import product
from unittest.mock import Mock, patch
from unittest.mock import call as mock_call

import pytest

from hardware.cnc_positions import (
    AxisRange,
    LimitCalibration,
    drawer_offsets_mm,
    load_limit_calibration,
    resolve_named_position,
)
from hardware.control_fmc4030 import (
    MOVE_PACKET,
    Axis,
    CommandOutcomeUnknownError,
    Mode,
    all_discrete_position_names,
    build_move_payload,
    build_parser,
    check_connection,
    main,
    send_move,
)
from hardware.fmc4030 import (
    HOME_PACKET,
    STOP_PACKET,
    AxisLimitCalibration,
    AxisPositions,
    CNCStatus,
    FMC4030Client,
    HomeDirection,
    LimitSeekResult,
    StopMode,
    build_home_payload,
    build_stop_payload,
    calibrate_axis_limits_timed,
    expected_motion_seconds,
    load_position_config,
    move_to_absolute_targets_timed,
    new_position_config,
    parse_status_response,
    read_status_until_deadline,
    read_status_with_retries,
    save_limit_calibration,
    save_position_config,
)


def _context_socket(response: bytes = b"\x01\x04\x00") -> Mock:
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    connection.recv.return_value = response
    return connection


def _parameters_response() -> bytes:
    return b"\x01\x12" + bytes(92)


def test_move_payload_has_expected_binary_layout() -> None:
    payload = build_move_payload(
        axis=Axis.X,
        position_mm=-150.0,
        speed_mm_s=100.0,
        acceleration_mm_s2=200.0,
        deceleration_mm_s2=200.0,
        mode=Mode.RELATIVE,
    )

    assert len(payload) == 20 == MOVE_PACKET.size
    assert payload == bytes.fromhex("01 04 00 00 00 16 c3 00 00 c8 42 00 00 48 43 00 00 48 43 01")
    assert struct.unpack("<BBBffffB", payload) == (
        1,
        4,
        0,
        -150.0,
        100.0,
        200.0,
        200.0,
        1,
    )


@pytest.mark.parametrize(
    ("speed", "acceleration", "deceleration"),
    [(0.0, 20.0, 20.0), (10.0, 0.0, 20.0), (10.0, 20.0, -1.0)],
)
def test_move_payload_rejects_nonpositive_motion_rates(
    speed: float,
    acceleration: float,
    deceleration: float,
) -> None:
    with pytest.raises(ValueError):
        build_move_payload(
            Axis.X,
            position_mm=1.0,
            speed_mm_s=speed,
            acceleration_mm_s2=acceleration,
            deceleration_mm_s2=deceleration,
            mode=Mode.RELATIVE,
        )


def test_connection_check_transmits_no_data() -> None:
    connection = _context_socket()
    with patch("hardware.control_fmc4030.socket.create_connection", return_value=connection):
        check_connection("192.168.0.30", 8088, 3.0)

    connection.sendall.assert_not_called()
    connection.recv.assert_not_called()


def test_send_move_transmits_payload_and_returns_response() -> None:
    payload = build_move_payload(Axis.Z, 1.0, 5.0, 10.0, 10.0, Mode.RELATIVE)
    connection = _context_socket(response=payload)

    with patch("hardware.control_fmc4030.socket.create_connection", return_value=connection):
        assert send_move("192.168.0.30", 8088, 3.0, payload) == payload

    connection.settimeout.assert_called_with(3.0)
    connection.sendall.assert_called_once_with(payload)


def test_send_timeout_reports_unknown_machine_state() -> None:
    connection = _context_socket()
    connection.recv.side_effect = TimeoutError
    payload = build_move_payload(Axis.Y, 1.0, 5.0, 10.0, 10.0, Mode.RELATIVE)

    with (
        patch("hardware.control_fmc4030.socket.create_connection", return_value=connection),
        pytest.raises(CommandOutcomeUnknownError, match="state is unknown"),
    ):
        send_move("192.168.0.30", 8088, 3.0, payload)


def test_interrupted_cli_move_does_not_open_a_connection() -> None:
    with (
        patch("builtins.input", side_effect=KeyboardInterrupt),
        patch("hardware.control_fmc4030.socket.create_connection") as create_connection,
    ):
        exit_code = main(
            [
                "move",
                "--axis",
                "x",
                "--position",
                "1",
                "--mode",
                "relative",
            ]
        )

    assert exit_code == 1
    create_connection.assert_not_called()


def test_timed_axis_calibration_uses_manual_workflow_defaults() -> None:
    args = build_parser().parse_args(["calibrate-axis", "--axis", "x"])

    assert args.status_timeout == 3.0
    assert args.max_travel == 600.0
    assert args.speed == 20.0
    assert args.acceleration == 200.0
    assert args.deceleration == 200.0
    assert args.backoff == 5.0
    assert args.settle_seconds == 2.0
    assert args.status_read_timeout == 120.0


def test_named_position_command_defaults() -> None:
    args = build_parser().parse_args(
        ["goto-position", "--position", "b", "--drawer", "9"]
    )

    assert args.position == "B"
    assert args.drawer == 9
    assert args.order == [Axis.Z, Axis.X, Axis.Y]
    assert args.speed == 200.0
    assert args.acceleration == 200.0
    assert args.deceleration == 200.0
    assert args.status_read_timeout == 120.0
    assert not args.read_initial_status
    assert not args.read_final_status


def test_all_axis_calibration_defaults_to_sequential_xyz() -> None:
    args = build_parser().parse_args(["calibrate-all-axes"])

    assert args.order == [Axis.X, Axis.Y, Axis.Z]
    assert args.max_travel == 600.0
    assert args.speed == 20.0
    assert args.acceleration == 200.0
    assert args.deceleration == 200.0
    assert args.backoff == 5.0
    assert args.status_read_timeout == 120.0


def _axis_limit_calibration(axis: Axis) -> AxisLimitCalibration:
    base = float(int(axis) * 100)
    return AxisLimitCalibration(
        axis=axis,
        minimum=LimitSeekResult(axis, "min", base - 55.0, base - 50.0),
        maximum=LimitSeekResult(axis, "max", base + 55.0, base + 50.0),
    )


def test_all_axis_calibration_uses_one_confirmation_and_one_atomic_save() -> None:
    client = Mock(spec=FMC4030Client)
    calibrations = [_axis_limit_calibration(axis) for axis in Axis]

    with (
        patch("builtins.input", return_value="") as confirmation,
        patch("hardware.control_fmc4030.FMC4030Client", return_value=client),
        patch(
            "hardware.control_fmc4030.calibrate_axis_limits_timed",
            side_effect=calibrations,
        ) as calibrate,
        patch("hardware.control_fmc4030.save_limit_calibrations") as save,
    ):
        exit_code = main(["calibrate-all-axes"])

    assert exit_code == 0
    confirmation.assert_called_once()
    assert [call.args[1] for call in calibrate.call_args_list] == list(Axis)
    save.assert_called_once()
    assert save.call_args.args[3] == [
        endpoint
        for calibration in calibrations
        for endpoint in (calibration.minimum, calibration.maximum)
    ]


def test_all_axis_calibration_saves_nothing_if_later_axis_fails() -> None:
    client = Mock(spec=FMC4030Client)

    with (
        patch("builtins.input", return_value=""),
        patch("hardware.control_fmc4030.FMC4030Client", return_value=client),
        patch(
            "hardware.control_fmc4030.calibrate_axis_limits_timed",
            side_effect=(_axis_limit_calibration(Axis.X), RuntimeError("Y failed")),
        ),
        patch("hardware.control_fmc4030.save_limit_calibrations") as save,
    ):
        exit_code = main(["calibrate-all-axes"])

    assert exit_code == 1
    save.assert_not_called()


def _status(
    positions: tuple[float, float, float] = (0.0, 0.0, 0.0),
    speeds: tuple[float, float, float] = (0.0, 0.0, 0.0),
    negative_limits: int = 0,
    positive_limits: int = 0,
    home_status: int = 8,
) -> CNCStatus:
    return CNCStatus(
        positions_mm=positions,
        speeds_mm_s=speeds,
        input_mask=1,
        output_mask=2,
        negative_limit_mask=negative_limits,
        positive_limit_mask=positive_limits,
        run_status=3,
        axis_statuses=(4, 5, 6),
        home_status=home_status,
    )


def test_home_payload_has_expected_binary_layout() -> None:
    payload = build_home_payload(
        Axis.Z,
        HomeDirection.NEGATIVE,
        speed_mm_s=50.0,
        acceleration_mm_s2=200.0,
        backoff_mm=5.0,
    )

    assert len(payload) == HOME_PACKET.size == 16
    assert struct.unpack("<BBBfffB", payload) == (1, 6, 2, 50.0, 200.0, 5.0, 2)


def test_home_transmits_without_waiting_for_an_acknowledgement() -> None:
    connection = _context_socket()
    client = FMC4030Client("192.168.0.30", 8088, timeout=3.0)

    with patch("hardware.fmc4030.socket.create_connection", return_value=connection):
        response = client.home(
            Axis.Y,
            HomeDirection.NEGATIVE,
            speed_mm_s=5.0,
            acceleration_mm_s2=20.0,
            backoff_mm=5.0,
        )

    assert response is None
    connection.settimeout.assert_called_once_with(3.0)
    connection.sendall.assert_called_once()
    connection.recv.assert_not_called()


def test_immediate_stop_payload_matches_documented_protocol() -> None:
    payload = build_stop_payload(Axis.Y, StopMode.IMMEDIATE)

    assert len(payload) == STOP_PACKET.size == 4
    assert payload == bytes.fromhex("01 07 01 02")


def test_status_response_parses_all_three_axes_and_masks() -> None:
    body = struct.pack(
        "<ffffff9I",
        1.25,
        2.5,
        -3.75,
        0.1,
        0.2,
        0.3,
        5,
        6,
        0b101,
        0b010,
        7,
        8,
        9,
        10,
        0x08,
    )
    status = parse_status_response(b"\x01\x03\x00\x00" + body)

    assert status.positions_mm == pytest.approx((1.25, 2.5, -3.75))
    assert status.speeds_mm_s == pytest.approx((0.1, 0.2, 0.3))
    assert status.negative_limit(Axis.X)
    assert not status.negative_limit(Axis.Y)
    assert status.positive_limit(Axis.Y)
    assert status.axis_statuses == (8, 9, 10)
    assert status.home_status == 0x08


def test_status_response_rejects_truncation() -> None:
    with pytest.raises(ValueError, match="expected at least 64"):
        parse_status_response(bytes(63))


def test_status_query_reads_documented_big_endian_length_frame() -> None:
    body = struct.pack(
        "<ffffff9I",
        1.0,
        2.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0b010,
        1,
        2,
        3,
        4,
        8,
    )
    payload = body + bytes(600)
    frame = b"\x01\x03" + len(payload).to_bytes(2, "big") + payload
    connection = _context_socket()
    connection.recv.side_effect = (_parameters_response(), frame[:11], frame[11:])
    client = FMC4030Client("192.168.0.30", 8088, timeout=3.0)

    with patch("hardware.fmc4030.socket.create_connection", return_value=connection):
        status = client.read_status()

    assert status.positions_mm == pytest.approx((1.0, 2.0, 3.0))
    assert status.positive_limit(Axis.Y)
    assert connection.sendall.call_args_list == [
        mock_call(b"\x01\x12"),
        mock_call(b"\x01\x03"),
    ]


def test_status_response_timeout_is_independent_from_general_tcp_timeout() -> None:
    connection = _context_socket()
    connection.recv.side_effect = TimeoutError("controller did not answer")
    client = FMC4030Client(
        "192.168.0.30",
        8088,
        timeout=10.0,
        status_timeout=0.5,
    )

    with (
        patch("hardware.fmc4030.socket.create_connection", return_value=connection),
        pytest.raises(TimeoutError, match="after 1 attempt"),
    ):
        client.read_status(attempts=1)

    assert mock_call(0.5) in connection.settimeout.call_args_list


def test_delayed_status_retries_receive_on_one_connection_without_resending() -> None:
    body = struct.pack("<ffffff9I", *(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, *([0] * 9)))
    frame = b"\x01\x03\x02\x94" + body + bytes(600)
    connection = _context_socket()
    connection.recv.side_effect = (
        _parameters_response(),
        TimeoutError("not ready yet"),
        TimeoutError("still preparing"),
        frame,
    )
    client = FMC4030Client(
        "192.168.0.30",
        8088,
        timeout=10.0,
        status_timeout=0.5,
    )

    with (
        patch(
            "hardware.fmc4030.socket.create_connection", return_value=connection
        ) as create_connection,
        patch("hardware.fmc4030.time.sleep"),
    ):
        status = client.read_status(attempts=3)

    assert status.positions_mm == pytest.approx((1.0, 2.0, 3.0))
    assert create_connection.call_count == 1
    assert connection.sendall.call_args_list == [
        mock_call(b"\x01\x12"),
        mock_call(b"\x01\x03"),
    ]


def test_retry_helper_delegates_one_retry_policy_to_client() -> None:
    client = Mock(spec=FMC4030Client)
    expected = _status(positions=(1.0, 2.0, 3.0))
    client.read_status.return_value = expected

    status = read_status_with_retries(client)

    assert status is expected
    client.read_status.assert_called_once_with(attempts=5, retry_delay=0.25)


def test_status_retry_count_is_five_not_nested_to_twenty_five() -> None:
    client = FMC4030Client("192.168.0.30", 8088, timeout=3.0)

    with (
        patch(
            "hardware.fmc4030.socket.create_connection",
            side_effect=TimeoutError("busy"),
        ) as create_connection,
        patch("hardware.fmc4030.time.sleep"),
        pytest.raises(TimeoutError, match="after 5 attempts"),
    ):
        read_status_with_retries(client)

    assert create_connection.call_count == 5


def test_consecutive_status_reads_reuse_one_tcp_connection() -> None:
    body = struct.pack("<ffffff9I", *(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, *([0] * 9)))
    frame = b"\x01\x03\x02\x94" + body + bytes(600)
    connection = _context_socket()
    connection.recv.side_effect = (_parameters_response(), frame, frame)
    client = FMC4030Client("192.168.0.30", 8088, timeout=3.0)

    with patch(
        "hardware.fmc4030.socket.create_connection", return_value=connection
    ) as create_connection:
        first = client.read_status(attempts=1)
        second = client.read_status(attempts=1)

    assert first.positions_mm == pytest.approx((1.0, 2.0, 3.0))
    assert second.positions_mm == pytest.approx((1.0, 2.0, 3.0))
    assert create_connection.call_count == 1
    assert connection.sendall.call_args_list == [
        mock_call(b"\x01\x12"),
        mock_call(b"\x01\x03"),
        mock_call(b"\x01\x03"),
    ]


def test_status_does_not_wait_for_unused_filename_trailer() -> None:
    body = struct.pack("<ffffff9I", *(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, *([0] * 9)))
    required_status = b"\x01\x03\x02\x94" + body
    connection = _context_socket()
    connection.recv.side_effect = (
        _parameters_response(),
        required_status,
        TimeoutError("trailer delayed"),
    )
    client = FMC4030Client("192.168.0.30", 8088, timeout=3.0)

    with patch("hardware.fmc4030.socket.create_connection", return_value=connection):
        status = client.read_status(attempts=1)

    assert status.positions_mm == pytest.approx((1.0, 2.0, 3.0))
    connection.close.assert_called_once()


def test_immediate_stop_retries_connection_five_times() -> None:
    payload = bytes.fromhex("01 07 02 02")
    connection = _context_socket(response=payload)
    failures = [TimeoutError("busy")] * 4
    client = FMC4030Client("192.168.0.30", 8088, timeout=3.0)

    with (
        patch(
            "hardware.fmc4030.socket.create_connection",
            side_effect=(*failures, connection),
        ) as create_connection,
        patch("hardware.fmc4030.time.sleep"),
    ):
        client.stop(Axis.Z, StopMode.IMMEDIATE)

    assert create_connection.call_count == 5
    connection.sendall.assert_called_once_with(payload)
    connection.recv.assert_called_once()


def test_status_poll_can_follow_stop_without_an_acknowledgement() -> None:
    body = struct.pack("<ffffff9I", *(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, *([0] * 9)))
    frame = b"\x01\x03\x02\x94" + body + bytes(600)
    connection = _context_socket()
    connection.recv.side_effect = (
        TimeoutError("no stop echo"),
        _parameters_response(),
        frame,
    )
    client = FMC4030Client("192.168.0.30", 8088, timeout=3.0)

    with (
        patch(
            "hardware.fmc4030.socket.create_connection", return_value=connection
        ) as create_connection,
        patch("hardware.fmc4030.time.sleep"),
    ):
        client.stop(Axis.X, StopMode.IMMEDIATE)
        status = client.read_status(attempts=1)

    assert status.positions_mm == pytest.approx((1.0, 2.0, 3.0))
    assert create_connection.call_count == 1
    assert connection.sendall.call_args_list == [
        mock_call(bytes.fromhex("01 07 00 02")),
        mock_call(b"\x01\x12"),
        mock_call(b"\x01\x03"),
    ]


def test_position_config_round_trip(tmp_path) -> None:
    axes = {
        "x": AxisPositions(0.0, 50.0, 100.0, 105.0),
        "y": AxisPositions(1.0, 31.0, 61.0, 66.0),
        "z": AxisPositions(-100.0, -50.0, 0.0, 5.0),
    }
    config = new_position_config("192.168.0.30", 8088, axes)
    path = tmp_path / "positions.json"

    save_position_config(path, config)

    assert load_position_config(path) == config


def test_position_config_rejects_bad_ordering() -> None:
    axes = {
        "x": AxisPositions(0.0, 50.0, 100.0, 105.0),
        "y": AxisPositions(0.0, 0.0, 100.0, 105.0),
        "z": AxisPositions(0.0, 50.0, 100.0, 105.0),
    }
    with pytest.raises(ValueError, match="min < mid < max"):
        new_position_config("192.168.0.30", 8088, axes)


def _named_position_calibration() -> LimitCalibration:
    return LimitCalibration(
        ip="192.168.0.30",
        port=8088,
        axes={
            Axis.X: AxisRange(-100.0, 0.0, 100.0),
            Axis.Y: AxisRange(-100.0, 0.0, 100.0),
            Axis.Z: AxisRange(-100.0, 0.0, 100.0),
        },
    )


@pytest.mark.parametrize(
    ("drawer", "expected"),
    [
        (1, (-50.0, 50.0)),
        (2, (0.0, 50.0)),
        (3, (50.0, 50.0)),
        (4, (-50.0, 0.0)),
        (5, (0.0, 0.0)),
        (6, (50.0, 0.0)),
        (7, (-50.0, -50.0)),
        (8, (0.0, -50.0)),
        (9, (50.0, -50.0)),
    ],
)
def test_drawer_offsets_convert_centimetres_to_millimetres(
    drawer: int,
    expected: tuple[float, float],
) -> None:
    assert drawer_offsets_mm(drawer) == expected


def test_named_positions_resolve_from_safe_calibrated_limits() -> None:
    calibration = _named_position_calibration()

    assert resolve_named_position(calibration, "A") == {
        Axis.X: 0.0,
        Axis.Y: -100.0,
        Axis.Z: 100.0,
    }
    assert resolve_named_position(calibration, "B", drawer=9) == {
        Axis.X: -50.0,
        Axis.Y: -100.0,
        Axis.Z: -100.0,
    }


@pytest.mark.parametrize(
    ("drawer", "expected_x", "expected_y"),
    [
        (1, 50.0, 0.0),
        (2, 0.0, 0.0),
        (3, -50.0, 0.0),
        (4, 50.0, -50.0),
        (5, 0.0, -50.0),
        (6, -50.0, -50.0),
        (7, 50.0, -100.0),
        (8, 0.0, -100.0),
        (9, -50.0, -100.0),
    ],
)
def test_new_b_inverts_drawer_x_and_applies_drawer_y_row(
    drawer: int,
    expected_x: float,
    expected_y: float,
) -> None:
    targets = resolve_named_position(_named_position_calibration(), "B", drawer)

    assert targets[Axis.X] == expected_x
    assert targets[Axis.Y] == expected_y
    assert targets[Axis.Z] == -100.0


def test_named_position_starts_without_confirmation() -> None:
    client = Mock(spec=FMC4030Client)
    calibration = _named_position_calibration()

    with (
        patch("builtins.input") as confirmation,
        patch("hardware.control_fmc4030.FMC4030Client", return_value=client),
        patch("hardware.control_fmc4030.load_limit_calibration", return_value=calibration),
        patch("hardware.control_fmc4030.move_to_absolute_targets_timed", return_value=None),
    ):
        exit_code = main(["goto-position", "--position", "A"])

    assert exit_code == 0
    confirmation.assert_not_called()


def test_limit_calibration_loader_uses_safe_endpoints_and_computes_midpoint(
    tmp_path,
) -> None:
    path = tmp_path / "limits.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller": {"ip": "192.168.0.30", "port": 8088},
                "units": "mm",
                "axes": {
                    name: {
                        "min": {"position_mm": minimum},
                        "max": {"position_mm": maximum},
                    }
                    for name, minimum, maximum in (
                        ("x", -250.0, 230.0),
                        ("y", 5.0, 290.0),
                        ("z", -175.0, 231.0),
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = load_limit_calibration(path)

    assert calibration.axes[Axis.X] == AxisRange(-250.0, -10.0, 230.0)
    assert calibration.axes[Axis.Y] == AxisRange(5.0, 147.5, 290.0)


def test_named_position_requires_drawer_only_for_b() -> None:
    calibration = _named_position_calibration()

    with pytest.raises(ValueError, match="requires --drawer"):
        resolve_named_position(calibration, "B")
    with pytest.raises(ValueError, match="only for position B"):
        resolve_named_position(calibration, "A", drawer=1)


def test_named_position_rejects_target_outside_safe_interval() -> None:
    calibration = LimitCalibration(
        ip="192.168.0.30",
        port=8088,
        axes={
            Axis.X: AxisRange(-40.0, 0.0, 40.0),
            Axis.Y: AxisRange(-100.0, 0.0, 100.0),
            Axis.Z: AxisRange(-100.0, 0.0, 100.0),
        },
    )

    with pytest.raises(ValueError, match="outside its safe calibrated interval"):
        resolve_named_position(calibration, "B", drawer=9)


def test_timed_absolute_targets_move_in_parallel_and_verify_once() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0
    initial = _status(positions=(0.0, 0.0, 0.0))
    final = _status(positions=(10.0, 20.0, 30.0))
    client.read_status.side_effect = (initial, final)
    progress: list[tuple[str, Axis | None, float]] = []

    with patch("hardware.fmc4030.time.sleep") as sleep:
        result = move_to_absolute_targets_timed(
            client,
            {Axis.X: 10.0, Axis.Y: 20.0, Axis.Z: 30.0},
            [Axis.Z, Axis.X, Axis.Y],
            speed_mm_s=20.0,
            acceleration_mm_s2=200.0,
            deceleration_mm_s2=200.0,
            progress=lambda phase, axis, seconds: progress.append(
                (phase, axis, seconds)
            ),
        )

    assert result is final
    assert [(call.args[0], call.args[1], call.args[5]) for call in client.move.call_args_list] == [
        (Axis.Z, 30.0, Mode.ABSOLUTE),
        (Axis.X, 10.0, Mode.ABSOLUTE),
        (Axis.Y, 20.0, Mode.ABSOLUTE),
    ]
    assert client.read_status.call_count == 2
    sleep.assert_called_once_with(pytest.approx(3.6))
    assert [phase for phase, _, _ in progress] == [
        "read_initial",
        "command",
        "command",
        "command",
        "wait_parallel",
        "read_final",
    ]


def test_timed_absolute_targets_request_stops_if_final_verification_fails() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0
    client.read_status.side_effect = (
        _status(positions=(0.0, 0.0, 0.0)),
        _status(positions=(0.0, 20.0, 30.0)),
    )

    with (
        patch("hardware.fmc4030.time.sleep"),
        pytest.raises(RuntimeError, match="did not reach"),
    ):
        move_to_absolute_targets_timed(
            client,
            {Axis.X: 10.0, Axis.Y: 20.0, Axis.Z: 30.0},
            [Axis.Z, Axis.X, Axis.Y],
            20.0,
            200.0,
            200.0,
        )

    assert [call.args for call in client.stop.call_args_list] == [
        (Axis.Z, StopMode.IMMEDIATE),
        (Axis.X, StopMode.IMMEDIATE),
        (Axis.Y, StopMode.IMMEDIATE),
    ]


def test_timed_absolute_targets_can_skip_initial_status_with_safe_span_bounds() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0
    final = _status(positions=(10.0, 20.0, 30.0))
    client.read_status.return_value = final
    progress: list[tuple[str, Axis | None, float]] = []

    with patch("hardware.fmc4030.time.sleep") as sleep:
        result = move_to_absolute_targets_timed(
            client,
            {Axis.X: 10.0, Axis.Y: 20.0, Axis.Z: 30.0},
            [Axis.Z, Axis.X, Axis.Y],
            speed_mm_s=200.0,
            acceleration_mm_s2=200.0,
            deceleration_mm_s2=200.0,
            read_initial_status=False,
            read_final_status=True,
            travel_distance_bounds_mm={
                Axis.X: 480.0,
                Axis.Y: 285.0,
                Axis.Z: 406.0,
            },
            progress=lambda phase, axis, seconds: progress.append(
                (phase, axis, seconds)
            ),
        )

    assert result is final
    assert client.read_status.call_count == 1
    assert client.move.call_count == 3
    sleep.assert_called_once_with(pytest.approx(5.4))
    assert [phase for phase, _, _ in progress] == [
        "command",
        "command",
        "command",
        "wait_parallel",
        "read_final",
    ]


def test_timed_absolute_targets_can_skip_both_status_reads() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0

    with patch("hardware.fmc4030.time.sleep") as sleep:
        result = move_to_absolute_targets_timed(
            client,
            {Axis.X: 10.0, Axis.Y: 20.0, Axis.Z: 30.0},
            [Axis.Z, Axis.X, Axis.Y],
            speed_mm_s=200.0,
            acceleration_mm_s2=200.0,
            deceleration_mm_s2=200.0,
            read_initial_status=False,
            read_final_status=False,
            travel_distance_bounds_mm={
                Axis.X: 480.0,
                Axis.Y: 285.0,
                Axis.Z: 406.0,
            },
        )

    assert result is None
    client.read_status.assert_not_called()
    assert client.move.call_count == 3
    sleep.assert_not_called()


def test_expected_motion_time_matches_bounded_trapezoidal_move() -> None:
    duration = expected_motion_seconds(
        distance_mm=600.0,
        speed_mm_s=20.0,
        acceleration_mm_s2=200.0,
        deceleration_mm_s2=200.0,
    )

    assert duration == pytest.approx(30.1)


def test_timed_axis_calibration_matches_successful_manual_sequence() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0
    minimum = _status(positions=(-260.9, 0.0, 0.0), negative_limits=0b001)
    maximum = _status(positions=(238.3, 0.0, 0.0), positive_limits=0b001)
    client.read_status.side_effect = (minimum, maximum)
    progress: list[tuple[str, float]] = []

    with patch("hardware.fmc4030.time.sleep") as sleep:
        result = calibrate_axis_limits_timed(
            client,
            Axis.X,
            max_travel_mm=600.0,
            speed_mm_s=20.0,
            acceleration_mm_s2=200.0,
            deceleration_mm_s2=200.0,
            backoff_mm=5.0,
            settle_seconds=2.0,
            progress=lambda phase, seconds: progress.append((phase, seconds)),
        )

    assert result.minimum.axis is Axis.X
    assert result.minimum.limit == "min"
    assert result.minimum.switch_position_mm == pytest.approx(-260.9)
    assert result.minimum.safe_position_mm == pytest.approx(-255.9)
    assert result.maximum.axis is Axis.X
    assert result.maximum.limit == "max"
    assert result.maximum.switch_position_mm == pytest.approx(238.3)
    assert result.maximum.safe_position_mm == pytest.approx(233.3)
    assert [call.args[1] for call in client.move.call_args_list] == [
        -600.0,
        5.0,
        600.0,
        -5.0,
    ]
    assert client.read_status.call_count == 2
    assert [phase for phase, _ in progress] == [
        "move_min",
        "read_min",
        "backoff_min",
        "move_max",
        "read_max",
        "backoff_max",
    ]
    assert sleep.call_count == 4


def test_timed_axis_calibration_fails_and_stops_when_limit_is_not_active() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0
    client.read_status.return_value = _status(positions=(-600.0, 0.0, 0.0))

    with (
        patch("hardware.fmc4030.time.sleep"),
        pytest.raises(RuntimeError, match="min limit input was not triggered"),
    ):
        calibrate_axis_limits_timed(
            client,
            Axis.X,
            600.0,
            20.0,
            200.0,
            200.0,
            5.0,
        )

    client.stop.assert_called_once_with(Axis.X, StopMode.IMMEDIATE)


def test_timed_axis_calibration_rejects_status_that_is_still_moving() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0
    client.read_status.return_value = _status(
        positions=(-260.9, 0.0, 0.0),
        speeds=(0.5, 0.0, 0.0),
        negative_limits=0b001,
    )

    with (
        patch("hardware.fmc4030.time.sleep"),
        pytest.raises(RuntimeError, match="still moving"),
    ):
        calibrate_axis_limits_timed(
            client,
            Axis.X,
            600.0,
            20.0,
            200.0,
            200.0,
            5.0,
        )

    client.stop.assert_called_once_with(Axis.X, StopMode.IMMEDIATE)


def test_long_status_deadline_continues_beyond_five_receive_windows() -> None:
    client = Mock(spec=FMC4030Client)
    client.status_timeout = 3.0
    expected = _status(positions=(-260.9, 0.0, 0.0), negative_limits=0b001)
    client.read_status.side_effect = [TimeoutError("slow controller")] * 6 + [expected]

    with patch("hardware.fmc4030.time.sleep"):
        result = read_status_until_deadline(client, timeout=120.0)

    assert result is expected
    assert client.read_status.call_count == 7
    assert client.status_timeout == 3.0


def test_limit_calibration_preserves_endpoints_and_computes_midpoint(tmp_path) -> None:
    path = tmp_path / "limits.json"
    save_limit_calibration(
        path,
        "192.168.0.30",
        8088,
        LimitSeekResult(Axis.Y, "min", -2.0, 3.0),
    )
    payload = save_limit_calibration(
        path,
        "192.168.0.30",
        8088,
        LimitSeekResult(Axis.Y, "max", 102.0, 97.0),
    )

    y_axis = payload["axes"]["y"]
    assert y_axis["min"]["position_mm"] == 3.0
    assert y_axis["max"]["position_mm"] == 97.0
    assert y_axis["mid_position_mm"] == 50.0


def test_discrete_position_grid_contains_exactly_27_unique_combinations() -> None:
    combinations = all_discrete_position_names()

    assert len(combinations) == len(set(combinations)) == 27
    assert set(combinations) == set(product(("min", "mid", "max"), repeat=3))
