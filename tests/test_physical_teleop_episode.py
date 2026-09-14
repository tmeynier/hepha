from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hardware.fmc4030 import Axis, FMC4030Client, StopMode
from hardware.teleop_episode import (
    CNCTransition,
    EpisodeFrame,
    EpisodePhase,
    NamedCNCController,
    NullEpisodeSink,
    TeleopEpisodeSequence,
    choose_drawer,
)
from hardware.teleoperate_episode import parse_args


def _write_cnc_calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller": {"ip": "192.168.0.30", "port": 8088},
                "units": "mm",
                "axes": {
                    "x": {
                        "min": {"position_mm": -100.0},
                        "max": {"position_mm": 100.0},
                    },
                    "y": {
                        "min": {"position_mm": -100.0},
                        "max": {"position_mm": 100.0},
                    },
                    "z": {
                        "min": {"position_mm": -100.0},
                        "max": {"position_mm": 100.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_episode_sequence_is_a_b_a_b_a_complete() -> None:
    sequence = TeleopEpisodeSequence(drawer=7)

    assert sequence.initial_transition == CNCTransition(EpisodePhase.START_A, "A", None)
    assert sequence.next_instruction == "Press SPACE to move to B and open drawer 7."

    drawer_move = sequence.advance()
    assert drawer_move == CNCTransition(EpisodePhase.OPEN_DRAWER_B, "B", 7)
    assert sequence.next_instruction == "Press SPACE to return to A and pick the cube."

    return_move = sequence.advance()
    assert return_move == CNCTransition(EpisodePhase.PICK_CUBE_A, "A", None)
    assert sequence.next_instruction.startswith("Press SPACE to move to B")

    placement_move = sequence.advance()
    assert placement_move == CNCTransition(EpisodePhase.PLACE_CLOSE_B, "B", 7)
    assert sequence.next_instruction == "Press SPACE to return to A."

    final_move = sequence.advance()
    assert final_move == CNCTransition(EpisodePhase.FINAL_A, "A", None)
    assert sequence.next_instruction == "Press SPACE to finish the episode."

    assert sequence.advance() is None
    assert sequence.phase is EpisodePhase.COMPLETE
    with pytest.raises(RuntimeError, match="already complete"):
        sequence.advance()


def test_drawer_can_be_fixed_or_reproducibly_random() -> None:
    assert choose_drawer(drawer=9, seed=None) == 9
    assert choose_drawer(drawer=None, seed=1234) == choose_drawer(drawer=None, seed=1234)
    assert 1 <= choose_drawer(drawer=None, seed=None) <= 9

    with pytest.raises(ValueError, match="between 1 and 9"):
        choose_drawer(drawer=10, seed=None)


def test_named_cnc_controller_sends_validated_absolute_a_move(tmp_path: Path) -> None:
    calibration_path = tmp_path / "cnc.json"
    _write_cnc_calibration(calibration_path)
    client = Mock(spec=FMC4030Client)

    controller = NamedCNCController(
        calibration_path=calibration_path,
        client=client,
        speed=200.0,
        acceleration=200.0,
        deceleration=200.0,
        settle_seconds=2.0,
    )
    with patch("hardware.teleop_episode.move_to_absolute_targets_timed") as command:
        move = controller.command(CNCTransition(EpisodePhase.START_A, "A", None))

    assert move.targets_mm == {Axis.X: 0.0, Axis.Y: -100.0, Axis.Z: 100.0}
    assert move.conservative_duration_seconds == pytest.approx(4.0)
    command.assert_called_once_with(
        client,
        move.targets_mm,
        [Axis.Z, Axis.X, Axis.Y],
        200.0,
        200.0,
        200.0,
        settle_seconds=2.0,
        read_initial_status=False,
        read_final_status=False,
        travel_distance_bounds_mm={Axis.X: 200.0, Axis.Y: 200.0, Axis.Z: 200.0},
    )


def test_named_cnc_controller_rejects_wrong_controller(tmp_path: Path) -> None:
    calibration_path = tmp_path / "cnc.json"
    _write_cnc_calibration(calibration_path)

    with pytest.raises(ValueError, match="Calibration belongs"):
        NamedCNCController(calibration_path=calibration_path, ip="192.168.0.99")


def test_named_cnc_controller_stops_every_axis_after_interruption(tmp_path: Path) -> None:
    calibration_path = tmp_path / "cnc.json"
    _write_cnc_calibration(calibration_path)
    client = Mock(spec=FMC4030Client)
    controller = NamedCNCController(calibration_path=calibration_path, client=client)

    controller.stop_all()

    assert [invocation.args for invocation in client.stop.call_args_list] == [
        (Axis.Z, StopMode.IMMEDIATE),
        (Axis.X, StopMode.IMMEDIATE),
        (Axis.Y, StopMode.IMMEDIATE),
    ]


def test_episode_cli_defaults_and_fixed_drawer() -> None:
    args = parse_args(["--drawer", "4"])

    assert args.drawer == 4
    assert args.cnc_speed == 200.0
    assert args.cnc_acceleration == 200.0
    assert args.cnc_deceleration == 200.0
    assert args.cnc_order == [Axis.Z, Axis.X, Axis.Y]


def test_null_sink_accepts_future_dataset_events() -> None:
    sink = NullEpisodeSink()
    sink.start(drawer=3, joints=("shoulder_r",))
    sink.append(
        EpisodeFrame(
            index=0,
            elapsed_seconds=0.0,
            drawer=3,
            phase=EpisodePhase.OPEN_DRAWER_B,
            cnc_position="B",
            cnc_motion_pending=False,
            leader_positions_rad={"shoulder_r": 0.1},
            follower_targets_rad={"shoulder_r": 0.1},
            follower_positions_rad={"shoulder_r": 0.09},
        )
    )
    sink.finish(completed=True)
