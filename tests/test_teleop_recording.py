from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hepha_lerobot.recording import teleop

from hardware.feetech_leader import (
    FeetechLeader,
    compose_action,
    configure_cnc_start_pose,
    load_calibrated_servos,
)
from simulation.backends.mujoco import ACTUATOR_NAMES


def _write_calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "port": "/dev/cu.test",
                "baudrate": 1_000_000,
                "axes": {
                    "2": {
                        "label": "shoulder left",
                        "mujoco_actuator": "shoulder_l",
                        "q_min": -1.0,
                        "q_home": 0.0,
                        "q_max": 1.0,
                        "raw_home": 1000,
                        "raw_min_delta": -500,
                        "raw_max_delta": 500,
                    },
                    "4": {
                        "label": "forearm left",
                        "mujoco_actuator": "forearm_l",
                        "q_min": -0.5,
                        "q_home": 0.0,
                        "q_max": 1.5,
                        "raw_home": 2000,
                        "raw_min_delta": -250,
                        "raw_max_delta": 750,
                    },
                },
            }
        )
    )


def test_load_calibrated_servos_defaults_to_every_saved_axis(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    _write_calibration(path)

    calibration, servos = load_calibrated_servos(path, None)

    assert calibration["baudrate"] == 1_000_000
    assert [servo.servo_id for servo in servos] == [2, 4]
    assert [servo.actuator for servo in servos] == ["shoulder_l", "forearm_l"]


def test_load_calibrated_servos_rejects_missing_requested_axis(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    _write_calibration(path)

    with pytest.raises(RuntimeError, match="Servo ID 8 is not calibrated"):
        load_calibrated_servos(path, [8])


def test_compose_action_overlays_only_selected_actuators() -> None:
    base = np.arange(len(ACTUATOR_NAMES), dtype=float)

    action = compose_action(
        base,
        {"shoulder_l": -0.25, "wrist_r": 0.75},
        ACTUATOR_NAMES,
    )

    assert action[ACTUATOR_NAMES.index("shoulder_l")] == -0.25
    assert action[ACTUATOR_NAMES.index("wrist_r")] == 0.75
    assert action[ACTUATOR_NAMES.index("cnc_x")] == base[0]
    assert action is not base


def test_configure_cnc_start_pose_uses_head_lower_limit() -> None:
    action = np.ones(len(ACTUATOR_NAMES))
    control_low = np.full(len(ACTUATOR_NAMES), -9.0)
    control_low[ACTUATOR_NAMES.index("head_z")] = -0.1

    configured = configure_cnc_start_pose(action, control_low, ACTUATOR_NAMES)

    assert configured[ACTUATOR_NAMES.index("cnc_x")] == 0.0
    assert configured[ACTUATOR_NAMES.index("cnc_y")] == 0.0
    assert configured[ACTUATOR_NAMES.index("head_z")] == -0.1


def test_leader_rejects_stale_mujoco_limits(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    _write_calibration(path)
    leader = FeetechLeader(calibration_path=path, servo_ids=[2])
    low = np.full(len(ACTUATOR_NAMES), -1.0)
    high = np.full(len(ACTUATOR_NAMES), 1.0)
    high[ACTUATOR_NAMES.index("shoulder_l")] = 2.0

    with pytest.raises(RuntimeError, match="do not match MuJoCo"):
        leader.validate_mujoco_ranges(ACTUATOR_NAMES, low, high)


@pytest.mark.parametrize("raw, expected", [(" save ", "SAVE"), ("retry", "RETRY")])
def test_parse_episode_decision(raw: str, expected: str) -> None:
    assert teleop.parse_episode_decision(raw) == expected


def test_parse_episode_decision_rejects_unknown_choice() -> None:
    with pytest.raises(ValueError, match="SAVE, DISCARD, RETRY, QUIT"):
        teleop.parse_episode_decision("maybe")


def test_clear_unsaved_episode_only_clears_populated_buffer() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.writer = SimpleNamespace(episode_buffer={"size": 3})
            self.clear_calls = 0

        def clear_episode_buffer(self) -> None:
            self.clear_calls += 1
            self.writer.episode_buffer["size"] = 0

    dataset = FakeDataset()

    teleop.clear_unsaved_episode(dataset)
    teleop.clear_unsaved_episode(dataset)

    assert dataset.clear_calls == 1


def test_prepare_dataset_root_delays_and_honors_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    marker = root / "marker"
    marker.write_text("existing")

    with pytest.raises(FileExistsError):
        teleop.prepare_dataset_root(root, overwrite=False)
    assert marker.exists()

    teleop.prepare_dataset_root(root, overwrite=True)
    assert not root.exists()
    assert root.parent.exists()


def test_record_attempt_records_observation_and_exact_sent_action(monkeypatch) -> None:
    events: list[str] = []
    recorded: list[dict] = []

    class FakeBackend:
        @staticmethod
        def viewer_is_running() -> bool:
            return True

        @staticmethod
        def get_observation(*, advance: bool) -> dict:
            assert not advance
            events.append("observe")
            return {"joint": 0.0}

        @staticmethod
        def send_action(action: np.ndarray) -> dict:
            events.append("send")
            return {"sent.pos": float(action[0])}

        @staticmethod
        def step() -> None:
            events.append("step")

    class FakeLeader:
        @staticmethod
        def read_action(base_action: np.ndarray, actuator_names) -> np.ndarray:
            assert actuator_names == ACTUATOR_NAMES
            events.append("leader")
            action = base_action.copy()
            action[0] = 0.25
            return action

    def capture_frame(dataset, **kwargs) -> None:
        del dataset
        events.append("record")
        recorded.append(kwargs)

    monkeypatch.setattr(teleop, "add_robot_frame", capture_frame)
    monkeypatch.setattr(teleop.time, "sleep", lambda _seconds: None)
    frames = teleop.record_attempt(
        backend=FakeBackend(),
        leader=FakeLeader(),
        dataset=object(),
        base_action=np.zeros(len(ACTUATOR_NAMES)),
        drawer=4,
        task="Use drawer {drawer_index}",
        fps=2,
        episode_seconds=1.0,
        viewer_required=True,
    )

    assert frames == 2
    assert events == ["observe", "leader", "send", "record", "step"] * 2
    assert [frame["action"] for frame in recorded] == [
        {"sent.pos": 0.25},
        {"sent.pos": 0.25},
    ]
    assert all(frame["current_task_phase"] is None for frame in recorded)
    assert all(frame["next_task_phase"] is None for frame in recorded)


def test_record_dataset_saves_and_finalizes_operator_approved_episode(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []
    action_size = len(ACTUATOR_NAMES)

    class FakeBackend:
        def __init__(self, config) -> None:
            del config
            self.control_low = np.full(action_size, -1.0)
            self.control_high = np.full(action_size, 1.0)

        def __enter__(self):
            events.append("backend-enter")
            return self

        def __exit__(self, *_args) -> None:
            events.append("backend-exit")

        @staticmethod
        def viewer_is_running() -> bool:
            return False

    class FakeLeader:
        port = "/dev/cu.test"
        baudrate = 1_000_000
        servos = ()

        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            events.append("leader-enter")
            return self

        def __exit__(self, *_args) -> None:
            events.append("leader-exit")

        @staticmethod
        def validate_mujoco_ranges(*_args) -> None:
            events.append("validate")

        @staticmethod
        def disable_torque() -> None:
            events.append("disable-torque")

    class FakeDataset:
        def __init__(self) -> None:
            self.writer = SimpleNamespace(episode_buffer={"size": 1})
            self.saved = 0
            self.finalized = False

        def save_episode(self) -> None:
            self.saved += 1
            self.writer.episode_buffer["size"] = 0
            events.append("save")

        def clear_episode_buffer(self) -> None:
            self.writer.episode_buffer["size"] = 0
            events.append("clear")

        def finalize(self) -> None:
            self.finalized = True
            events.append("finalize")

    dataset = FakeDataset()
    answers = iter(("", "", "SAVE"))
    monkeypatch.setattr(teleop, "MujocoBackend", FakeBackend)
    monkeypatch.setattr(teleop, "FeetechLeader", FakeLeader)
    monkeypatch.setattr(teleop, "create_dataset", lambda **_kwargs: dataset)
    monkeypatch.setattr(
        teleop,
        "initialize_from_leader",
        lambda *_args, **_kwargs: (np.zeros(action_size), 3),
    )
    monkeypatch.setattr(teleop, "countdown", lambda _seconds: None)
    monkeypatch.setattr(teleop, "record_attempt", lambda **_kwargs: 7)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    args = SimpleNamespace(
        camera="head_camera",
        width=64,
        height=64,
        fps=10,
        debug=False,
        calibration=tmp_path / "calibration.json",
        ids=[2],
        port=None,
        retries=2,
        smoothing=0.25,
        root=tmp_path / "dataset",
        overwrite=False,
        repo_id="hepha/test_teleop",
        no_video=True,
        episodes=1,
        seed=0,
        drawer_index=3,
        viewer=False,
        countdown=0.0,
        episode_seconds=1.0,
        task="Use drawer {drawer_index}",
        push_to_hub=False,
    )

    result = teleop.record_dataset(args)

    assert result == args.root
    assert dataset.saved == 1
    assert dataset.finalized
    assert events == [
        "backend-enter",
        "validate",
        "leader-enter",
        "disable-torque",
        "save",
        "finalize",
        "leader-exit",
        "backend-exit",
    ]
