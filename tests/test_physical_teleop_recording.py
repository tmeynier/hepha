from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hepha_lerobot.recording import physical_teleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features


def test_physical_dataset_schema_contains_only_arm_and_camera(monkeypatch, tmp_path) -> None:
    captured = {}
    sentinel = object()

    def fake_create(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(physical_teleop.LeRobotDataset, "create", fake_create)
    result = physical_teleop.create_physical_dataset(
        repo_id="user/physical",
        root=tmp_path / "dataset",
        joints=("shoulder_r", "finger_l"),
        camera_name="head_camera",
        width=320,
        height=240,
        fps=30,
        use_videos=True,
    )

    assert result is sentinel
    assert captured["robot_type"] == "hepha_physical_feetech"
    assert captured["features"]["action"]["names"] == [
        "shoulder_r.pos",
        "finger_l.pos",
    ]
    assert captured["features"]["observation.state"]["names"] == [
        "shoulder_r.pos",
        "finger_l.pos",
    ]
    assert captured["features"]["observation.images.head_camera"]["shape"] == (
        240,
        320,
        3,
    )
    assert not any("cnc" in name for name in captured["features"])
    assert not any(
        "cnc" in name
        for feature in captured["features"].values()
        for name in feature.get("names") or []
    )


def test_add_physical_frame_records_follower_state_and_command() -> None:
    joints = ("shoulder_r", "shoulder_l")
    joint_features = physical_teleop.joint_feature_names(joints)
    features = combine_feature_dicts(
        hw_to_dataset_features(joint_features, ACTION, use_video=False),
        hw_to_dataset_features(
            {**joint_features, "head_camera": (2, 3, 3)},
            OBS_STR,
            use_video=False,
        ),
    )

    class FakeDataset:
        def __init__(self) -> None:
            self.features = features
            self.frames = []

        def add_frame(self, frame) -> None:
            self.frames.append(frame)

    dataset = FakeDataset()
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    physical_teleop.add_physical_frame(
        dataset,
        joints=joints,
        follower_positions={"shoulder_r": 0.1, "shoulder_l": 0.2},
        follower_commands={"shoulder_r": 0.3, "shoulder_l": 0.4},
        camera_name="head_camera",
        image=image,
        task="Move the robot",
    )

    frame = dataset.frames[0]
    assert np.allclose(frame["observation.state"], [0.1, 0.2])
    assert np.allclose(frame["action"], [0.3, 0.4])
    assert frame["observation.images.head_camera"] is image
    assert frame["task"] == "Move the robot"


def test_camera_frame_converts_bgr_to_rgb() -> None:
    bgr = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    capture = SimpleNamespace(read=lambda: (True, bgr))

    preview, rgb = physical_teleop.camera_frame(capture, width=1, height=1)

    assert preview is bgr
    assert rgb.tolist() == [[[3, 2, 1]]]


def test_preview_window_is_large_and_resizable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(physical_teleop.cv2, "namedWindow", lambda *args: calls.append(args))
    monkeypatch.setattr(physical_teleop.cv2, "resizeWindow", lambda *args: calls.append(args))

    physical_teleop.configure_preview_window(
        "camera",
        width=1280,
        height=720,
        fullscreen=False,
    )

    assert calls == [
        (
            "camera",
            physical_teleop.cv2.WINDOW_NORMAL | physical_teleop.cv2.WINDOW_KEEPRATIO,
        ),
        ("camera", 1280, 720),
    ]


def test_preview_window_can_be_fullscreen(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(physical_teleop.cv2, "namedWindow", lambda *args: calls.append(args))
    monkeypatch.setattr(physical_teleop.cv2, "resizeWindow", lambda *args: calls.append(args))
    monkeypatch.setattr(physical_teleop.cv2, "imshow", lambda *args: calls.append(args[:1]))
    monkeypatch.setattr(physical_teleop.cv2, "waitKey", lambda *_args: -1)
    monkeypatch.setattr(
        physical_teleop.cv2,
        "setWindowProperty",
        lambda *args: calls.append(args),
    )

    interface = physical_teleop.RecordingInterface(
        capture=None,
        window_name="camera",
        record_width=256,
        record_height=256,
        preview_width=1280,
        preview_height=720,
        fullscreen=True,
    )
    interface.open(np.zeros((240, 320, 3), dtype=np.uint8))

    assert calls[-1] == (
        "camera",
        physical_teleop.cv2.WND_PROP_FULLSCREEN,
        physical_teleop.cv2.WINDOW_FULLSCREEN,
    )


def test_record_episode_uses_observation_then_action_order(monkeypatch) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    calls = []

    def read_camera():
        calls.append("read_camera")
        return image, image

    def read_leader():
        calls.append("read_leader")
        return {"shoulder_r": 0.3}

    def read_follower():
        calls.append("read_state")
        return {"shoulder_r": 0.1}

    def limit(targets, _dt):
        calls.append("limit")
        return targets

    def send(targets):
        calls.append("send")
        return targets

    interface = SimpleNamespace(
        read=read_camera,
        present=lambda *_args, **_kwargs: "space",
    )
    leader = SimpleNamespace(read_joint_positions=read_leader)
    follower = SimpleNamespace(
        write_joint_positions=send,
        read_joint_positions=read_follower,
    )
    limiter = SimpleNamespace(apply=limit)
    captured = []

    def capture_frame(*_args, **kwargs):
        calls.append("save")
        captured.append(kwargs)

    monkeypatch.setattr(
        physical_teleop,
        "add_physical_frame",
        capture_frame,
    )

    frames = physical_teleop.record_episode(
        args=SimpleNamespace(
            episode_seconds=60.0,
            fps=30,
            task="test",
            camera_name="head_camera",
        ),
        leader=leader,
        follower=follower,
        limiter=limiter,
        dataset=object(),
        joints=("shoulder_r",),
        interface=interface,
    )

    assert frames == 1
    assert len(captured) == 1
    assert calls == ["read_state", "read_camera", "read_leader", "limit", "send", "save"]
    assert captured[0]["follower_positions"] == {"shoulder_r": 0.1}
    assert captured[0]["follower_commands"] == {"shoulder_r": 0.3}
