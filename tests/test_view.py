from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from simulation import SimulationConfig, view
from simulation.backends.mujoco.backend import (
    CAMERA_FRAME_GEOM_GROUP,
    COLLISION_GEOM_GROUP,
    HIDDEN_MARKER_GEOM_GROUP,
    IK_TARGET_GEOM_GROUP,
    VISUAL_GEOM_GROUP,
    MujocoBackend,
)


def test_mjpython_relaunch_is_guarded(monkeypatch, tmp_path: Path) -> None:
    launcher = tmp_path / "mjpython"
    launcher.touch()
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(view.sys, "platform", "darwin")
    monkeypatch.setattr(view.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(view.sys, "argv", ["hepha-view", "--backend", "mujoco"])
    monkeypatch.setattr(view, "_find_environment_executable", lambda _: str(launcher))
    monkeypatch.setattr(view.os, "execv", lambda path, args: calls.append((path, args)))
    monkeypatch.delenv(view.MJPYTHON_GUARD, raising=False)

    view._ensure_mjpython_on_macos()

    assert calls == [
        (
            str(launcher),
            [str(launcher), "-m", "simulation.view", "--backend", "mujoco"],
        )
    ]
    assert view.os.environ[view.MJPYTHON_GUARD] == "1"

    view._ensure_mjpython_on_macos()
    assert len(calls) == 1


def test_debug_accepts_flag_or_explicit_boolean(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["hepha-view", "--debug"])
    assert view.parse_args().debug is True

    monkeypatch.setattr("sys.argv", ["hepha-view", "--debug", "false"])
    assert view.parse_args().debug is False


def test_mjpython_relaunch_can_target_recording_module(monkeypatch, tmp_path: Path) -> None:
    launcher = tmp_path / "mjpython"
    launcher.touch()
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(view.sys, "platform", "darwin")
    monkeypatch.setattr(view.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(view.sys, "argv", ["hepha-record", "--viewer"])
    monkeypatch.setattr(view, "_find_environment_executable", lambda _: str(launcher))
    monkeypatch.setattr(view.os, "execv", lambda path, args: calls.append((path, args)))
    monkeypatch.delenv(view.MJPYTHON_GUARD, raising=False)

    view._ensure_mjpython_on_macos("hepha_lerobot.recording.record")

    assert calls == [
        (
            str(launcher),
            [
                str(launcher),
                "-m",
                "hepha_lerobot.recording.record",
                "--viewer",
            ],
        )
    ]


class _FakeViewer:
    def __init__(self) -> None:
        self.opt = type("Options", (), {"geomgroup": np.ones(6, dtype=bool)})()

    def lock(self):
        return nullcontext()

    def close(self) -> None:
        return None


def _geom_group(backend: MujocoBackend, name: str) -> int:
    geom_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0
    return int(backend.model.geom_group[geom_id])


def test_debug_controls_collisions_camera_axes_and_ik_targets() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        viewer = _FakeViewer()
        backend._viewer = viewer

        assert backend._render_options.geomgroup[VISUAL_GEOM_GROUP]
        assert not backend._render_options.geomgroup[COLLISION_GEOM_GROUP]
        assert not backend._render_options.geomgroup[CAMERA_FRAME_GEOM_GROUP]
        assert not backend._render_options.geomgroup[IK_TARGET_GEOM_GROUP]

        backend._configure_viewer_groups(debug=False)
        assert viewer.opt.geomgroup[VISUAL_GEOM_GROUP]
        assert not viewer.opt.geomgroup[COLLISION_GEOM_GROUP]
        assert not viewer.opt.geomgroup[CAMERA_FRAME_GEOM_GROUP]
        assert not viewer.opt.geomgroup[IK_TARGET_GEOM_GROUP]
        assert _geom_group(backend, "floor") == VISUAL_GEOM_GROUP
        assert (
            _geom_group(backend, "head_camera_marker_x")
            == CAMERA_FRAME_GEOM_GROUP
        )
        assert _geom_group(backend, "cube_frame_marker_x") == HIDDEN_MARKER_GEOM_GROUP
        assert _geom_group(backend, "hand_tip_marker_x") == IK_TARGET_GEOM_GROUP
        assert (
            _geom_group(backend, "drawer_target_marker_x")
            == IK_TARGET_GEOM_GROUP
        )

        backend._configure_viewer_groups(debug=True)
        assert viewer.opt.geomgroup[COLLISION_GEOM_GROUP]
        assert viewer.opt.geomgroup[CAMERA_FRAME_GEOM_GROUP]
        assert viewer.opt.geomgroup[IK_TARGET_GEOM_GROUP]
        assert not viewer.opt.geomgroup[HIDDEN_MARKER_GEOM_GROUP]
