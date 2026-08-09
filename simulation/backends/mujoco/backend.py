"""Deterministic MuJoCo backend shared by recording and policy rollout."""

from __future__ import annotations

import time
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from simulation.base import FeatureSpec, SimulationBackend, SimulationConfig

ACTUATOR_NAMES = (
    "cnc_x",
    "cnc_y",
    "head_z",
    "shoulder_l",
    "forearm_l",
    "arm_l",
    "wrist_l",
    "hand_l",
    "finger_l",
    "shoulder_r",
    "forearm_r",
    "arm_r",
    "wrist_r",
    "hand_r",
    "finger_r",
)

COLLISION_GEOM_GROUP = 0
VISUAL_GEOM_GROUP = 1
CAMERA_FRAME_GEOM_GROUP = 2
HIDDEN_MARKER_GEOM_GROUP = 5


def default_model_path() -> Path:
    resource = files("simulation.backends.mujoco") / "assets" / "robot.xml"
    with as_file(resource) as model_path:
        return Path(model_path)


class MujocoBackend(SimulationBackend):
    """A single MuJoCo model, data, renderer, and optional native viewer."""

    name = "mujoco"

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        unknown = set(config.options) - {"model_path"}
        if unknown:
            raise ValueError(f"Unknown MuJoCo backend options: {sorted(unknown)}")
        configured_path = config.options.get("model_path")
        self.model_path = Path(configured_path) if configured_path else default_model_path()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._closed = False
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

        self.actuator_ids = self._resolve_actuator_ids()
        self.joint_ids = self.model.actuator_trnid[self.actuator_ids, 0].astype(int)
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.control_low = self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.control_high = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.steps_per_control = max(
            1, round((1 / config.fps) / self.model.opt.timestep)
        )

        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, config.camera
        )
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera not found: {config.camera}")
        if config.render:
            self._renderer = mujoco.Renderer(
                self.model, height=config.height, width=config.width
            )
        self.reset()
        if config.viewer:
            self.open_viewer()

    @classmethod
    def observation_features_for(
        cls, config: SimulationConfig
    ) -> dict[str, FeatureSpec]:
        state = {f"{name}.pos": float for name in ACTUATOR_NAMES}
        return {**state, config.camera: (config.height, config.width, 3)}

    @classmethod
    def action_features_for(cls, config: SimulationConfig) -> dict[str, type]:
        del config
        return {f"{name}.pos": float for name in ACTUATOR_NAMES}

    def _resolve_actuator_ids(self) -> np.ndarray:
        ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ACTUATOR_NAMES
            ],
            dtype=int,
        )
        missing = [
            name
            for name, actuator_id in zip(ACTUATOR_NAMES, ids, strict=True)
            if actuator_id < 0
        ]
        if missing:
            raise RuntimeError(f"MJCF is missing required actuators: {missing}")
        return ids

    @property
    def is_open(self) -> bool:
        return not self._closed

    def home_action(self) -> np.ndarray:
        home = 0.5 * (self.control_low + self.control_high)
        for finger in ("finger_l", "finger_r"):
            home[ACTUATOR_NAMES.index(finger)] = 1.0
        return home

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        home = self.home_action()
        self.data.qpos[self.qpos_ids] = home
        self.data.ctrl[self.actuator_ids] = home
        mujoco.mj_forward(self.model, self.data)
        self.sync_viewer()

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].astype(np.float32, copy=True)

    def action_dict(self, values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=float)
        if values.shape != (len(ACTUATOR_NAMES),):
            raise ValueError(
                f"Expected {len(ACTUATOR_NAMES)} actions, got shape {values.shape}"
            )
        return {
            f"{name}.pos": float(value)
            for name, value in zip(ACTUATOR_NAMES, values, strict=True)
        }

    def action_array(self, action: dict[str, Any]) -> np.ndarray:
        try:
            values = np.array(
                [action[f"{name}.pos"] for name in ACTUATOR_NAMES], dtype=float
            )
        except KeyError as error:
            raise ValueError(f"Action is missing required key: {error.args[0]}") from error
        return np.clip(values, self.control_low, self.control_high)

    def send_action(self, action: dict[str, Any] | Any) -> dict[str, float]:
        values = (
            self.action_array(action)
            if isinstance(action, dict)
            else np.asarray(action, dtype=float)
        )
        if values.shape != (len(ACTUATOR_NAMES),):
            raise ValueError(
                f"Expected {len(ACTUATOR_NAMES)} actions, got shape {values.shape}"
            )
        sent = np.clip(values, self.control_low, self.control_high)
        self.data.ctrl[self.actuator_ids] = sent
        return self.action_dict(sent)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.steps_per_control)
        self.sync_viewer()

    def render(self) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("Rendering was disabled for this simulation")
        self._renderer.update_scene(self.data, camera=self.config.camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def get_observation(self, *, advance: bool = True) -> dict[str, Any]:
        if advance:
            self.step()
        positions = self.joint_positions()
        observation: dict[str, Any] = {
            f"{name}.pos": float(value)
            for name, value in zip(ACTUATOR_NAMES, positions, strict=True)
        }
        observation[self.config.camera] = self.render()
        return observation

    def _configure_viewer_groups(self, *, debug: bool) -> None:
        if self._viewer is None:
            raise RuntimeError("Viewer must be open before configuring visualization")

        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            self.model.geom_group[floor_id] = VISUAL_GEOM_GROUP

        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name or "_marker_" not in name:
                continue
            self.model.geom_group[geom_id] = (
                CAMERA_FRAME_GEOM_GROUP
                if name.startswith("head_camera_marker_")
                else HIDDEN_MARKER_GEOM_GROUP
            )

        with self._viewer.lock():
            self._viewer.opt.geomgroup[:] = False
            self._viewer.opt.geomgroup[VISUAL_GEOM_GROUP] = True
            self._viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = debug
            self._viewer.opt.geomgroup[CAMERA_FRAME_GEOM_GROUP] = debug
            self._viewer.opt.geomgroup[HIDDEN_MARKER_GEOM_GROUP] = False

    def open_viewer(self, *, debug: bool = False) -> None:
        if self._viewer is not None:
            return
        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=True,
            show_right_ui=True,
        )
        self._configure_viewer_groups(debug=debug)

    def sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def viewer_is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def run_interactive(self, *, debug: bool = False) -> None:
        self.open_viewer(debug=debug)
        print("Use MuJoCo's native Control panel to move actuators within their limits.")
        interval = 1 / self.config.fps
        while self.viewer_is_running():
            started = time.perf_counter()
            self.step()
            time.sleep(max(0.0, interval - (time.perf_counter() - started)))

    def close(self) -> None:
        if self._closed:
            return
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True
