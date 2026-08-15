"""Deterministic MuJoCo backend shared by recording and policy rollout."""

from __future__ import annotations

import colorsys
import time
from dataclasses import dataclass
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
IK_TARGET_GEOM_GROUP = 3
HIDDEN_MARKER_GEOM_GROUP = 5

IK_TARGET_MARKER_PREFIXES = (
    "hand_tip_marker_",
    "drawer_target_marker_",
    "above_drawer_target_marker_",
    "drawer_close_target_marker_",
)

STORAGE_TRANSLATION_NOISE_M = 0.030
STORAGE_YAW_NOISE_DEG = 5.0
DRAWER_STACK_TRANSLATION_NOISE_M = 0.010
DRAWER_STACK_YAW_NOISE_DEG = 3.0
DRAWER_INITIAL_OPENING_NOISE_M = 0.010
HEAD_CAMERA_TRANSLATION_NOISE_M = 0.003
HEAD_CAMERA_ROTATION_NOISE_DEG = 2.0
HEAD_CAMERA_FOVY_NOISE_DEG = 3.0


@dataclass(frozen=True)
class _ImageRandomization:
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    blur_blend: float = 0.0
    noise_std: float = 0.0


def _yaw_quaternion(angle: float) -> np.ndarray:
    return np.array((np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)))


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(result, left, right)
    return result


def _rotation_from_rpy(angles: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(angles, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def _matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).reshape(9))
    return quaternion


def _kelvin_rgb(temperature_kelvin: float) -> np.ndarray:
    """Approximate black-body color as linearized RGB multipliers."""

    temperature = np.clip(temperature_kelvin, 1000.0, 40000.0) / 100.0
    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * np.log(temperature) - 161.1195681661
        blue = (
            0.0
            if temperature <= 19.0
            else 138.5177312231 * np.log(temperature - 10.0) - 305.0447927307
        )
    else:
        red = 329.698727446 * (temperature - 60.0) ** -0.1332047592
        green = 288.1221695283 * (temperature - 60.0) ** -0.0755148492
        blue = 255.0
    srgb = np.clip((red, green, blue), 0.0, 255.0) / 255.0
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )


def _random_color(
    rng: np.random.Generator,
    *,
    saturation: tuple[float, float] = (0.25, 0.75),
    value: tuple[float, float] = (0.45, 0.95),
) -> np.ndarray:
    rgb = colorsys.hsv_to_rgb(
        float(rng.uniform(0.0, 1.0)),
        float(rng.uniform(*saturation)),
        float(rng.uniform(*value)),
    )
    return np.array((*rgb, 1.0), dtype=float)


def default_model_path() -> Path:
    resource = files("simulation.backends.mujoco") / "assets" / "robot.xml"
    with as_file(resource) as model_path:
        return Path(model_path)


class MujocoBackend(SimulationBackend):
    """A single MuJoCo model, data, renderer, and optional native viewer."""

    name = "mujoco"

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        unknown = set(config.options) - {"model_path", "domain_randomization_scale"}
        if unknown:
            raise ValueError(f"Unknown MuJoCo backend options: {sorted(unknown)}")
        configured_path = config.options.get("model_path")
        self.model_path = Path(configured_path) if configured_path else default_model_path()
        self.domain_randomization_scale = float(
            config.options.get("domain_randomization_scale", 1.0)
        )
        if self.domain_randomization_scale < 0.0:
            raise ValueError("domain_randomization_scale must be non-negative")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._configure_geom_groups()
        self._closed = False
        self._renderer: mujoco.Renderer | None = None
        self._render_options = mujoco.MjvOption()
        self._render_options.geomgroup[:] = False
        self._render_options.geomgroup[VISUAL_GEOM_GROUP] = True
        self._viewer: Any | None = None
        self._image_randomization = _ImageRandomization()
        self._image_rng = np.random.default_rng(0)
        self._appearance_rng = np.random.default_rng(0)

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
        self._capture_domain_defaults()
        if config.render:
            self._renderer = mujoco.Renderer(
                self.model, height=config.height, width=config.width
            )
        self.reset()
        if config.viewer:
            self.open_viewer(debug=config.debug)

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
        self._restore_domain_defaults()
        if seed is not None:
            self._apply_domain_randomization(seed)
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
        self._randomize_colors(self._appearance_rng, self.domain_randomization_scale)
        self._randomize_lights(self._appearance_rng, self.domain_randomization_scale)
        self._renderer.update_scene(
            self.data,
            camera=self.config.camera,
            scene_option=self._render_options,
        )
        image = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
        if self._image_randomization == _ImageRandomization():
            return image
        return self._postprocess_image(image)

    def capture_deferred_observation(self) -> tuple[dict[str, Any], np.ndarray]:
        """Capture compact state for rendering after an attempt is accepted."""

        positions = self.joint_positions()
        observation = {
            f"{name}.pos": float(value)
            for name, value in zip(ACTUATOR_NAMES, positions, strict=True)
        }
        return observation, self.data.qpos.copy()

    def materialize_deferred_observation(
        self,
        observation: dict[str, Any],
        render_state: np.ndarray,
    ) -> dict[str, Any]:
        """Render one previously captured state without advancing its physics."""

        self.data.qpos[:] = render_state
        mujoco.mj_forward(self.model, self.data)
        return {**observation, self.config.camera: self.render()}

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

    def _configure_geom_groups(self) -> None:
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            self.model.geom_group[floor_id] = VISUAL_GEOM_GROUP

        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name or "_marker_" not in name:
                continue
            if name.startswith("head_camera_marker_"):
                group = CAMERA_FRAME_GEOM_GROUP
            elif name.startswith(IK_TARGET_MARKER_PREFIXES):
                group = IK_TARGET_GEOM_GROUP
            else:
                group = HIDDEN_MARKER_GEOM_GROUP
            self.model.geom_group[geom_id] = group

    def _capture_domain_defaults(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self._nominal_geom_rgba = self.model.geom_rgba.copy()
        self._nominal_body_pos = self.model.body_pos.copy()
        self._nominal_body_quat = self.model.body_quat.copy()
        self._nominal_qpos0 = self.model.qpos0.copy()
        self._nominal_cam_pos = self.model.cam_pos.copy()
        self._nominal_cam_quat = self.model.cam_quat.copy()
        self._nominal_cam_fovy = self.model.cam_fovy.copy()
        self._nominal_light_pos = self.model.light_pos.copy()
        self._nominal_light_dir = self.model.light_dir.copy()
        self._nominal_light_ambient = self.model.light_ambient.copy()
        self._nominal_light_diffuse = self.model.light_diffuse.copy()
        self._nominal_light_specular = self.model.light_specular.copy()
        self._nominal_haze = np.asarray(self.model.vis.rgba.haze).copy()

        body_names = (
            "storage_bin_link",
            "rack_structure_link",
            *(f"drawer_{index}_link" for index in range(1, 10)),
        )
        self._domain_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        }
        self._drawer_joint_ids = tuple(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"base_link_base_drawer_{index}_joint",
            )
            for index in range(1, 10)
        )
        self._head_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )

        storage_marker_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "storage_bin_center_marker",
        )
        self._storage_pivot = (
            self.data.xpos[storage_marker_id].copy()
            if storage_marker_id >= 0
            else np.zeros(3)
        )
        rack_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith("rack_structure_link_collision_")
        ]
        self._drawer_stack_pivot = (
            np.mean(self.data.geom_xpos[rack_geom_ids], axis=0)
            if rack_geom_ids
            else np.zeros(3)
        )

    def _restore_domain_defaults(self) -> None:
        self.model.geom_rgba[:] = self._nominal_geom_rgba
        self.model.body_pos[:] = self._nominal_body_pos
        self.model.body_quat[:] = self._nominal_body_quat
        self.model.qpos0[:] = self._nominal_qpos0
        self.model.cam_pos[:] = self._nominal_cam_pos
        self.model.cam_quat[:] = self._nominal_cam_quat
        self.model.cam_fovy[:] = self._nominal_cam_fovy
        self.model.light_pos[:] = self._nominal_light_pos
        self.model.light_dir[:] = self._nominal_light_dir
        self.model.light_ambient[:] = self._nominal_light_ambient
        self.model.light_diffuse[:] = self._nominal_light_diffuse
        self.model.light_specular[:] = self._nominal_light_specular
        self.model.vis.rgba.haze[:] = self._nominal_haze
        self._image_randomization = _ImageRandomization()

    def _transform_bodies(
        self,
        body_ids: tuple[int, ...],
        *,
        pivot: np.ndarray,
        translation: np.ndarray,
        yaw: float,
    ) -> None:
        rotation = _rotation_from_rpy(np.array((0.0, 0.0, yaw)))
        yaw_quaternion = _yaw_quaternion(yaw)
        for body_id in body_ids:
            if body_id < 0:
                continue
            nominal_position = self._nominal_body_pos[body_id]
            self.model.body_pos[body_id] = (
                pivot + rotation @ (nominal_position - pivot) + translation
            )
            self.model.body_quat[body_id] = _quaternion_multiply(
                yaw_quaternion, self._nominal_body_quat[body_id]
            )

    def _randomize_colors(self, rng: np.random.Generator, scale: float) -> None:
        if scale <= 0.0:
            return

        def geom_id(name: str) -> int:
            return mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )

        rack_id = geom_id("rack_structure_link_geom")
        storage_id = geom_id("storage_bin_link_geom")
        cube_id = geom_id("cube_link_collision_box_01_geom")
        floor_id = geom_id("floor")
        if rack_id >= 0:
            self.model.geom_rgba[rack_id] = _random_color(rng)
        if storage_id >= 0:
            self.model.geom_rgba[storage_id] = _random_color(rng)
        if cube_id >= 0:
            self.model.geom_rgba[cube_id] = _random_color(
                rng, saturation=(0.55, 0.95), value=(0.65, 1.0)
            )
        if floor_id >= 0:
            self.model.geom_rgba[floor_id] = _random_color(
                rng, saturation=(0.02, 0.25), value=(0.25, 0.75)
            )
        for drawer_index in range(1, 10):
            drawer_id = geom_id(f"drawer_{drawer_index}_link_geom")
            if drawer_id >= 0:
                self.model.geom_rgba[drawer_id] = _random_color(rng)

        excluded_prefixes = ("rack_structure_", "storage_bin_", "drawer_", "cube_")
        robot_color = _random_color(
            rng, saturation=(0.05, 0.45), value=(0.45, 0.9)
        )
        for candidate in range(self.model.ngeom):
            name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, candidate
                )
                or ""
            )
            if (
                self.model.geom_group[candidate] != VISUAL_GEOM_GROUP
                or name == "floor"
                or name.startswith(excluded_prefixes)
                or not name.endswith("_geom")
            ):
                continue
            part_jitter = rng.uniform(-0.08, 0.08, size=3)
            self.model.geom_rgba[candidate, :3] = np.clip(
                robot_color[:3] + part_jitter, 0.0, 1.0
            )
            self.model.geom_rgba[candidate, 3] = 1.0

    def _randomize_lights(self, rng: np.random.Generator, scale: float) -> None:
        if scale <= 0.0:
            return

        for light_id in range(self.model.nlight):
            self.model.light_pos[light_id] = (
                self._nominal_light_pos[light_id]
                + rng.uniform((-0.45, -0.45, -0.25), (0.45, 0.45, 0.25)) * scale
            )
            direction = self._nominal_light_dir[light_id] + rng.uniform(
                -0.20 * scale, 0.20 * scale, size=3
            )
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-9:
                self.model.light_dir[light_id] = direction / direction_norm
            color = _kelvin_rgb(rng.uniform(3500.0, 7500.0))
            intensity = rng.uniform(0.65, 1.25)
            ambient = rng.uniform(0.08, 0.28)
            specular = rng.uniform(0.10, 0.40)
            self.model.light_diffuse[light_id] = np.clip(
                color * intensity, 0.0, 1.0
            )
            self.model.light_ambient[light_id] = np.clip(
                color * ambient, 0.0, 1.0
            )
            self.model.light_specular[light_id] = np.clip(
                color * specular, 0.0, 1.0
            )

    def _apply_domain_randomization(self, seed: int) -> None:
        scale = self.domain_randomization_scale
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x444F4D41])
        )
        self._image_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x494D4147])
        )
        self._appearance_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x4652414D])
        )
        if scale == 0.0:
            return

        background = _random_color(
            rng, saturation=(0.02, 0.30), value=(0.18, 0.75)
        )
        self.model.vis.rgba.haze[:] = background

        storage_translation = np.array(
            [
                rng.uniform(-STORAGE_TRANSLATION_NOISE_M, STORAGE_TRANSLATION_NOISE_M),
                rng.uniform(-STORAGE_TRANSLATION_NOISE_M, STORAGE_TRANSLATION_NOISE_M),
                0.0,
            ]
        ) * scale
        storage_yaw = np.deg2rad(
            rng.uniform(-STORAGE_YAW_NOISE_DEG, STORAGE_YAW_NOISE_DEG) * scale
        )
        self._transform_bodies(
            (self._domain_body_ids["storage_bin_link"],),
            pivot=self._storage_pivot,
            translation=storage_translation,
            yaw=storage_yaw,
        )

        drawer_translation = np.array(
            [
                rng.uniform(
                    -DRAWER_STACK_TRANSLATION_NOISE_M,
                    DRAWER_STACK_TRANSLATION_NOISE_M,
                ),
                rng.uniform(
                    -DRAWER_STACK_TRANSLATION_NOISE_M,
                    DRAWER_STACK_TRANSLATION_NOISE_M,
                ),
                0.0,
            ]
        ) * scale
        drawer_yaw = np.deg2rad(
            rng.uniform(-DRAWER_STACK_YAW_NOISE_DEG, DRAWER_STACK_YAW_NOISE_DEG)
            * scale
        )
        drawer_body_ids = (
            self._domain_body_ids["rack_structure_link"],
            *(self._domain_body_ids[f"drawer_{index}_link"] for index in range(1, 10)),
        )
        self._transform_bodies(
            drawer_body_ids,
            pivot=self._drawer_stack_pivot,
            translation=drawer_translation,
            yaw=drawer_yaw,
        )
        for joint_id in self._drawer_joint_ids:
            if joint_id < 0:
                continue
            opening = rng.uniform(
                0.0,
                min(
                    DRAWER_INITIAL_OPENING_NOISE_M * scale,
                    float(np.ptp(self.model.jnt_range[joint_id])),
                ),
            )
            qpos_id = int(self.model.jnt_qposadr[joint_id])
            self.model.qpos0[qpos_id] = self.model.jnt_range[joint_id, 1] - opening

        if self._head_camera_id >= 0:
            camera_id = self._head_camera_id
            self.model.cam_pos[camera_id] = (
                self._nominal_cam_pos[camera_id]
                + rng.uniform(
                    -HEAD_CAMERA_TRANSLATION_NOISE_M * scale,
                    HEAD_CAMERA_TRANSLATION_NOISE_M * scale,
                    size=3,
                )
            )
            camera_angles = np.deg2rad(
                rng.uniform(
                    -HEAD_CAMERA_ROTATION_NOISE_DEG * scale,
                    HEAD_CAMERA_ROTATION_NOISE_DEG * scale,
                    size=3,
                )
            )
            camera_delta = _matrix_quaternion(_rotation_from_rpy(camera_angles))
            self.model.cam_quat[camera_id] = _quaternion_multiply(
                self._nominal_cam_quat[camera_id], camera_delta
            )
            self.model.cam_fovy[camera_id] = np.clip(
                self._nominal_cam_fovy[camera_id]
                + rng.uniform(-HEAD_CAMERA_FOVY_NOISE_DEG, HEAD_CAMERA_FOVY_NOISE_DEG)
                * scale,
                20.0,
                120.0,
            )

        self._image_randomization = _ImageRandomization(
            brightness=float(rng.uniform(-0.08, 0.08) * scale),
            contrast=float(1.0 + rng.uniform(-0.15, 0.15) * scale),
            saturation=float(1.0 + rng.uniform(-0.18, 0.18) * scale),
            gamma=float(1.0 + rng.uniform(-0.12, 0.12) * scale),
            blur_blend=float(np.clip(rng.uniform(0.0, 0.35) * scale, 0.0, 1.0)),
            noise_std=float(rng.uniform(0.0, 0.012) * scale),
        )

    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        randomization = self._image_randomization
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        luminance = np.sum(
            pixels * np.array((0.2126, 0.7152, 0.0722), dtype=np.float32),
            axis=2,
            keepdims=True,
        )
        pixels = luminance + randomization.saturation * (pixels - luminance)
        pixels = (pixels - 0.5) * randomization.contrast + 0.5
        pixels += randomization.brightness
        pixels = np.clip(pixels, 0.0, 1.0) ** (1.0 / randomization.gamma)

        if randomization.blur_blend > 0.0:
            padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
            blurred = (
                4.0 * padded[1:-1, 1:-1]
                + 2.0
                * (
                    padded[:-2, 1:-1]
                    + padded[2:, 1:-1]
                    + padded[1:-1, :-2]
                    + padded[1:-1, 2:]
                )
                + padded[:-2, :-2]
                + padded[:-2, 2:]
                + padded[2:, :-2]
                + padded[2:, 2:]
            ) / 16.0
            pixels = (
                (1.0 - randomization.blur_blend) * pixels
                + randomization.blur_blend * blurred
            )

        if randomization.noise_std > 0.0:
            pixels += self._image_rng.normal(
                0.0, randomization.noise_std, size=pixels.shape
            ).astype(np.float32)
        return np.asarray(
            np.clip(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0), 0.0, 255.0),
            dtype=np.uint8,
        )

    def _configure_viewer_groups(self, *, debug: bool) -> None:
        if self._viewer is None:
            raise RuntimeError("Viewer must be open before configuring visualization")

        with self._viewer.lock():
            self._viewer.opt.geomgroup[:] = False
            self._viewer.opt.geomgroup[VISUAL_GEOM_GROUP] = True
            self._viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = debug
            self._viewer.opt.geomgroup[CAMERA_FRAME_GEOM_GROUP] = debug
            self._viewer.opt.geomgroup[IK_TARGET_GEOM_GROUP] = debug
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
