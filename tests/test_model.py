from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
from hepha_lerobot.recording.controllers import available_controllers

from simulation import SimulationConfig, available_backends, create_backend
from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.backends.mujoco.ik import (
    CUBE_GRASP,
    CUBE_GRASP_APPROACH_HEIGHT_M,
    CUBE_PLACE_HANDLE_Y_OFFSET_M,
    CUBE_PLACE_HANDLE_Z_OFFSET_M,
    CUBE_SPAWN_RADIUS_M,
    DRAWER_APPROACH_DISTANCE_M,
    DRAWER_CLOSE_APPROACH_DISTANCE_M,
    DRAWER_GRASP,
    DRAWER_PUSH_MARGIN_M,
    DRAWER_TARGET_Z_OFFSET_M,
    FINGER_CLOSED,
    FINGER_OPEN,
    HANDOFF_DONOR_ARM_ELEVATION_RAD,
    IDLE_FOREARM_DRAWER_PULL_POSITION,
    IK_TARGET_MARKERS,
    MAX_DRAWER_CLOSED_OPENING_M,
    MIN_DRAWER_OPENING_M,
    SIDE_JOINTS,
    TOP_ROW_CUBE_PLACE_CLEARANCE_M,
    CollisionAwareIK,
    MujocoIKController,
    _controlled_joints,
    _cube_above_drawer_pose,
    _drawer_close_target_pose,
    _drawer_hand_target_pose,
    _joint_id,
    _randomize_cube,
    drawer_handle_pose,
    hand_pose,
)


def test_cube_spawn_is_inside_ten_centimeter_disk() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        joint_id = _joint_id(backend.model, "cube_link_free_joint")
        qpos_id = int(backend.model.jnt_qposadr[joint_id])
        center = backend.model.qpos0[qpos_id : qpos_id + 2]
        radii = []

        for seed in range(200):
            _randomize_cube(
                backend.model,
                backend.data,
                np.random.default_rng(seed),
            )
            radii.append(
                np.linalg.norm(backend.data.qpos[qpos_id : qpos_id + 2] - center)
            )

        assert max(radii) <= CUBE_SPAWN_RADIUS_M
        assert min(radii) < 0.02
        assert max(radii) > 0.09


def test_mujoco_model_matches_control_contract() -> None:
    assert "mujoco" in available_backends()
    with create_backend("mujoco", SimulationConfig(render=False)) as backend:
        assert isinstance(backend, MujocoBackend)
        assert backend.model.nu == len(ACTUATOR_NAMES)
        assert backend.joint_positions().shape == (len(ACTUATOR_NAMES),)
        assert tuple(backend.action_features) == tuple(
            f"{name}.pos" for name in ACTUATOR_NAMES
        )
        assert backend.observation_features["head_camera"] == (256, 256, 3)


def test_cube_is_one_rigid_geom_on_the_standard_collision_layer() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        cube_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "cube_link_collision_box_01_geom",
        )
        assert backend.model.geom_contype[cube_id] == 1
        assert backend.model.geom_conaffinity[cube_id] == 1
        assert backend.model.geom_group[cube_id] == 1
        assert backend.model.nflex == 0
        cube_body_id = mujoco.mj_name2id(
            backend.model, mujoco.mjtObj.mjOBJ_BODY, "cube_link"
        )
        assert backend.model.body_geomnum[cube_body_id] == 1

        for side in ("l", "r"):
            for body in ("hand", "finger"):
                for box_index in range(1, 4):
                    geom_id = mujoco.mj_name2id(
                        backend.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"{body}_{side}_link_collision_box_{box_index:02d}_geom",
                    )
                    assert backend.model.geom_contype[geom_id] == 1
                    assert backend.model.geom_conaffinity[geom_id] == 1

        floor_id = mujoco.mj_name2id(
            backend.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        assert backend.model.geom_conaffinity[floor_id] & 1


def test_closed_drawers_are_aligned_to_the_rack_grid() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        model = backend.model
        data = backend.data

        def geom_id(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        shelf_ids = [
            geom_id(f"rack_structure_link_collision_box_0{index}_geom")
            for index in (2, 3, 4)
        ]
        divider_ids = [
            geom_id(f"rack_structure_link_collision_box_0{index}_geom")
            for index in (7, 5, 6, 8)
        ]
        divider_x = data.geom_xpos[divider_ids, 0]
        column_x = (divider_x[:-1] + divider_x[1:]) / 2
        rack_depth = data.geom_xpos[shelf_ids[0], 1]

        for drawer_index in range(1, 10):
            row = (drawer_index - 1) // 3
            column = 2 - ((drawer_index - 1) % 3)
            drawer_id = geom_id(
                f"drawer_{drawer_index}_link_collision_box_01_geom"
            )
            expected = np.array(
                [
                    column_x[column],
                    rack_depth,
                    data.geom_xpos[shelf_ids[row], 2]
                    + model.geom_size[shelf_ids[row], 2]
                    + model.geom_size[drawer_id, 2],
                ]
            )
            assert np.allclose(data.geom_xpos[drawer_id], expected, atol=1e-9)


def test_collision_aware_ik_uses_the_full_side_chain(monkeypatch) -> None:
    def use_middle_of_bounds(_objective, bounds, **_kwargs):
        return SimpleNamespace(x=np.mean(np.asarray(bounds), axis=1))

    monkeypatch.setattr(
        "simulation.backends.mujoco.ik.differential_evolution",
        use_middle_of_bounds,
    )
    monkeypatch.setattr(
        "simulation.backends.mujoco.ik.minimize",
        lambda _objective, initial, **_kwargs: SimpleNamespace(x=initial),
    )

    with MujocoBackend(SimulationConfig(render=False)) as backend:
        target, rotation = hand_pose(backend.model, backend.data, "r")
        solution = CollisionAwareIK(backend).solve(
            side="r",
            target=target,
            target_rotation=rotation,
            seed=0,
        )

        assert len(solution.joint_names) == 8
        assert solution.joint_values.shape == (8,)
        assert solution.error_m < 1e-9
        assert solution.orientation_error_deg < 1e-6


def test_default_ik_controller_starts_physical_task_with_closed_fingers() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        first_action = controller.action(0.0)

        assert controller.phase == "initialize_cnc"
        assert 1 <= controller.drawer_index <= 9
        assert first_action.shape == (len(ACTUATOR_NAMES),)
        assert first_action[ACTUATOR_NAMES.index("finger_l")] == 0.0
        assert first_action[ACTUATOR_NAMES.index("finger_r")] == 0.0
        assert "cube=(" in controller.status
        assert f"drawer={controller.drawer_index}" in controller.status
        assert not controller.done


def test_every_episode_opens_drawer_first_with_closest_hand() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        base = "base_link_base_cnc_x_joint"
        base_id = _joint_id(backend.model, base)
        base_qpos_id = int(backend.model.jnt_qposadr[base_id])
        midpoint = float(np.mean(backend.model.jnt_range[base_id]))
        actuator_id = backend.actuator_ids[ACTUATOR_NAMES.index("cnc_x")]
        lateral_id = _joint_id(
            backend.model, "cnc_x_link_cnc_x_cnc_y_joint"
        )
        vertical_id = _joint_id(
            backend.model, "cnc_y_link_cnc_y_head_joint"
        )

        for episode_seed in range(10):
            controller.reset(episode_seed=episode_seed)
            assert np.isclose(backend.data.qpos[base_qpos_id], midpoint, atol=1e-5)
            assert np.isclose(backend.data.ctrl[actuator_id], midpoint)
            assert np.isclose(controller.initial_targets[base], midpoint, atol=1e-5)
            handle, _ = drawer_handle_pose(
                backend.model, backend.data, controller.drawer_index
            )
            distances = {
                side: np.linalg.norm(hand_pose(backend.model, backend.data, side)[0] - handle)
                for side in ("l", "r")
            }
            assert controller.drawer_hand == min(distances, key=distances.__getitem__)

            controller.action(0.0)
            assert controller.motion is not None
            assert np.isclose(
                controller.motion.target[ACTUATOR_NAMES.index("cnc_x")],
                backend.model.jnt_range[base_id, 1],
            )
            assert np.isclose(
                controller.motion.target[ACTUATOR_NAMES.index("cnc_y")],
                np.mean(backend.model.jnt_range[lateral_id]),
            )
            assert np.isclose(
                controller.motion.target[ACTUATOR_NAMES.index("head_z")],
                np.mean(backend.model.jnt_range[vertical_id]),
            )
            for finger in ("finger_l", "finger_r"):
                assert controller.motion.target[ACTUATOR_NAMES.index(finger)] == FINGER_CLOSED
            assert controller.motion.after == "open_before_first_ik"

            controller.phase = "open_before_first_ik"
            controller.motion = None
            controller.action(0.0)
            assert controller.motion is not None
            assert np.isclose(
                controller.motion.target[ACTUATOR_NAMES.index("cnc_x")],
                backend.model.jnt_range[base_id, 1],
            )
            assert np.isclose(
                controller.motion.target[ACTUATOR_NAMES.index("cnc_y")],
                np.mean(backend.model.jnt_range[lateral_id]),
            )
            assert np.isclose(
                controller.motion.target[ACTUATOR_NAMES.index("head_z")],
                np.mean(backend.model.jnt_range[vertical_id]),
            )
            for finger in ("finger_l", "finger_r"):
                assert controller.motion.target[ACTUATOR_NAMES.index(finger)] == FINGER_OPEN
            assert controller.motion.after == "drawer_ik"


def test_seed_15_top_row_uses_selected_drawer_for_all_fixed_targets() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=15)
        controller.reset(episode_seed=0)
        assert controller.drawer_index == 2
        expected_open, _ = _drawer_hand_target_pose(backend.model, backend.data, 2)
        expected_close, _ = _drawer_close_target_pose(backend.model, backend.data, 2)
        assert np.allclose(controller.ik_targets["drawer_open"][0], expected_open)
        assert np.allclose(controller.ik_targets["drawer_close"][0], expected_close)


def test_same_cube_and_placement_hand_skips_handoff() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.cube_hand == controller.placement_hand

        drawer_id = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = (
            backend.model.jnt_range[drawer_id, 0]
        )
        mujoco.mj_forward(backend.model, backend.data)

        controller.phase = "drawer_hand_rest"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "view_cube"

        controller.phase = "lift_cube"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "cube_above_drawer"


def test_post_drawer_pose_restores_full_initial_view_with_open_fingers() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.drawer_hand is not None

        expected = controller._initial_view_targets()
        drawer_id = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = (
            backend.model.jnt_range[drawer_id, 0]
        )
        mujoco.mj_forward(backend.model, backend.data)

        controller.phase = "release_open_drawer"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "drawer_hand_rest"
        assert np.allclose(
            controller.motion.target[
                ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")
            ],
            FINGER_OPEN,
        )

        controller.phase = "drawer_hand_rest"
        controller.motion = None
        controller.action(0.0)

        assert controller.motion is not None
        for joint_name, value in expected.items():
            actuator_index = controller._actuator_index_for_joint(joint_name)
            assert np.isclose(controller.motion.target[actuator_index], value)
        assert controller.motion.after == "view_cube"
        for side in ("l", "r"):
            assert (
                controller.motion.target[ACTUATOR_NAMES.index(f"finger_{side}")]
                == FINGER_OPEN
            )

        controller.phase = "view_cube"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "cube_ik"


def test_different_cube_and_placement_hands_trigger_physical_handoff(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=0)
        assert controller.cube_hand == "r"
        assert controller.placement_hand == "l"

        controller.phase = "lift_cube"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_donor_ik"

        captured: dict[str, object] = {}

        def capture_solution(**kwargs):
            captured.update(kwargs)
            names = tuple(kwargs["joint_names"])
            return SimpleNamespace(
                error_m=0.0,
                orientation_error_deg=0.0,
                joint_names=names,
                joint_values=np.array(
                    [
                        backend.data.qpos[
                            backend.model.jnt_qposadr[_joint_id(backend.model, name)]
                        ]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", capture_solution)
        controller.phase = "handoff_donor_ik"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_receiver_ik"
        assert captured["side"] == "r"
        assert tuple(captured["joint_names"]) == _controlled_joints("r")[3:]
        assert controller.handoff_target is not None
        expected_midpoint = 0.5 * (
            hand_pose(backend.model, backend.data, "r")[0]
            + hand_pose(backend.model, backend.data, "l")[0]
        )
        assert np.allclose(controller.handoff_target, expected_midpoint)

        captured.clear()
        controller.phase = "handoff_receiver_ik"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_grasp"
        assert captured["side"] == "l"
        assert tuple(captured["joint_names"]) == _controlled_joints("l")[3:]

        controller.phase = "handoff_grasp"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_verify"

        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_near_hand",
            lambda *_args: (True, 0.0),
        )
        controller.phase = "handoff_verify"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_release"

        controller.phase = "handoff_release"
        controller.motion = None
        controller.action(0.0)
        assert controller.cube_hand == "l"
        assert controller.handoff_donor == "r"
        assert controller.motion is not None
        assert controller.motion.after == "handoff_donor_rest"

        controller.phase = "handoff_donor_rest"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "cube_above_drawer"
        for joint_name, value in controller._handoff_donor_rest_targets("r").items():
            actuator_index = controller._actuator_index_for_joint(joint_name)
            assert np.isclose(controller.motion.target[actuator_index], value)
        assert (
            controller.motion.target[ACTUATOR_NAMES.index("finger_r")]
            == FINGER_CLOSED
        )
        assert (
            controller.motion.target[ACTUATOR_NAMES.index("finger_l")]
            == CUBE_GRASP
        )
        raised_joint = SIDE_JOINTS["r"][2]
        assert np.isclose(
            controller.motion.target[
                controller._actuator_index_for_joint(raised_joint)
            ],
            -HANDOFF_DONOR_ARM_ELEVATION_RAD,
        )


def test_ik_controller_fails_early_when_handoff_receiver_misses_cube(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=0)
        assert controller.cube_hand == "r"
        assert controller.placement_hand == "l"
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_near_hand",
            lambda *_args: (False, 0.25),
        )
        controller.phase = "handoff_verify"
        controller.motion = None

        controller.action(0.0)

        assert controller.done
        assert not controller.successful
        assert controller.cube_hand == "r"
        assert "from the receiving hand" in controller.status


def test_drawer_opening_ik_uses_xyz_and_all_arm_joints(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        captured: dict[str, object] = {}

        def capture_solution(**kwargs):
            captured.update(kwargs)
            joint_names = _controlled_joints(kwargs["side"])
            joint_values = np.array(
                [
                    backend.data.qpos[
                        backend.model.jnt_qposadr[_joint_id(backend.model, name)]
                    ]
                    for name in joint_names
                ]
            )
            return SimpleNamespace(
                error_m=0.0,
                orientation_error_deg=0.0,
                joint_names=joint_names,
                joint_values=joint_values,
            )

        monkeypatch.setattr(controller.ik, "solve", capture_solution)
        controller.phase = "drawer_ik"
        controller.action(0.0)

        assert "joint_names" not in captured
        assert captured["finger_position"] == FINGER_OPEN
        base_id = _joint_id(backend.model, "base_link_base_cnc_x_joint")
        assert captured["joint_bounds"]["base_link_base_cnc_x_joint"] == (
            float(backend.model.jnt_range[base_id, 0])
            + DRAWER_APPROACH_DISTANCE_M,
            float(backend.model.jnt_range[base_id, 1]),
        )
        assert controller.drawer_hand is not None
        assert controller.motion is not None
        assert controller.motion.after == "approach_drawer"
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")]
            == FINGER_OPEN
        )
        optimized_joints = _controlled_joints(controller.drawer_hand)
        assert len(optimized_joints) == 8
        assert optimized_joints[:3] == (
            "base_link_base_cnc_x_joint",
            "cnc_x_link_cnc_x_cnc_y_joint",
            "cnc_y_link_cnc_y_head_joint",
        )


def test_cube_grasp_uses_vertical_cnc_for_descent_and_lift(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.cube_hand is not None
        captured: dict[str, object] = {}

        def current_joint_solution(**kwargs):
            captured.update(kwargs)
            names = _controlled_joints(kwargs["side"])
            return SimpleNamespace(
                error_m=1.0,
                orientation_error_deg=180.0,
                joint_names=names,
                joint_values=np.array(
                    [
                        backend.data.qpos[
                            backend.model.jnt_qposadr[_joint_id(backend.model, name)]
                        ]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", current_joint_solution)
        controller.phase = "cube_ik"
        controller.action(0.0)
        assert not controller.done
        assert controller.motion is not None
        assert controller.motion.after == "descend_to_cube"
        vertical = "cnc_y_link_cnc_y_head_joint"
        vertical_id = _joint_id(backend.model, vertical)
        assert captured["joint_bounds"][vertical] == (
            float(backend.model.jnt_range[vertical_id, 0]),
            float(backend.model.jnt_range[vertical_id, 1])
            - CUBE_GRASP_APPROACH_HEIGHT_M,
        )
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.cube_hand}")]
            == FINGER_OPEN
        )

        controller.phase = "descend_to_cube"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        changed = np.flatnonzero(
            ~np.isclose(controller.motion.target, controller.motion.start)
        )
        assert set(changed) <= {
            ACTUATOR_NAMES.index("head_z"),
            ACTUATOR_NAMES.index(f"finger_{controller.cube_hand}"),
        }
        approach_qpos = controller.cube_approach_vertical_qpos
        assert approach_qpos is not None

        controller.phase = "cube_clearance"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "return_without_base"
        assert np.isclose(
            controller.motion.target[ACTUATOR_NAMES.index("head_z")],
            approach_qpos,
        )


def test_drawer_contact_and_retreat_use_only_world_y_cnc() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.drawer_hand is not None

        for phase, after in (("approach_drawer", "grasp_drawer"),):
            controller.phase = phase
            controller.motion = None
            controller.action(0.0)
            assert controller.motion is not None
            assert controller.motion.after == after
            changed_nonfinger = {
                int(index)
                for index in np.flatnonzero(
                    ~np.isclose(controller.motion.target, controller.motion.start)
                )
                if index
                not in {
                    ACTUATOR_NAMES.index("finger_l"),
                    ACTUATOR_NAMES.index("finger_r"),
                }
            }
            assert changed_nonfinger <= {ACTUATOR_NAMES.index("cnc_x")}


def test_drawer_opening_closes_finger_to_tighter_grasp() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.drawer_hand is not None
        controller.phase = "grasp_drawer"
        controller.motion = None

        controller.action(0.0)

        assert controller.motion is not None
        assert controller.motion.after == "pull_drawer"
        assert np.isclose(
            controller.motion.target[
                ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")
            ],
            DRAWER_GRASP,
        )


def test_seed_16_pull_parks_idle_arm_and_fully_opens_drawer() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=16)
        controller.reset(episode_seed=0)
        assert controller.drawer_index == 5
        assert controller.drawer_hand == "r"

        for frame in range(600):
            backend.send_action(controller.action(frame / 599))
            backend.step()
            if controller.phase == "release_open_drawer":
                break

        assert controller.phase == "release_open_drawer"
        drawer_id = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        drawer_position = float(
            backend.data.qpos[backend.model.jnt_qposadr[drawer_id]]
        )
        assert drawer_position <= backend.model.jnt_range[drawer_id, 0] + 0.002
        assert controller.motion is not None
        idle_forearm = SIDE_JOINTS["l"][1]
        idle_forearm_index = controller._actuator_index_for_joint(idle_forearm)
        assert np.isclose(
            controller.motion.target[idle_forearm_index],
            IDLE_FOREARM_DRAWER_PULL_POSITION,
        )


def test_drawer_closing_ik_targets_standoff_and_uses_cnc_round_trip(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.drawer_hand is not None
        captured: dict[str, object] = {}

        def current_joint_solution(**kwargs):
            captured.update(kwargs)
            names = _controlled_joints(kwargs["side"])
            return SimpleNamespace(
                error_m=0.0,
                orientation_error_deg=0.0,
                joint_names=names,
                joint_values=np.array(
                    [
                        backend.data.qpos[
                            backend.model.jnt_qposadr[_joint_id(backend.model, name)]
                        ]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", current_joint_solution)
        controller.phase = "close_drawer_ik"
        controller.action(0.0)

        base = "base_link_base_cnc_x_joint"
        base_id = _joint_id(backend.model, base)
        drawer_id = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        expected_low = (
            float(backend.model.jnt_range[base_id, 0])
            + DRAWER_CLOSE_APPROACH_DISTANCE_M
            + float(np.ptp(backend.model.jnt_range[drawer_id]))
            + DRAWER_PUSH_MARGIN_M
        )
        assert captured["joint_bounds"][base] == (
            expected_low,
            float(backend.model.jnt_range[base_id, 1]),
        )
        assert captured["finger_position"] == FINGER_CLOSED
        assert "allow_drawer_contact" not in captured
        assert controller.motion is not None
        assert controller.motion.after == "push_drawer"
        assert (
            controller.motion.target[
                ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")
            ]
            == FINGER_CLOSED
        )

        cnc_start = float(backend.data.qpos[backend.model.jnt_qposadr[base_id]])
        drawer_opening_delta = float(
            backend.data.qpos[backend.model.jnt_qposadr[drawer_id]]
            - backend.model.jnt_range[drawer_id, 1]
        )
        controller.phase = "push_drawer"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "release_closed_drawer"
        assert controller.drawer_close_cnc_start == cnc_start
        expected_push = np.clip(
            cnc_start
            - DRAWER_CLOSE_APPROACH_DISTANCE_M
            + drawer_opening_delta
            - DRAWER_PUSH_MARGIN_M,
            *backend.model.jnt_range[base_id],
        )
        assert np.isclose(
            controller.motion.target[ACTUATOR_NAMES.index("cnc_x")], expected_push
        )
        assert (
            controller.motion.target[
                ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")
            ]
            == FINGER_CLOSED
        )

        controller.phase = "release_closed_drawer"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "final_return"
        assert np.isclose(
            controller.motion.target[ACTUATOR_NAMES.index("cnc_x")], cnc_start
        )


def test_debug_shows_live_pregrasp_targets() -> None:
    with MujocoBackend(SimulationConfig(render=False, debug=True, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)

        assert tuple(controller.ik_targets) == (
            "cube_grasp",
            "drawer_open",
            "cube_place",
            "drawer_close",
        )
        cube_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "cube_link_collision_box_01_geom",
        )
        assert np.allclose(
            controller.ik_targets["cube_grasp"][0],
            backend.data.geom_xpos[cube_id]
            + np.array([0.0, 0.0, CUBE_GRASP_APPROACH_HEIGHT_M]),
        )
        for marker_name, (target, rotation) in zip(
            IK_TARGET_MARKERS, controller.ik_targets.values(), strict=True
        ):
            marker_id = mujoco.mj_name2id(
                backend.model, mujoco.mjtObj.mjOBJ_BODY, marker_name
            )
            assert marker_id >= 0
            assert np.allclose(backend.data.xpos[marker_id], target)
            assert np.allclose(
                backend.data.xmat[marker_id].reshape(3, 3), rotation
            )

        expected_position, expected_rotation = _drawer_hand_target_pose(
            backend.model, backend.data, controller.drawer_index
        )
        assert np.allclose(
            controller.ik_targets["drawer_open"][0], expected_position
        )
        assert np.allclose(
            controller.ik_targets["drawer_open"][1], expected_rotation
        )
        expected_close_position, _ = _drawer_close_target_pose(
            backend.model, backend.data, controller.drawer_index
        )
        handle_center, _ = drawer_handle_pose(
            backend.model, backend.data, controller.drawer_index
        )
        assert np.allclose(
            controller.ik_targets["drawer_close"][0], expected_close_position
        )
        assert np.allclose(
            controller.ik_targets["drawer_close"][1], expected_rotation
        )
        drawer_joint = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        inward_axis = backend.data.xaxis[drawer_joint]
        assert np.allclose(
            expected_close_position,
            handle_center - inward_axis * DRAWER_CLOSE_APPROACH_DISTANCE_M,
        )
        assert np.allclose(
            expected_position,
            handle_center
            - inward_axis * DRAWER_APPROACH_DISTANCE_M
            + np.array([0.0, 0.0, DRAWER_TARGET_Z_OFFSET_M]),
        )

        cube_marker_id = mujoco.mj_name2id(
            backend.model, mujoco.mjtObj.mjOBJ_BODY, "cube_frame_marker"
        )
        assert np.allclose(backend.data.xpos[cube_marker_id], (0.0, 0.0, -10.0))


def test_cube_placement_target_uses_extra_top_row_clearance() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        for drawer_index in (2, 5, 8):
            handle_center, _ = drawer_handle_pose(
                backend.model, backend.data, drawer_index
            )
            target, _ = _cube_above_drawer_pose(
                backend.model, backend.data, drawer_index
            )

            assert np.isclose(
                target[1] - handle_center[1], CUBE_PLACE_HANDLE_Y_OFFSET_M
            )
            expected_height = CUBE_PLACE_HANDLE_Z_OFFSET_M + (
                TOP_ROW_CUBE_PLACE_CLEARANCE_M if drawer_index <= 3 else 0.0
            )
            assert np.isclose(target[2] - handle_center[2], expected_height)


def test_cube_placement_ik_directly_aligns_hand_with_target_frame(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.cube_hand is not None
        captured: dict[str, object] = {}

        def current_joint_solution(**kwargs):
            captured.update(kwargs)
            names = _controlled_joints(kwargs["side"])
            return SimpleNamespace(
                error_m=0.0,
                orientation_error_deg=0.0,
                joint_names=names,
                joint_values=np.array(
                    [
                        backend.data.qpos[
                            backend.model.jnt_qposadr[_joint_id(backend.model, name)]
                        ]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", current_joint_solution)
        target_position, target_rotation = controller.ik_targets["cube_place"]
        controller.phase = "cube_above_drawer"
        controller.motion = None
        controller.action(0.0)

        assert np.allclose(captured["target"], target_position)
        assert np.allclose(captured["target_rotation"], target_rotation)
        assert captured["align_blue_axis"] is True
        assert captured["directed_axes"] is True
        assert captured["finger_position"] == CUBE_GRASP
        assert np.allclose(target_rotation[:, 2], (0.0, 0.0, 1.0))
        assert np.isclose(target_rotation[2, 0], 0.0)
        assert controller.motion is not None
        assert (
            controller.motion.target[
                ACTUATOR_NAMES.index(f"finger_{controller.cube_hand}")
            ]
            == CUBE_GRASP
        )


def test_middle_drawer_closing_randomly_uses_both_hands_reproducibly() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        choices = []
        for episode_seed in range(12):
            controller = MujocoIKController(backend, seed=0)
            controller.reset(episode_seed=episode_seed)
            controller.drawer_index = 5
            controller._refresh_drawer_targets()
            controller.phase = "choose_closing_hand"
            controller.motion = None
            controller.action(0.0)
            choices.append(controller.drawer_hand)
            assert controller.motion is not None
            assert controller.motion.after == "close_drawer_ik"

            repeated = MujocoIKController(backend, seed=0)
            repeated.reset(episode_seed=episode_seed)
            repeated.drawer_index = 5
            repeated._refresh_drawer_targets()
            repeated.phase = "choose_closing_hand"
            repeated.motion = None
            repeated.action(0.0)
            assert repeated.drawer_hand == controller.drawer_hand

        assert set(choices) == {"l", "r"}


def test_drawer_targets_follow_handle_while_other_targets_stay_fixed() -> None:
    with MujocoBackend(SimulationConfig(render=False, debug=True, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        fixed_marker_ids = [
            mujoco.mj_name2id(
                backend.model, mujoco.mjtObj.mjOBJ_BODY, IK_TARGET_MARKERS[index]
            )
            for index in (0, 2)
        ]
        drawer_marker_ids = [
            mujoco.mj_name2id(
                backend.model, mujoco.mjtObj.mjOBJ_BODY, IK_TARGET_MARKERS[index]
            )
            for index in (1, 3)
        ]
        fixed_positions = backend.data.xpos[fixed_marker_ids].copy()
        drawer_position = controller.ik_targets["drawer_open"][0].copy()

        drawer_joint = f"base_link_base_drawer_{controller.drawer_index}_joint"
        drawer_id = _joint_id(backend.model, drawer_joint)
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = (
            backend.model.jnt_range[drawer_id, 0]
        )
        mujoco.mj_forward(backend.model, backend.data)
        expected_open_position, expected_rotation = _drawer_hand_target_pose(
            backend.model, backend.data, controller.drawer_index
        )
        expected_close_position, _ = _drawer_close_target_pose(
            backend.model, backend.data, controller.drawer_index
        )

        controller.action(0.0)

        assert not np.allclose(expected_open_position, drawer_position)
        assert np.allclose(
            controller.ik_targets["drawer_open"][0], expected_open_position
        )
        assert np.allclose(
            controller.ik_targets["drawer_close"][0], expected_close_position
        )
        for target_name in ("drawer_open", "drawer_close"):
            assert np.allclose(controller.ik_targets[target_name][1], expected_rotation)
        assert np.allclose(
            backend.data.xpos[drawer_marker_ids[0]], expected_open_position
        )
        assert np.allclose(
            backend.data.xpos[drawer_marker_ids[1]], expected_close_position
        )
        assert np.allclose(backend.data.xpos[fixed_marker_ids], fixed_positions)


def test_initial_cube_placement_target_stays_fixed_after_drawer_opening() -> None:
    with MujocoBackend(SimulationConfig(render=False, debug=True, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        initial_position = controller.ik_targets["cube_place"][0].copy()
        initial_rotation = controller.ik_targets["cube_place"][1].copy()
        marker_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "above_drawer_target_marker",
        )

        drawer_joint = f"base_link_base_drawer_{controller.drawer_index}_joint"
        drawer_id = _joint_id(backend.model, drawer_joint)
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = (
            backend.model.jnt_range[drawer_id, 0]
        )
        mujoco.mj_forward(backend.model, backend.data)
        controller._select_post_drawer_hands_and_targets()

        assert np.allclose(controller.ik_targets["cube_place"][0], initial_position)
        assert np.allclose(controller.ik_targets["cube_place"][1], initial_rotation)
        assert np.allclose(backend.data.xpos[marker_id], initial_position)
        assert np.allclose(
            backend.data.xmat[marker_id].reshape(3, 3), initial_rotation
        )


def test_mujoco_exposes_only_the_physical_ik_controller() -> None:
    assert available_controllers("mujoco") == ("ik",)


def test_ik_controller_fails_early_when_drawer_did_not_open() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        controller.drawer_hand = "l"
        controller.phase = "release_open_drawer"

        controller.action(0.0)

        assert controller.done
        assert not controller.successful
        assert "early failure after drawer opening" in controller.status


def test_ik_controller_fails_early_when_cube_was_not_grasped() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        controller.cube_hand = "r"
        controller.phase = "return_without_base"

        controller.action(0.0)

        assert controller.done
        assert not controller.successful
        assert "early failure after cube grasp" in controller.status


def test_ik_controller_fails_early_when_cube_missed_drawer() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        controller.cube_hand = "r"
        controller.phase = "cube_hand_rest"

        controller.action(0.0)

        assert controller.done
        assert not controller.successful
        assert "early failure after cube release" in controller.status


def test_ik_controller_fails_early_when_drawer_remains_open() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4)
        controller.reset(episode_seed=7)
        controller.drawer_hand = "l"
        drawer_joint = f"base_link_base_drawer_{controller.drawer_index}_joint"
        drawer_id = _joint_id(backend.model, drawer_joint)
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = (
            backend.model.jnt_range[drawer_id, 1]
            - MAX_DRAWER_CLOSED_OPENING_M
            - MIN_DRAWER_OPENING_M
        )
        mujoco.mj_forward(backend.model, backend.data)
        controller.phase = "release_closed_drawer"

        controller.action(0.0)

        assert controller.done
        assert not controller.successful
        assert "early failure after drawer closing" in controller.status
