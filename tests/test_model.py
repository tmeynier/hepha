from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
from hepha_lerobot.recording.controllers import available_controllers
from hepha_lerobot.workspaces import Workspace

from simulation import SimulationConfig, available_backends, create_backend
from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
from simulation.backends.mujoco.episode import WORKING_ARM_POSE_RADIANS
from simulation.backends.mujoco.ik import (
    CONTACT_MARGIN_M,
    CUBE_DRAWER_SETTLE_DURATION_S,
    CUBE_DROP_ASSIST_FALL_SPEED_M_S,
    CUBE_GRASP,
    CUBE_GRASP_APPROACH_HEIGHT_M,
    CUBE_GRASP_IK_ORIENTATION_WEIGHT,
    CUBE_GRASP_IK_POSITION_WEIGHT,
    CUBE_GRASP_LOCK_EQUALITIES,
    CUBE_GRASP_SETTLE_DURATION_S,
    CUBE_GRIPPER_CONTACT_PAIRS,
    CUBE_GRIPPER_GRASP_FRICTION,
    CUBE_GRIPPER_RELEASE_FRICTION,
    CUBE_PLACE_HANDLE_Y_OFFSET_M,
    CUBE_PLACE_HANDLE_Z_OFFSET_M,
    CUBE_PLACE_RETREAT_DISTANCE_M,
    CUBE_SPAWN_RADIUS_M,
    DEFAULT_IK_MAX_POSTURE_DEVIATION_RADIANS,
    DEFAULT_IK_POSTURE_WEIGHT,
    DEFAULT_IK_PREVIOUS_POSTURE_WEIGHT,
    DRAWER_APPROACH_DISTANCE_M,
    DRAWER_CLOSE_APPROACH_DISTANCE_M,
    DRAWER_CLOSE_RETREAT_VERTICAL_DISTANCE_M,
    DRAWER_FINGER_OPEN,
    DRAWER_GRASP,
    DRAWER_IK_ORIENTATION_WEIGHT,
    DRAWER_OPEN_IK_POSITION_WEIGHT,
    DRAWER_OPEN_RETREAT_EXTRA_DISTANCE_M,
    DRAWER_PULL_OVERSHOOT_M,
    DRAWER_RETREAT_IK_POSITION_WEIGHT,
    DRAWER_TARGET_Z_OFFSET_M,
    FINGER_CLOSED,
    FINGER_OPEN,
    HAND_TASK_FRAME_INSET_M,
    HANDOFF_CONTACT_ABORT_FRAMES,
    HANDOFF_DONOR_ARM_ELEVATION_RAD,
    HANDOFF_DONOR_RETREAT_DISTANCE_M,
    HANDOFF_HAND_FRAME_INSET_M,
    HANDOFF_OVERLAP_HOLD_DURATION_SCALE,
    HANDOFF_RECEIVER_APPROACH_DISTANCE_M,
    HANDOFF_RECEIVER_APPROACH_POSITION_WEIGHT,
    HANDOFF_RECEIVER_APPROACH_RED_ORIENTATION_WEIGHT,
    HANDOFF_RECEIVER_GRASP,
    HANDOFF_RECEIVER_TRANSLATION_BLUE_ORIENTATION_WEIGHT,
    HANDOFF_RECEIVER_TRANSLATION_CONTINUITY_WEIGHT,
    HANDOFF_RECEIVER_TRANSLATION_POSITION_WEIGHT,
    HANDOFF_RECEIVER_TRANSLATION_RED_ORIENTATION_WEIGHT,
    HANDOFF_RECEIVER_VERTICAL_OFFSET_M,
    HANDOFF_RECEIVER_WIDE_OPEN,
    IK_TARGET_MARKERS,
    MAX_DRAWER_CLOSED_OPENING_M,
    MIN_DRAWER_OPENING_M,
    SIDE_JOINTS,
    STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
    TOP_ROW_CUBE_PLACE_CLEARANCE_M,
    MujocoIKController,
    PositionOrientationIK,
    _closed_drawer_vertical_retreat,
    _controlled_joints,
    _cube_above_drawer_pose,
    _drawer_close_target_pose,
    _drawer_hand_target_pose,
    _fixed_finger_tip,
    _joint_id,
    _joint_qpos,
    _moving_finger_tip,
    _randomize_cube,
    _set_cube_grasp_lock,
    _world_point,
    drawer_handle_pose,
    grasp_pose,
    hand_pose,
)
from simulation.backends.mujoco.workspaces import (
    resolve_mujoco_workspace,
    set_workspace_qpos,
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
            radii.append(np.linalg.norm(backend.data.qpos[qpos_id : qpos_id + 2] - center))

        assert max(radii) <= CUBE_SPAWN_RADIUS_M
        assert min(radii) < 0.02
        assert max(radii) > 0.09


def test_hand_task_frame_is_inset_from_fingertips_toward_palm() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        for side in ("l", "r"):
            fixed_body, fixed_tip, _ = _fixed_finger_tip(backend.model, side)
            moving_body, moving_tip = _moving_finger_tip(backend.model, side)
            fingertip_midpoint = 0.5 * (
                _world_point(backend.data, fixed_body, fixed_tip)
                + _world_point(backend.data, moving_body, moving_tip)
            )

            task_position, task_rotation = hand_pose(backend.model, backend.data, side)

            assert np.allclose(
                task_position - fingertip_midpoint,
                HAND_TASK_FRAME_INSET_M * task_rotation[:, 1],
            )


def test_mujoco_model_matches_control_contract() -> None:
    assert "mujoco" in available_backends()
    with create_backend("mujoco", SimulationConfig(render=False)) as backend:
        assert isinstance(backend, MujocoBackend)
        assert backend.model.nu == len(ACTUATOR_NAMES)
        assert backend.joint_positions().shape == (len(ACTUATOR_NAMES),)
        assert tuple(backend.action_features) == tuple(f"{name}.pos" for name in ACTUATOR_NAMES)
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
        assert backend.model.geom_priority[cube_id] == 1
        assert backend.model.geom_condim[cube_id] == 3
        assert np.allclose(
            backend.model.geom_friction[cube_id],
            (0.35, 0.002, 0.0001),
        )
        assert backend.model.nflex == 0
        cube_body_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, "cube_link")
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

        floor_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert backend.model.geom_conaffinity[floor_id] & 1


def test_cube_gripper_contacts_are_moderate_and_compliant() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        for side in ("l", "r"):
            for pair_name in (
                f"cube_hand_{side}_pad",
                f"cube_finger_{side}_shaft",
                f"cube_finger_{side}_tip",
            ):
                pair_id = mujoco.mj_name2id(
                    backend.model,
                    mujoco.mjtObj.mjOBJ_PAIR,
                    pair_name,
                )
                assert pair_id >= 0
                assert backend.model.pair_dim[pair_id] == 4
                assert np.allclose(
                    backend.model.pair_friction[pair_id],
                    (1.5, 1.5, 0.03, 0.001, 0.001),
                )
                assert np.allclose(
                    backend.model.pair_solref[pair_id],
                    (0.015, 1.0),
                )
                assert np.allclose(
                    backend.model.pair_solimp[pair_id, :3],
                    (0.90, 0.98, 0.002),
                )


def test_closed_drawers_are_aligned_to_the_rack_grid() -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        model = backend.model
        data = backend.data

        def geom_id(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

        shelf_ids = [
            geom_id(f"rack_structure_link_collision_box_0{index}_geom") for index in (2, 3, 4)
        ]
        divider_ids = [
            geom_id(f"rack_structure_link_collision_box_0{index}_geom") for index in (7, 5, 6, 8)
        ]
        divider_x = data.geom_xpos[divider_ids, 0]
        column_x = (divider_x[:-1] + divider_x[1:]) / 2
        rack_depth = data.geom_xpos[shelf_ids[0], 1]

        for drawer_index in range(1, 10):
            row = (drawer_index - 1) // 3
            column = 2 - ((drawer_index - 1) % 3)
            drawer_id = geom_id(f"drawer_{drawer_index}_link_collision_box_01_geom")
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


def test_position_orientation_ik_uses_the_arm_only_chain(monkeypatch) -> None:
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
        solution = PositionOrientationIK(backend).solve(
            side="r",
            target=target,
            target_rotation=rotation,
            seed=0,
        )

        assert len(solution.joint_names) == 5
        assert solution.joint_values.shape == (5,)
        assert np.isfinite(solution.error_m)
        assert np.isfinite(solution.orientation_error_deg)


def test_ik_softly_prefers_the_fixed_nominal_posture(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False)) as backend:
        names = _controlled_joints("l")
        reference = {
            name: WORKING_ARM_POSE_RADIANS[actuator]
            for name, actuator in zip(
                names,
                ("shoulder_l", "forearm_l", "arm_l", "wrist_l", "hand_l"),
                strict=True,
            )
        }
        reference_vector = np.array([reference[name] for name in names])

        def choose_reference(objective, bounds, **_kwargs):
            for name, (low, high) in zip(names, bounds, strict=True):
                joint_id = _joint_id(backend.model, name)
                model_low, model_high = backend.model.jnt_range[joint_id]
                assert np.isclose(
                    low,
                    max(
                        model_low,
                        reference[name] - DEFAULT_IK_MAX_POSTURE_DEVIATION_RADIANS,
                    ),
                )
                assert np.isclose(
                    high,
                    min(
                        model_high,
                        reference[name] + DEFAULT_IK_MAX_POSTURE_DEVIATION_RADIANS,
                    ),
                )
            middle = np.mean(np.asarray(bounds), axis=1)
            assert objective(reference_vector) < objective(middle)
            return SimpleNamespace(x=reference_vector)

        monkeypatch.setattr(
            "simulation.backends.mujoco.ik.differential_evolution",
            choose_reference,
        )
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik.minimize",
            lambda _objective, initial, **_kwargs: SimpleNamespace(x=initial),
        )
        target, rotation = hand_pose(backend.model, backend.data, "l")
        solution = PositionOrientationIK(backend).solve(
            side="l",
            target=target,
            target_rotation=rotation,
            seed=0,
            position_weight=0.0,
            orientation_weight=0.0,
            include_collision_penalty=False,
            posture_reference=reference,
        )

        assert np.allclose(solution.joint_values, reference_vector)


def test_default_ik_controller_starts_physical_task_with_closed_fingers() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=7)
        first_action = controller.action(0.0)

        assert controller.phase == "start_at_a"
        assert 1 <= controller.drawer_index <= 9
        assert first_action.shape == (len(ACTUATOR_NAMES),)
        assert first_action[ACTUATOR_NAMES.index("finger_l")] == 0.0
        assert first_action[ACTUATOR_NAMES.index("finger_r")] == 0.0
        assert "cube=(" in controller.status
        assert f"drawer={controller.drawer_index}" in controller.status
        assert not controller.done


def test_ik_episode_starts_with_mirrored_bent_working_arms() -> None:
    assert WORKING_ARM_POSE_RADIANS == {
        "shoulder_l": 0.220,
        "forearm_l": 0.346,
        "arm_l": -0.550,
        "wrist_l": -0.518,
        "hand_l": -1.570,
        "shoulder_r": -0.220,
        "forearm_r": 0.346,
        "arm_r": -0.550,
        "wrist_r": 0.518,
        "hand_r": 1.570,
    }
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=0)

        for actuator_name, angle_radians in WORKING_ARM_POSE_RADIANS.items():
            action_index = ACTUATOR_NAMES.index(actuator_name)
            actuator_id = backend.actuator_ids[action_index]
            joint_id = backend.model.actuator_trnid[actuator_id, 0]
            assert np.isclose(
                backend.data.qpos[backend.model.jnt_qposadr[joint_id]],
                angle_radians,
            )


def test_every_episode_opens_drawer_first_with_closest_hand() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)

        for episode_seed in range(10):
            controller.reset(episode_seed=episode_seed)
            workspace_a = resolve_mujoco_workspace(backend.model, Workspace.STORAGE)
            for joint_name, value in workspace_a.items():
                assert np.isclose(backend.data.joint(joint_name).qpos[0], value)

            planning_data = mujoco.MjData(backend.model)
            planning_data.qpos[:] = backend.data.qpos
            set_workspace_qpos(
                backend.model,
                planning_data,
                Workspace.DRAWERS,
                drawer=controller.drawer_index,
            )
            mujoco.mj_forward(backend.model, planning_data)
            handle, _ = drawer_handle_pose(backend.model, planning_data, controller.drawer_index)
            distances = {
                side: np.linalg.norm(hand_pose(backend.model, planning_data, side)[0] - handle)
                for side in ("l", "r")
            }
            assert controller.drawer_hand == min(distances, key=distances.__getitem__)

            controller.action(0.0)
            assert controller.motion is not None
            for finger in ("finger_l", "finger_r"):
                assert controller.motion.target[ACTUATOR_NAMES.index(finger)] == FINGER_CLOSED
            assert controller.motion.after == "move_to_b_for_open"

            controller.phase = "move_to_b_for_open"
            controller.motion = None
            controller.action(0.0)
            assert controller.motion is not None
            assert controller.recording_workspace == "B"
            assert controller.motion.after == "enter_b_for_open"


def test_seed_15_top_row_uses_selected_drawer_for_all_fixed_targets() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=15, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=0)
        assert controller.drawer_index == 2
        expected_open, _ = _drawer_hand_target_pose(
            backend.model, backend.data, 2, controller.drawer_hand
        )
        expected_close, _ = _drawer_close_target_pose(
            backend.model, backend.data, 2, controller.drawer_hand
        )
        assert np.allclose(controller.ik_targets["drawer_open"][0], expected_open)
        assert np.allclose(controller.ik_targets["drawer_close"][0], expected_close)


def test_drawer_grasp_orientation_is_perpendicular_to_handle() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        drawer_index = 5
        _, drawer_rotation = drawer_handle_pose(backend.model, backend.data, drawer_index)

        for side, direction in (("l", 1.0), ("r", -1.0)):
            _, hand_rotation = _drawer_hand_target_pose(
                backend.model,
                backend.data,
                drawer_index,
                side,
            )
            relative_rotation = drawer_rotation.T @ hand_rotation
            angle_degrees = np.degrees(
                np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
            )
            assert np.isclose(angle_degrees, 90.0)
            assert np.allclose(
                hand_rotation[:, 0],
                direction * drawer_rotation[:, 2],
            )


def test_same_cube_and_placement_hand_skips_handoff() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
        for episode_seed in range(50):
            controller.reset(episode_seed=episode_seed)
            if controller.cube_hand == controller.placement_hand:
                break
        else:
            raise AssertionError("Expected at least one no-handoff episode seed")

        drawer_id = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = backend.model.jnt_range[
            drawer_id, 0
        ]
        mujoco.mj_forward(backend.model, backend.data)

        controller.cube_initial_z = -1.0
        controller.phase = "return_without_base"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "move_to_b_for_place"


def test_center_drawers_always_use_cube_hand_for_placement() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(
            backend,
            seed=0,
            trajectory_randomization_scale=0.0,
        )
        controller.reset(episode_seed=0)

        for drawer_index in (2, 5, 8):
            controller.drawer_index = drawer_index
            controller._initialize_ik_targets()
            assert controller.placement_hand == controller.cube_hand

            controller._select_post_drawer_hands_and_targets()
            assert controller.placement_hand == controller.cube_hand


def test_post_drawer_retreat_continues_without_a_rest_pose(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        assert controller.drawer_hand is not None

        drawer_id = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = backend.model.jnt_range[
            drawer_id, 0
        ]
        mujoco.mj_forward(backend.model, backend.data)

        controller.phase = "release_open_drawer"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "retreat_open_drawer"
        assert np.allclose(
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")],
            DRAWER_FINGER_OPEN,
        )

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
        controller.phase = "retreat_open_drawer"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "move_to_a_for_pick"
        expected_retreat = (
            controller.ik_targets["drawer_open"][0]
            + DRAWER_OPEN_RETREAT_EXTRA_DISTANCE_M * controller._drawer_robotward_axis()
        )
        assert np.allclose(captured["target"], expected_retreat)

        controller.phase = "move_to_a_for_pick"
        controller.motion = None
        controller.action(0.0)

        assert controller.motion is not None
        assert controller.motion.after == "align_a_for_pick"
        assert controller.recording_workspace == "A"
        for side in ("l", "r"):
            assert controller.motion.target[ACTUATOR_NAMES.index(f"finger_{side}")] == FINGER_OPEN

        controller.phase = "align_a_for_pick"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "cube_ik"


def test_different_cube_and_placement_hands_trigger_physical_handoff(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=17)
        assert controller.cube_hand == "r"
        assert controller.placement_hand == "l"

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
                        backend.data.qpos[backend.model.jnt_qposadr[_joint_id(backend.model, name)]]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", capture_solution)
        controller.cube_initial_z = -1.0
        controller.phase = "return_without_base"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_receiver_ik"

        expected_workspace_center = controller._handoff_workspace_center()
        shoulder_id = _joint_id(backend.model, SIDE_JOINTS["r"][0])
        shoulder_qpos = int(backend.model.jnt_qposadr[shoulder_id])
        backend.data.qpos[shoulder_qpos] += 0.2
        mujoco.mj_forward(backend.model, backend.data)
        live_hand_midpoint = 0.5 * (
            hand_pose(
                backend.model,
                backend.data,
                "r",
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )[0]
            + hand_pose(
                backend.model,
                backend.data,
                "l",
                task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
            )[0]
        )
        assert not np.allclose(live_hand_midpoint, expected_workspace_center)

        controller.phase = "handoff_donor_ik"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_receiver_ik"
        assert captured["side"] == "r"
        assert tuple(captured["joint_names"]) == SIDE_JOINTS["r"]
        assert controller.handoff_target is not None
        assert np.allclose(controller.handoff_target, expected_workspace_center)
        assert captured["hand_frame_inset_m"] == HANDOFF_HAND_FRAME_INSET_M

        captured.clear()
        controller.phase = "handoff_receiver_ik"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_receiver_contact_ik"
        assert captured["side"] == "l"
        assert tuple(captured["joint_names"]) == SIDE_JOINTS["l"]
        assert captured["align_red_axis"] is True
        assert captured["align_blue_axis"] is True
        assert captured["directed_axes"] is True
        assert captured["directed_red_axis"] is False
        assert captured["protect_other_hand"] is True
        assert captured["finger_position"] == HANDOFF_RECEIVER_WIDE_OPEN
        assert captured["hand_frame_inset_m"] == HANDOFF_HAND_FRAME_INSET_M
        assert captured["position_weight"] == HANDOFF_RECEIVER_APPROACH_POSITION_WEIGHT
        assert (
            captured["red_orientation_weight"]
            == HANDOFF_RECEIVER_APPROACH_RED_ORIENTATION_WEIGHT
        )
        plan = controller.handoff_receiver_plan
        assert plan is not None
        cube_rotation = backend.data.geom(
            "cube_link_collision_box_01_geom"
        ).xmat.reshape(3, 3)
        donor_rotation = hand_pose(
            backend.model,
            backend.data,
            "r",
            task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
        )[1]
        expected_donor_axis = max(
            (0, 1),
            key=lambda axis: abs(
                float(donor_rotation[:, 0] @ cube_rotation[:, axis])
            ),
        )
        assert plan.donor_face_axis == expected_donor_axis
        assert plan.receiver_face_axis == 1 - expected_donor_axis
        assert np.isclose(
            abs(
                cube_rotation[:, plan.receiver_face_axis]
                @ plan.receiver_face
            ),
            1.0,
        )
        assert np.isclose(
            cube_rotation[:, plan.donor_face_axis] @ plan.receiver_face,
            0.0,
            atol=1e-9,
        )
        assert np.isclose(
            abs(plan.contact_rotation[:, 0] @ cube_rotation[:, 2]),
            1.0,
        )
        assert np.isclose(plan.contact_rotation[:, 2] @ plan.receiver_face, 1.0)
        assert np.allclose(plan.approach_rotation, plan.contact_rotation)
        assert np.allclose(captured["target_rotation"], plan.approach_rotation)
        assert np.isclose(plan.approach_direction[2], 0.0)
        cube_center = backend.data.geom("cube_link_collision_box_01_geom").xpos.copy()
        assert np.allclose(
            plan.contact_target - cube_center,
            HANDOFF_RECEIVER_VERTICAL_OFFSET_M * np.array([0.0, 0.0, 1.0]),
        )
        assert np.isclose(
            np.linalg.norm(plan.approach_target - plan.contact_target),
            HANDOFF_RECEIVER_APPROACH_DISTANCE_M,
        )
        assert np.allclose(
            plan.approach_target - plan.contact_target,
            HANDOFF_RECEIVER_APPROACH_DISTANCE_M * plan.approach_direction,
        )
        donor_hold = controller.handoff_donor_hold_targets
        assert donor_hold is not None
        for joint_name, value in donor_hold.items():
            assert np.isclose(
                controller.motion.target[controller._actuator_index_for_joint(joint_name)],
                value,
            )
        assert (
            controller.motion.target[ACTUATOR_NAMES.index("finger_l")]
            == HANDOFF_RECEIVER_WIDE_OPEN
        )

        captured.clear()
        controller.phase = "handoff_receiver_contact_ik"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_grasp"
        assert np.allclose(captured["target"], plan.contact_target)
        assert np.allclose(captured["target_rotation"], plan.contact_rotation)
        assert captured["align_red_axis"] is True
        assert captured["align_blue_axis"] is True
        assert captured["directed_red_axis"] is False
        assert captured["protect_other_hand"] is True
        assert captured["finger_position"] == HANDOFF_RECEIVER_WIDE_OPEN
        assert captured["hand_frame_inset_m"] == HANDOFF_HAND_FRAME_INSET_M
        assert (
            captured["posture_reference"]
            == controller.handoff_receiver_approach_targets
        )
        assert (
            captured["previous_posture_weight"]
            == HANDOFF_RECEIVER_TRANSLATION_CONTINUITY_WEIGHT
        )
        assert (
            captured["position_weight"]
            == HANDOFF_RECEIVER_TRANSLATION_POSITION_WEIGHT
        )
        assert (
            captured["red_orientation_weight"]
            == HANDOFF_RECEIVER_TRANSLATION_RED_ORIENTATION_WEIGHT
        )
        assert (
            captured["blue_orientation_weight"]
            == HANDOFF_RECEIVER_TRANSLATION_BLUE_ORIENTATION_WEIGHT
        )
        for joint_name, value in donor_hold.items():
            assert np.isclose(
                controller.motion.target[controller._actuator_index_for_joint(joint_name)],
                value,
            )
        assert (
            controller.motion.target[ACTUATOR_NAMES.index("finger_l")]
            == HANDOFF_RECEIVER_WIDE_OPEN
        )

        controller.phase = "handoff_grasp"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_verify"

        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_near_hand",
            lambda *_args, **_kwargs: (True, 0.0),
        )
        controller.phase = "handoff_verify"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_load_transfer"
        expected_frames = round(
            controller.move_duration_s * HANDOFF_OVERLAP_HOLD_DURATION_SCALE * backend.config.fps
        )
        assert controller.motion.frame_count == expected_frames

        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_hand_contact_count",
            lambda *_args: 2,
        )
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_hand_has_bilateral_contact",
            lambda *_args: True,
        )
        controller.phase = "handoff_load_transfer"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "handoff_release"
        assert (
            controller.motion.target[ACTUATOR_NAMES.index("finger_r")]
            == controller.trajectory.cube_grasp_position
        )

        controller.phase = "handoff_release"
        controller.motion = None
        donor_position_before_release = hand_pose(
            backend.model,
            backend.data,
            "r",
            task_frame_inset_m=HANDOFF_HAND_FRAME_INSET_M,
        )[0]
        controller.action(0.0)
        assert controller.cube_hand == "l"
        assert controller.handoff_donor == "r"
        assert controller.motion is not None
        assert controller.motion.after == "handoff_receiver_stability"
        assert captured["side"] == "r"
        assert np.isclose(
            np.linalg.norm(captured["target"] - donor_position_before_release),
            HANDOFF_DONOR_RETREAT_DISTANCE_M,
        )
        assert captured["finger_position"] == FINGER_OPEN
        assert captured["hand_frame_inset_m"] == HANDOFF_HAND_FRAME_INSET_M
        for joint_name in SIDE_JOINTS["l"]:
            assert np.isclose(
                controller.motion.target[controller._actuator_index_for_joint(joint_name)],
                _joint_qpos(backend.model, backend.data, joint_name),
            )

        controller.phase = "handoff_receiver_stability"
        controller.motion = None
        controller.action(0.0)
        assert controller.phase == "handoff_donor_rest"

        controller.phase = "handoff_donor_rest"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "cube_above_drawer"
        for joint_name, value in controller._handoff_donor_rest_targets("r").items():
            actuator_index = controller._actuator_index_for_joint(joint_name)
            assert np.isclose(controller.motion.target[actuator_index], value)
        assert controller.motion.target[ACTUATOR_NAMES.index("finger_r")] == FINGER_CLOSED
        assert (
            controller.motion.target[ACTUATOR_NAMES.index("finger_l")]
            == HANDOFF_RECEIVER_GRASP
        )
        raised_joint = SIDE_JOINTS["r"][2]
        assert np.isclose(
            controller.motion.target[controller._actuator_index_for_joint(raised_joint)],
            -HANDOFF_DONOR_ARM_ELEVATION_RAD,
        )


def test_ik_controller_fails_early_when_handoff_receiver_misses_cube(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=17)
        assert controller.cube_hand == "r"
        assert controller.placement_hand == "l"
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_near_hand",
            lambda *_args, **_kwargs: (False, 0.25),
        )
        controller.phase = "handoff_release"
        controller.motion = None

        controller.action(0.0)

        assert controller.done
        assert not controller.successful
        assert controller.cube_hand == "r"
        assert "receiver grasp was not stable" in controller.status


def test_handoff_stops_before_release_if_grippers_touch(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=7)
        assert controller.cube_hand == "r"
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._hand_hand_deep_contact_count",
            lambda *_args: 1,
        )
        controller.phase = "handoff_grasp"
        controller.motion = None
        controller._handoff_contact_frames = HANDOFF_CONTACT_ABORT_FRAMES - 1

        action = controller.action(0.0)

        assert controller.done
        assert not controller.successful
        assert controller.cube_hand == "r"
        assert controller.handoff_donor is None
        assert controller.motion is None
        assert "grippers remained in contact" in controller.status
        assert action.shape == (len(ACTUATOR_NAMES),)


def test_disabled_early_failures_allow_sustained_gripper_contact(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, early_failures=False)
        controller.reset(episode_seed=7)
        controller.phase = "handoff_grasp"
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._hand_hand_deep_contact_count",
            lambda *_args: 1,
        )

        for _ in range(HANDOFF_CONTACT_ABORT_FRAMES + 2):
            assert not controller._check_handoff_hand_clearance()

        assert not controller.done
        assert "ignored early failure after cube handoff" in controller.status


def test_drawer_opening_ik_uses_only_all_arm_joints(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0)
        controller.reset(episode_seed=1)
        captured: dict[str, object] = {}

        def capture_solution(**kwargs):
            captured.update(kwargs)
            joint_names = _controlled_joints(kwargs["side"])
            joint_values = np.array(
                [
                    backend.data.qpos[backend.model.jnt_qposadr[_joint_id(backend.model, name)]]
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

        assert set(captured) == {
            "side",
            "target",
            "target_rotation",
            "seed",
            "reset_drawers",
            "align_blue_axis",
            "directed_axes",
            "position_weight",
            "orientation_weight",
            "include_collision_penalty",
            "posture_reference",
            "posture_weight",
            "previous_posture_weight",
            "max_posture_deviation_radians",
            "finger_position",
            "hand_frame_inset_m",
        }
        assert "joint_names" not in captured
        assert captured["finger_position"] == DRAWER_FINGER_OPEN
        assert captured["hand_frame_inset_m"] == 0.0
        assert captured["align_blue_axis"] is False
        assert captured["position_weight"] == DRAWER_OPEN_IK_POSITION_WEIGHT
        assert captured["orientation_weight"] == DRAWER_IK_ORIENTATION_WEIGHT
        assert captured["include_collision_penalty"] is False
        assert captured["posture_reference"] == controller._rest_targets(
            controller.drawer_hand
        )
        assert captured["posture_weight"] == DEFAULT_IK_POSTURE_WEIGHT
        assert captured["previous_posture_weight"] == DEFAULT_IK_PREVIOUS_POSTURE_WEIGHT
        assert (
            captured["max_posture_deviation_radians"]
            == DEFAULT_IK_MAX_POSTURE_DEVIATION_RADIANS
        )
        assert "joint_bounds" not in captured
        assert controller.drawer_hand is not None
        assert np.allclose(
            captured["target_rotation"],
            controller.ik_targets["drawer_open"][1],
        )
        assert controller.motion is not None
        assert controller.motion.after == "approach_drawer"
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")]
            == DRAWER_FINGER_OPEN
        )
        optimized_joints = _controlled_joints(controller.drawer_hand)
        assert len(optimized_joints) == 5
        assert optimized_joints == SIDE_JOINTS[controller.drawer_hand]


def test_cube_grasp_descent_and_lift_use_only_arm_ik(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, debug=True, fps=30)) as backend:
        controller = MujocoIKController(
            backend,
            seed=0,
            trajectory_randomization_scale=0.0,
        )
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
                        backend.data.qpos[backend.model.jnt_qposadr[_joint_id(backend.model, name)]]
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
        assert captured["position_weight"] == CUBE_GRASP_IK_POSITION_WEIGHT
        assert captured["orientation_weight"] == CUBE_GRASP_IK_ORIENTATION_WEIGHT
        assert captured["align_red_axis"] is False
        assert captured["align_blue_axis"] is True
        assert captured["directed_axes"] is True
        assert captured["hand_frame_inset_m"] == STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M
        source_position, source_rotation = hand_pose(
            backend.model,
            backend.data,
            controller.cube_hand,
            task_frame_inset_m=STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
        )
        source_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "active_ik_source_marker",
        )
        assert np.allclose(backend.data.xpos[source_id], source_position)
        fixed_body, fixed_tip, _ = _fixed_finger_tip(backend.model, controller.cube_hand)
        moving_body, moving_tip = _moving_finger_tip(backend.model, controller.cube_hand)
        fingertip_midpoint = 0.5 * (
            _world_point(backend.data, fixed_body, fixed_tip)
            + _world_point(backend.data, moving_body, moving_tip)
        )
        assert np.allclose(
            source_position - fingertip_midpoint,
            STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M * source_rotation[:, 1],
        )
        assert np.isclose(
            np.linalg.norm(source_position - fingertip_midpoint),
            STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M,
        )
        assert "joint_bounds" not in captured
        hand_joint = SIDE_JOINTS[controller.cube_hand][-1]
        assert captured["posture_reference"][hand_joint] == 0.0
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.cube_hand}")]
            == FINGER_OPEN
        )

        controller.phase = "descend_to_cube"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert captured["orientation_weight"] == CUBE_GRASP_IK_ORIENTATION_WEIGHT
        assert captured["hand_frame_inset_m"] == STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M
        changed = set(
            np.flatnonzero(~np.isclose(controller.motion.target, controller.motion.start))
        )
        cnc_indices = {ACTUATOR_NAMES.index(name) for name in ("cnc_x", "cnc_y", "head_z")}
        assert changed.isdisjoint(cnc_indices)

        controller.phase = "close_cube"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "settle_cube_grasp"
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.cube_hand}")]
            == CUBE_GRASP
        )

        controller.phase = "settle_cube_grasp"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "cube_clearance"
        assert controller.motion.frame_count == round(
            CUBE_GRASP_SETTLE_DURATION_S * backend.config.fps
        )

        controller.phase = "cube_clearance"
        controller.motion = None
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_near_hand",
            lambda *_args, **_kwargs: (True, 0.0),
        )
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_hand_contact_count",
            lambda *_args: 2,
        )
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_hand_has_bilateral_contact",
            lambda *_args: True,
        )
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "return_without_base"
        assert captured["orientation_weight"] == CUBE_GRASP_IK_ORIENTATION_WEIGHT
        assert captured["hand_frame_inset_m"] == STORAGE_CUBE_GRASP_HAND_FRAME_INSET_M
        changed = set(
            np.flatnonzero(~np.isclose(controller.motion.target, controller.motion.start))
        )
        assert changed.isdisjoint(cnc_indices)
        idle_hand = "r" if controller.cube_hand == "l" else "l"
        transit_targets = controller._transit_targets()
        for joint_name in SIDE_JOINTS[idle_hand]:
            assert np.isclose(
                controller.motion.target[controller._actuator_index_for_joint(joint_name)],
                transit_targets[joint_name],
            )


def test_drawer_contact_uses_only_the_selected_arm() -> None:
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
            arm_indices = {
                controller._actuator_index_for_joint(name)
                for name in SIDE_JOINTS[controller.drawer_hand]
            }
            assert changed_nonfinger <= arm_indices


def test_drawer_opening_closes_finger_to_tighter_grasp() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=1)
        assert controller.drawer_hand is not None
        controller.phase = "grasp_drawer"
        controller.motion = None

        controller.action(0.0)

        assert controller.motion is not None
        assert controller.motion.after == "pull_drawer"
        assert np.isclose(
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")],
            DRAWER_GRASP,
        )


def test_pull_drawer_uses_arm_only_ik(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=16)
        controller.reset(episode_seed=0)
        captured: dict[str, object] = {}

        def current_joint_solution(**kwargs):
            captured.update(kwargs)
            names = _controlled_joints(kwargs["side"])
            return SimpleNamespace(
                joint_names=names,
                joint_values=np.array([backend.data.joint(name).qpos[0] for name in names]),
            )

        monkeypatch.setattr(controller.ik, "solve", current_joint_solution)
        controller.phase = "pull_drawer"
        controller.motion = None
        controller.action(0.0)

        assert controller.motion is not None
        assert controller.motion.after == "release_open_drawer"
        drawer_joint = f"base_link_base_drawer_{controller.drawer_index}_joint"
        drawer_id = _joint_id(backend.model, drawer_joint)
        expected_target, _ = controller._drawer_contact_target(
            drawer_qpos=float(backend.model.jnt_range[drawer_id, 0])
        )
        expected_target = expected_target + (
            DRAWER_PULL_OVERSHOOT_M - controller.trajectory.drawer_pull_shortfall
        ) * controller._drawer_robotward_axis()
        assert np.allclose(captured["target"], expected_target)
        assert "joint_names" not in captured
        assert set(_controlled_joints(controller.drawer_hand)).isdisjoint(
            {
                "base_link_base_cnc_x_joint",
                "cnc_x_link_cnc_x_cnc_y_joint",
                "cnc_y_link_cnc_y_head_joint",
            }
        )


def test_drawer_closing_uses_arm_only_contact_and_push(monkeypatch) -> None:
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
                        backend.data.qpos[backend.model.jnt_qposadr[_joint_id(backend.model, name)]]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", current_joint_solution)
        controller.phase = "close_drawer_ik"
        controller.action(0.0)

        assert "joint_bounds" not in captured
        assert captured["finger_position"] == FINGER_CLOSED
        assert captured["align_blue_axis"] is False
        assert np.allclose(
            captured["target_rotation"],
            controller._working_hand_rotation(controller.drawer_hand),
        )
        assert controller.motion is not None
        assert controller.motion.after == "push_drawer"
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")]
            == FINGER_CLOSED
        )

        captured.clear()
        controller.phase = "push_drawer"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "push_drawer_closed"
        assert captured["align_blue_axis"] is False
        assert np.allclose(
            captured["target_rotation"],
            controller._working_hand_rotation(controller.drawer_hand),
        )
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.drawer_hand}")]
            == FINGER_CLOSED
        )

        cnc_indices = {ACTUATOR_NAMES.index(name) for name in ("cnc_x", "cnc_y", "head_z")}
        changed = set(
            np.flatnonzero(~np.isclose(controller.motion.target, controller.motion.start))
        )
        assert changed.isdisjoint(cnc_indices)


def test_debug_shows_live_pregrasp_targets() -> None:
    with MujocoBackend(SimulationConfig(render=False, debug=True, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4, trajectory_randomization_scale=0.0)
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
            backend.data.geom_xpos[cube_id] + np.array([0.0, 0.0, CUBE_GRASP_APPROACH_HEIGHT_M]),
        )
        for marker_name, (target, rotation) in zip(
            IK_TARGET_MARKERS, controller.ik_targets.values(), strict=True
        ):
            marker_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, marker_name)
            assert marker_id >= 0
            assert np.allclose(backend.data.xpos[marker_id], target)
            assert np.allclose(backend.data.xmat[marker_id].reshape(3, 3), rotation)

        expected_position, expected_rotation = _drawer_hand_target_pose(
            backend.model, backend.data, controller.drawer_index, controller.drawer_hand
        )
        assert np.allclose(controller.ik_targets["drawer_open"][0], expected_position)
        assert np.allclose(controller.ik_targets["drawer_open"][1], expected_rotation)
        expected_close_position, _ = _drawer_close_target_pose(
            backend.model, backend.data, controller.drawer_index, controller.drawer_hand
        )
        handle_center, _ = drawer_handle_pose(backend.model, backend.data, controller.drawer_index)
        assert np.allclose(controller.ik_targets["drawer_close"][0], expected_close_position)
        assert np.allclose(controller.ik_targets["drawer_close"][1], expected_rotation)
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
        assert np.allclose(backend.data.xpos[cube_marker_id], backend.data.geom_xpos[cube_id])
        assert np.allclose(
            backend.data.xmat[cube_marker_id].reshape(3, 3),
            backend.data.geom_xmat[cube_id].reshape(3, 3),
        )

def test_debug_active_ik_pair_uses_exact_source_and_target_frames() -> None:
    with MujocoBackend(SimulationConfig(render=False, debug=True, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=7)
        side = controller.cube_hand
        assert side is not None
        target_position = np.array([0.12, -0.34, 0.56])
        target_rotation = np.eye(3)

        controller._activate_ik_debug_frames(
            side=side,
            target=target_position,
            rotation=target_rotation,
            hand_frame_inset_m=HAND_TASK_FRAME_INSET_M,
        )

        expected_source_position, expected_source_rotation = hand_pose(
            backend.model,
            backend.data,
            side,
            task_frame_inset_m=HAND_TASK_FRAME_INSET_M,
        )
        source_id = mujoco.mj_name2id(
            backend.model, mujoco.mjtObj.mjOBJ_BODY, "active_ik_source_marker"
        )
        target_id = mujoco.mj_name2id(
            backend.model, mujoco.mjtObj.mjOBJ_BODY, "active_ik_target_marker"
        )
        assert np.allclose(backend.data.xpos[source_id], expected_source_position)
        assert np.allclose(
            backend.data.xmat[source_id].reshape(3, 3), expected_source_rotation
        )
        assert np.allclose(backend.data.xpos[target_id], target_position)
        assert np.allclose(backend.data.xmat[target_id].reshape(3, 3), target_rotation)


def test_cube_placement_target_uses_extra_top_row_clearance() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        for drawer_index in (2, 5, 8):
            handle_center, _ = drawer_handle_pose(backend.model, backend.data, drawer_index)
            target, _ = _cube_above_drawer_pose(backend.model, backend.data, drawer_index)

            assert np.isclose(target[1] - handle_center[1], CUBE_PLACE_HANDLE_Y_OFFSET_M)
            expected_height = CUBE_PLACE_HANDLE_Z_OFFSET_M + (
                TOP_ROW_CUBE_PLACE_CLEARANCE_M if drawer_index <= 3 else 0.0
            )
            assert np.isclose(target[2] - handle_center[2], expected_height)


def test_cube_placement_ik_directly_aligns_hand_with_target_frame(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
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
                        backend.data.qpos[backend.model.jnt_qposadr[_joint_id(backend.model, name)]]
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
        assert captured["align_red_axis"] is False
        assert captured["align_blue_axis"] is True
        assert captured["directed_axes"] is True
        planning_qpos = captured["planning_qpos"]
        workspace_b = resolve_mujoco_workspace(
            backend.model,
            Workspace.DRAWERS,
            drawer=controller.drawer_index,
        )
        for joint_name, expected in workspace_b.items():
            joint_id = _joint_id(backend.model, joint_name)
            assert np.isclose(
                planning_qpos[backend.model.jnt_qposadr[joint_id]],
                expected,
            )
        hand_joint = SIDE_JOINTS[controller.cube_hand][-1]
        assert captured["posture_reference"][hand_joint] == 0.0
        assert captured["finger_position"] == CUBE_GRASP
        assert np.allclose(target_rotation[:, 2], (0.0, 0.0, 1.0))
        assert np.isclose(target_rotation[2, 0], 0.0)
        assert controller.motion is not None
        assert controller.motion.after == "move_to_b_for_place"
        for actuator_name in ("cnc_x", "cnc_y", "head_z"):
            actuator_index = ACTUATOR_NAMES.index(actuator_name)
            assert np.isclose(
                controller.motion.target[actuator_index],
                controller.motion.start[actuator_index],
            )
        assert (
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.cube_hand}")]
            == CUBE_GRASP
        )

        controller.phase = "enter_b_for_place"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "drop_cube"


def test_cube_hand_retreats_horizontally_before_returning_to_rest(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
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
                        backend.data.qpos[backend.model.jnt_qposadr[_joint_id(backend.model, name)]]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", current_joint_solution)
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_center_inside_drawer",
            lambda *_args: True,
        )
        hand_position, hand_rotation = grasp_pose(
            backend.model,
            backend.data,
            controller.cube_hand,
        )
        controller.phase = "retreat_from_drawer"
        controller.motion = None
        controller.action(0.0)

        retreat_target = captured["target"]
        retreat_delta = retreat_target - hand_position
        assert np.isclose(np.linalg.norm(retreat_delta), CUBE_PLACE_RETREAT_DISTANCE_M)
        assert np.isclose(retreat_delta[2], 0.0)
        assert np.allclose(captured["target_rotation"], hand_rotation)
        assert captured["finger_position"] == FINGER_OPEN
        assert controller.motion is not None
        assert controller.motion.after == "choose_closing_hand"


def test_cube_release_removes_gripper_friction_until_next_reset() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(
            backend,
            seed=0,
            trajectory_randomization_scale=0.0,
        )
        controller.reset(episode_seed=1)
        assert controller.cube_hand is not None

        for pair_name in CUBE_GRIPPER_CONTACT_PAIRS:
            pair_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_PAIR,
                pair_name,
            )
            assert np.allclose(
                backend.model.pair_friction[pair_id],
                CUBE_GRIPPER_GRASP_FRICTION,
            )

        controller.phase = "drop_cube"
        controller.motion = None
        controller.action(0.0)

        assert controller.motion is not None
        assert controller.motion.after == "settle_cube_in_drawer"

        for pair_name in CUBE_GRIPPER_CONTACT_PAIRS:
            pair_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_PAIR,
                pair_name,
            )
            assert np.allclose(
                backend.model.pair_friction[pair_id],
                CUBE_GRIPPER_RELEASE_FRICTION,
            )

        controller.phase = "settle_cube_in_drawer"
        controller.motion = None
        controller.action(0.0)

        assert controller.motion is not None
        assert controller.motion.after == "retreat_from_drawer"
        assert controller.motion.frame_count == round(
            CUBE_DRAWER_SETTLE_DURATION_S * backend.config.fps
        )
        assert np.isclose(
            controller.motion.target[ACTUATOR_NAMES.index(f"finger_{controller.cube_hand}")],
            FINGER_OPEN,
        )

        controller.reset(episode_seed=2)
        for pair_name in CUBE_GRIPPER_CONTACT_PAIRS:
            pair_id = mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_PAIR,
                pair_name,
            )
            assert np.allclose(
                backend.model.pair_friction[pair_id],
                CUBE_GRIPPER_GRASP_FRICTION,
            )


def test_optional_cube_grasp_lock_captures_pose_transfers_and_releases() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(
            backend,
            seed=0,
            trajectory_randomization_scale=0.0,
            cube_grasp_lock=True,
        )
        controller.reset(episode_seed=1)

        equality_ids = {
            side: mujoco.mj_name2id(
                backend.model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                name,
            )
            for side, name in CUBE_GRASP_LOCK_EQUALITIES.items()
        }
        assert all(equality_id >= 0 for equality_id in equality_ids.values())
        assert not np.any(backend.data.eq_active)

        controller._lock_cube_to_hand("l")
        assert controller.cube_grasp_lock_side == "l"
        assert backend.data.eq_active[equality_ids["l"]]
        assert not backend.data.eq_active[equality_ids["r"]]

        controller._lock_cube_to_hand("r")
        assert controller.cube_grasp_lock_side == "r"
        assert not backend.data.eq_active[equality_ids["l"]]
        assert backend.data.eq_active[equality_ids["r"]]

        controller.phase = "drop_cube"
        controller.motion = None
        controller.action(0.0)
        assert controller.cube_grasp_lock_side is None
        assert not np.any(backend.data.eq_active)


def test_optional_cube_grasp_lock_accepts_a_close_approach_without_contact(
    monkeypatch,
) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(
            backend,
            seed=0,
            trajectory_randomization_scale=0.0,
            cube_grasp_lock=True,
        )
        controller.reset(episode_seed=1)
        assert controller.cube_hand is not None
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_near_hand",
            lambda *_args, **_kwargs: (True, 0.01),
        )
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_hand_contact_count",
            lambda *_args, **_kwargs: 0,
        )
        monkeypatch.setattr(
            "simulation.backends.mujoco.ik._cube_hand_has_bilateral_contact",
            lambda *_args, **_kwargs: False,
        )
        monkeypatch.setattr(controller, "_solve_cube_grasp", lambda **_kwargs: {})
        controller.phase = "cube_clearance"
        controller.motion = None

        controller.action(0.0)

        assert not controller.done
        assert controller.cube_grasp_lock_side == controller.cube_hand
        equality_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            CUBE_GRASP_LOCK_EQUALITIES[controller.cube_hand],
        )
        assert backend.data.eq_active[equality_id]


def test_cube_grasp_lock_preserves_the_live_hand_cube_transform() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        hand_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hand_l_link",
        )
        cube_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "cube_link",
        )
        equality_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            CUBE_GRASP_LOCK_EQUALITIES["l"],
        )
        hand_rotation = backend.data.xmat[hand_id].reshape(3, 3)
        expected_relative_position = hand_rotation.T @ (
            backend.data.xpos[cube_id] - backend.data.xpos[hand_id]
        )

        _set_cube_grasp_lock(backend.model, backend.data, "l")

        assert np.allclose(
            backend.model.eq_data[equality_id, 3:6],
            expected_relative_position,
        )
        assert np.isclose(
            np.linalg.norm(backend.model.eq_data[equality_id, 6:10]),
            1.0,
        )


def test_cube_drop_assist_keeps_the_cube_inside_and_removes_bounce() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(
            backend,
            seed=0,
            trajectory_randomization_scale=0.0,
            cube_drop_assist=True,
        )
        controller.reset(episode_seed=1)
        controller.drawer_index = 5
        floor_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "drawer_5_link_collision_box_02_geom",
        )
        cube_geom_id = mujoco.mj_name2id(
            backend.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "cube_link_collision_box_01_geom",
        )
        cube_joint_id = _joint_id(backend.model, "cube_link_free_joint")
        qpos_id = int(backend.model.jnt_qposadr[cube_joint_id])
        dof_id = int(backend.model.jnt_dofadr[cube_joint_id])
        floor_rotation = backend.data.geom_xmat[floor_id].reshape(3, 3)
        outside_local_position = np.array(
            [
                backend.model.geom_size[floor_id, 0] + 0.02,
                backend.model.geom_size[floor_id, 1] + 0.02,
                backend.model.geom_size[floor_id, 2] + 0.05,
            ]
        )
        backend.data.qpos[qpos_id : qpos_id + 3] = (
            backend.data.geom_xpos[floor_id]
            + floor_rotation @ outside_local_position
        )
        backend.data.qvel[dof_id : dof_id + 6] = np.array(
            [0.4, -0.3, 0.2, 1.0, 2.0, 3.0]
        )
        mujoco.mj_forward(backend.model, backend.data)

        controller._stabilize_cube_drawer_drop()

        cube_rotation = backend.data.geom_xmat[cube_geom_id].reshape(3, 3)
        cube_half_extents = (
            np.abs(floor_rotation.T @ cube_rotation)
            @ backend.model.geom_size[cube_geom_id]
        )
        safe_half_width = (
            backend.model.geom_size[floor_id, :2]
            - cube_half_extents[:2]
            - CONTACT_MARGIN_M
        )
        local_position = floor_rotation.T @ (
            backend.data.geom_xpos[cube_geom_id]
            - backend.data.geom_xpos[floor_id]
        )
        local_velocity = floor_rotation.T @ backend.data.qvel[dof_id : dof_id + 3]
        assert np.all(np.abs(local_position[:2]) <= safe_half_width + 1e-12)
        assert np.allclose(local_velocity[:2], 0.0)
        assert local_velocity[2] <= -CUBE_DROP_ASSIST_FALL_SPEED_M_S
        assert np.allclose(backend.data.qvel[dof_id + 3 : dof_id + 6], 0.0)


def test_closed_drawer_vertical_retreat_moves_away_from_handle() -> None:
    distance = DRAWER_CLOSE_RETREAT_VERTICAL_DISTANCE_M
    assert _closed_drawer_vertical_retreat(0.20, 0.15) == distance
    assert _closed_drawer_vertical_retreat(0.10, 0.15) == -distance
    assert _closed_drawer_vertical_retreat(0.15, 0.15) == 0.0


def test_closing_hand_retreats_away_from_handle_before_final_transit(monkeypatch) -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=0, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=1)
        controller.drawer_hand = "l"
        drawer_id = _joint_id(
            backend.model,
            f"base_link_base_drawer_{controller.drawer_index}_joint",
        )
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = backend.model.jnt_range[
            drawer_id, 1
        ]
        mujoco.mj_forward(backend.model, backend.data)
        fingertip_position, fingertip_rotation = hand_pose(
            backend.model,
            backend.data,
            "l",
            task_frame_inset_m=0.0,
        )
        closed_handle_position = drawer_handle_pose(
            backend.model,
            backend.data,
            controller.drawer_index,
        )[0]
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
                        backend.data.qpos[backend.model.jnt_qposadr[_joint_id(backend.model, name)]]
                        for name in names
                    ]
                ),
            )

        monkeypatch.setattr(controller.ik, "solve", current_joint_solution)
        controller.phase = "release_closed_drawer"
        controller.motion = None
        controller.action(0.0)

        retreat_delta = captured["target"] - fingertip_position
        assert np.allclose(retreat_delta[:2], 0.0)
        assert np.isclose(
            retreat_delta[2],
            _closed_drawer_vertical_retreat(
                float(fingertip_position[2]),
                float(closed_handle_position[2]),
            ),
        )
        assert np.allclose(captured["target_rotation"], fingertip_rotation)
        assert captured["hand_frame_inset_m"] == 0.0
        assert captured["position_weight"] == DRAWER_RETREAT_IK_POSITION_WEIGHT
        assert captured["finger_position"] == FINGER_CLOSED
        assert controller.motion is not None
        assert controller.motion.after == "closed_drawer_hand_rest"

        controller.phase = "closed_drawer_hand_rest"
        controller.motion = None
        controller.action(0.0)
        assert controller.motion is not None
        assert controller.motion.after == "move_to_a_final"


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
        controller = MujocoIKController(backend, seed=4, trajectory_randomization_scale=0.0)
        controller.reset(episode_seed=7)
        fixed_marker_ids = [
            mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, IK_TARGET_MARKERS[index])
            for index in (0, 2)
        ]
        drawer_marker_ids = [
            mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, IK_TARGET_MARKERS[index])
            for index in (1, 3)
        ]
        fixed_positions = backend.data.xpos[fixed_marker_ids].copy()
        drawer_position = controller.ik_targets["drawer_open"][0].copy()

        drawer_joint = f"base_link_base_drawer_{controller.drawer_index}_joint"
        drawer_id = _joint_id(backend.model, drawer_joint)
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = backend.model.jnt_range[
            drawer_id, 0
        ]
        mujoco.mj_forward(backend.model, backend.data)
        expected_open_position, expected_rotation = _drawer_hand_target_pose(
            backend.model, backend.data, controller.drawer_index, controller.drawer_hand
        )
        expected_close_position, _ = _drawer_close_target_pose(
            backend.model, backend.data, controller.drawer_index, controller.drawer_hand
        )

        controller.action(0.0)

        assert not np.allclose(expected_open_position, drawer_position)
        assert np.allclose(controller.ik_targets["drawer_open"][0], expected_open_position)
        assert np.allclose(controller.ik_targets["drawer_close"][0], expected_close_position)
        for target_name in ("drawer_open", "drawer_close"):
            assert np.allclose(controller.ik_targets[target_name][1], expected_rotation)
        assert np.allclose(backend.data.xpos[drawer_marker_ids[0]], expected_open_position)
        assert np.allclose(backend.data.xpos[drawer_marker_ids[1]], expected_close_position)
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
        backend.data.qpos[backend.model.jnt_qposadr[drawer_id]] = backend.model.jnt_range[
            drawer_id, 0
        ]
        mujoco.mj_forward(backend.model, backend.data)
        controller._select_post_drawer_hands_and_targets()

        assert np.allclose(controller.ik_targets["cube_place"][0], initial_position)
        assert np.allclose(controller.ik_targets["cube_place"][1], initial_rotation)
        assert np.allclose(backend.data.xpos[marker_id], initial_position)
        assert np.allclose(backend.data.xmat[marker_id].reshape(3, 3), initial_rotation)


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


def test_disabled_early_failures_warn_and_continue() -> None:
    with MujocoBackend(SimulationConfig(render=False, fps=30)) as backend:
        controller = MujocoIKController(backend, seed=4, early_failures=False)
        controller.reset(episode_seed=7)
        controller.drawer_hand = "l"
        controller.phase = "release_open_drawer"

        controller.action(0.0)

        assert not controller.done
        assert controller.motion is not None
        assert controller.motion.after == "retreat_open_drawer"
        assert "ignored early failure after drawer opening" in controller.status


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
        controller.phase = "retreat_from_drawer"

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
