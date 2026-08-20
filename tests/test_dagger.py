from __future__ import annotations

from argparse import Namespace

import numpy as np
from hepha_lerobot.dagger.correct import (
    _prepare_storage,
    _shard_root,
    smooth_action_sequence,
)
from hepha_lerobot.dagger.snapshot import (
    InterventionMetadata,
    load_snapshot,
    restore_snapshot,
    save_snapshot,
)

from simulation import SimulationConfig
from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import initialize_task_episode


def test_action_smoothing_preserves_shape_and_limits() -> None:
    actions = np.zeros((9, 2), dtype=np.float64)
    actions[4] = 1.0

    smoothed = smooth_action_sequence(
        actions,
        window=7,
        order=2,
        low=np.zeros(2),
        high=np.full(2, 0.8),
    )

    assert smoothed.shape == actions.shape
    assert np.all(smoothed >= 0.0)
    assert np.all(smoothed <= 0.8)
    assert smoothed[4, 0] < 0.8
    assert smoothed[3, 0] > 0.0


def test_correction_resume_preserves_completed_shards(tmp_path) -> None:
    root = tmp_path / "corrections"
    shard_root = _shard_root(root)
    _prepare_storage(
        Namespace(root=root, overwrite=True, resume=False),
        shard_root,
    )
    completed_marker = shard_root / "mark-000000" / "complete.txt"
    completed_marker.parent.mkdir()
    completed_marker.write_text("complete", encoding="utf-8")
    root.mkdir()
    (root / "stale.txt").write_text("stale", encoding="utf-8")

    _prepare_storage(
        Namespace(root=root, overwrite=False, resume=True),
        shard_root,
    )

    assert completed_marker.read_text(encoding="utf-8") == "complete"
    assert not root.exists()


def test_intervention_snapshot_restores_complete_mujoco_state(tmp_path) -> None:
    config = SimulationConfig(
        render=False,
        options={"domain_randomization_scale": 0.0},
    )
    with MujocoBackend(config) as backend:
        initialize_task_episode(backend, seed=123)
        backend.data.qpos[backend.qpos_ids[0]] += 0.01
        backend.data.ctrl[backend.actuator_ids[1]] += 0.02
        expected_qpos = backend.data.qpos.copy()
        expected_ctrl = backend.data.ctrl.copy()
        metadata = InterventionMetadata.create(
            seed=123,
            source_episode=0,
            source_step=10,
            drawer_index=4,
            current_task_phase=2,
            next_task_phase=2,
            fps=30,
            camera="head_camera",
            task="test task",
            policy_path="models/test",
            policy_type="act",
            domain_randomization_scale=0.0,
        )
        directory = save_snapshot(tmp_path, backend=backend, metadata=metadata)

        backend.data.qpos[:] = 0.0
        backend.data.ctrl[:] = 0.0
        restore_snapshot(backend, load_snapshot(directory))

        np.testing.assert_allclose(backend.data.qpos, expected_qpos)
        np.testing.assert_allclose(backend.data.ctrl, expected_ctrl)
        assert load_snapshot(directory).metadata.drawer_index == 4
