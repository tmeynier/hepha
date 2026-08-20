"""Serializable MuJoCo intervention states used by the DAgger workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np

from simulation.backends.mujoco import MujocoBackend
from simulation.backends.mujoco.episode import initialize_task_episode

SNAPSHOT_VERSION = 1
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


@dataclass(frozen=True)
class InterventionMetadata:
    version: int
    seed: int
    source_episode: int
    source_step: int
    drawer_index: int
    current_task_phase: int
    next_task_phase: int
    fps: int
    camera: str
    task: str
    policy_path: str
    policy_type: str
    domain_randomization_scale: float
    created_at: str

    @classmethod
    def create(cls, **values) -> InterventionMetadata:
        return cls(
            version=SNAPSHOT_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            **values,
        )


@dataclass(frozen=True)
class InterventionSnapshot:
    directory: Path
    metadata: InterventionMetadata
    state: np.ndarray


def next_snapshot_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = {
        path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("mark-")
    }
    index = 0
    while f"mark-{index:06d}" in existing:
        index += 1
    return root / f"mark-{index:06d}"


def capture_integration_state(backend: MujocoBackend) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(backend.model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(backend.model, backend.data, state, STATE_SPEC)
    return state


def save_snapshot(
    root: Path,
    *,
    backend: MujocoBackend,
    metadata: InterventionMetadata,
) -> Path:
    directory = next_snapshot_directory(root)
    directory.mkdir(parents=False)
    state = capture_integration_state(backend)
    with (directory / "state.npz").open("wb") as stream:
        np.savez_compressed(stream, state=state, state_spec=int(STATE_SPEC))
    (directory / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def load_snapshot(directory: Path) -> InterventionSnapshot:
    metadata_path = directory / "metadata.json"
    state_path = directory / "state.npz"
    if not metadata_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Invalid intervention snapshot directory: {directory}")
    metadata = InterventionMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
    if metadata.version != SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported intervention snapshot version {metadata.version}; "
            f"expected {SNAPSHOT_VERSION}"
        )
    with np.load(state_path) as payload:
        state_spec = int(payload["state_spec"])
        state = np.asarray(payload["state"], dtype=np.float64)
    if state_spec != int(STATE_SPEC):
        raise ValueError(f"Snapshot {directory} uses an incompatible MuJoCo state spec")
    return InterventionSnapshot(directory=directory, metadata=metadata, state=state)


def restore_snapshot(backend: MujocoBackend, snapshot: InterventionSnapshot) -> None:
    expected_scale = snapshot.metadata.domain_randomization_scale
    if not np.isclose(backend.domain_randomization_scale, expected_scale):
        raise ValueError(
            "Snapshot domain-randomization scale does not match the correction backend: "
            f"{expected_scale} != {backend.domain_randomization_scale}"
        )
    initialize_task_episode(backend, seed=snapshot.metadata.seed)
    expected_size = mujoco.mj_stateSize(backend.model, STATE_SPEC)
    if snapshot.state.shape != (expected_size,):
        raise ValueError(
            f"Snapshot state has shape {snapshot.state.shape}; expected ({expected_size},)"
        )
    mujoco.mj_setState(backend.model, backend.data, snapshot.state, STATE_SPEC)
    mujoco.mj_forward(backend.model, backend.data)
    backend.sync_viewer()


def discover_snapshots(root: Path) -> list[InterventionSnapshot]:
    if not root.is_dir():
        raise FileNotFoundError(f"Intervention directory not found: {root}")
    snapshots = [load_snapshot(path) for path in sorted(root.glob("mark-*")) if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No intervention snapshots found in {root}")
    return snapshots
