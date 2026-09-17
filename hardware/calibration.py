"""Shared Feetech calibration storage and bidirectional joint conversion."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

try:
    from .axes import AxisDefinition
except ImportError:  # Direct hardware script execution.
    from axes import AxisDefinition

STEPS_PER_REVOLUTION = 4096


@dataclass(frozen=True)
class CalibratedAxis:
    servo_id: int
    label: str
    joint_name: str
    calibration: dict[str, object]

    @property
    def mujoco_actuator(self) -> str:
        """Backward-compatible name for existing MuJoCo consumers."""
        return self.joint_name

    @property
    def actuator(self) -> str:
        """Backward-compatible short name used by existing teleoperation code."""
        return self.joint_name


def wrapped_encoder_delta(value: int, reference: int) -> int:
    """Return the shortest signed 12-bit encoder displacement from reference."""
    half_turn = STEPS_PER_REVOLUTION // 2
    return (value - reference + half_turn) % STEPS_PER_REVOLUTION - half_turn


def build_range_calibration(
    motor_id: int,
    axis: AxisDefinition,
    *,
    encoder_low: int,
    encoder_high: int,
    sample_count: int,
) -> dict[str, object]:
    """Build a calibration from continuously unwrapped sweep endpoints."""

    direction = axis.encoder_direction
    if direction not in (-1, 1):
        raise ValueError("Encoder direction must be -1 or +1.")
    if encoder_high <= encoder_low:
        raise ValueError("Recorded encoder range must have positive width.")
    span = encoder_high - encoder_low
    if span >= STEPS_PER_REVOLUTION:
        raise ValueError("Recorded encoder range spans a complete revolution.")

    encoder_home = round((encoder_low + encoder_high) / 2)
    if direction > 0:
        encoder_at_q_min = encoder_low
        encoder_at_q_max = encoder_high
    else:
        encoder_at_q_min = encoder_high
        encoder_at_q_max = encoder_low
    q_home = (axis.q_min + axis.q_max) / 2
    return {
        "servo_id": motor_id,
        **asdict(axis),
        "joint_name": axis.joint_name,
        "q_home": q_home,
        "raw_min": encoder_at_q_min % STEPS_PER_REVOLUTION,
        "raw_home": encoder_home % STEPS_PER_REVOLUTION,
        "raw_max": encoder_at_q_max % STEPS_PER_REVOLUTION,
        "raw_min_delta": encoder_at_q_min - encoder_home,
        "raw_max_delta": encoder_at_q_max - encoder_home,
        "direction": direction,
        "mean_steps_per_radian": span / (axis.q_max - axis.q_min),
        "recorded_samples": sample_count,
        "encoder_span": span,
    }


def save_calibration(path: Path, calibration: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(calibration, indent=2) + "\n")
    os.replace(temporary_path, path)


def load_calibration(path: Path) -> dict[str, object]:
    try:
        calibration = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load calibration file {path}: {exc}") from exc
    if calibration.get("schema_version") != 1 or not isinstance(calibration.get("axes"), dict):
        raise RuntimeError(f"Unsupported calibration file format: {path}")
    return calibration


def _encoder_to_joint_points(
    axis_calibration: dict[str, object],
) -> list[tuple[int, float]]:
    calibration_points = (
        (int(axis_calibration["raw_min_delta"]), float(axis_calibration["q_min"])),
        (0, float(axis_calibration["q_home"])),
        (int(axis_calibration["raw_max_delta"]), float(axis_calibration["q_max"])),
    )
    points_by_delta: dict[int, float] = {}
    for encoder_delta, q_value in calibration_points:
        previous = points_by_delta.get(encoder_delta)
        if previous is not None and not math.isclose(previous, q_value):
            raise ValueError("Calibration assigns different joint positions to one encoder value.")
        points_by_delta[encoder_delta] = q_value
    points = sorted(points_by_delta.items())
    if len(points) < 2:
        raise ValueError("Calibration needs at least two distinct encoder positions.")
    return points


def raw_to_joint_position(raw: int, axis_calibration: dict[str, object]) -> float:
    """Map a raw cyclic encoder reading through min, home, and max points."""
    raw_home = int(axis_calibration["raw_home"])
    points = _encoder_to_joint_points(axis_calibration)
    shortest_delta = wrapped_encoder_delta(raw, raw_home)
    interval_min, interval_max = points[0][0], points[-1][0]

    def distance_to_interval(candidate: int) -> int:
        if candidate < interval_min:
            return interval_min - candidate
        if candidate > interval_max:
            return candidate - interval_max
        return 0

    delta = min(
        (
            shortest_delta - STEPS_PER_REVOLUTION,
            shortest_delta,
            shortest_delta + STEPS_PER_REVOLUTION,
        ),
        key=lambda candidate: (distance_to_interval(candidate), abs(candidate)),
    )

    if delta <= points[0][0]:
        return points[0][1]
    if delta >= points[-1][0]:
        return points[-1][1]
    for (left_delta, left_q), (right_delta, right_q) in pairwise(points):
        if left_delta <= delta <= right_delta:
            fraction = (delta - left_delta) / (right_delta - left_delta)
            return left_q + fraction * (right_q - left_q)
    raise AssertionError("Encoder interpolation interval was not found.")


def joint_to_raw_position(q: float, axis_calibration: dict[str, object]) -> int:
    """Invert the piecewise calibration and return a bounded cyclic raw target."""
    q_min = float(axis_calibration["q_min"])
    q_home = float(axis_calibration["q_home"])
    q_max = float(axis_calibration["q_max"])
    bounded_q = min(q_max, max(q_min, float(q)))
    q_points = sorted(
        {
            q_min: int(axis_calibration["raw_min_delta"]),
            q_home: 0,
            q_max: int(axis_calibration["raw_max_delta"]),
        }.items()
    )
    if len(q_points) < 2:
        raise ValueError("Calibration needs at least two distinct joint positions.")

    if bounded_q <= q_points[0][0]:
        delta = q_points[0][1]
    elif bounded_q >= q_points[-1][0]:
        delta = q_points[-1][1]
    else:
        delta = 0.0
        for (left_q, left_delta), (right_q, right_delta) in pairwise(q_points):
            if left_q <= bounded_q <= right_q:
                fraction = (bounded_q - left_q) / (right_q - left_q)
                delta = left_delta + fraction * (right_delta - left_delta)
                break
    raw_home = int(axis_calibration["raw_home"])
    return (raw_home + round(delta)) % STEPS_PER_REVOLUTION


def load_calibrated_axes(
    calibration_path: Path,
    servo_ids: Sequence[int] | None = None,
    joint_names: Sequence[str] | None = None,
) -> tuple[dict[str, object], tuple[CalibratedAxis, ...]]:
    """Load calibrated axes selected by servo ID or semantic joint name."""
    if servo_ids and joint_names:
        raise ValueError("Select calibrated axes by IDs or joint names, not both.")
    calibration = load_calibration(calibration_path)
    axes = calibration["axes"]
    assert isinstance(axes, dict)
    available: list[CalibratedAxis] = []
    for key, value in axes.items():
        if not isinstance(value, dict):
            raise RuntimeError(f"Calibration axis {key!r} is invalid.")
        try:
            servo_id = int(value.get("servo_id", key))
            joint_name = str(
                value["joint_name"] if "joint_name" in value else value["mujoco_actuator"]
            )
            label = str(value["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Calibration axis {key!r} is incomplete.") from exc
        available.append(CalibratedAxis(servo_id, label, joint_name, value))

    if servo_ids:
        selected_ids = set(int(servo_id) for servo_id in servo_ids)
        selected = [axis for axis in available if axis.servo_id in selected_ids]
        missing = sorted(selected_ids - {axis.servo_id for axis in selected})
        if missing:
            if len(missing) == 1:
                raise RuntimeError(
                    f"Servo ID {missing[0]} is not calibrated in {calibration_path}."
                )
            raise RuntimeError(f"Servo IDs are not calibrated in {calibration_path}: {missing}")
    elif joint_names:
        selected_names = set(joint_names)
        selected = [axis for axis in available if axis.joint_name in selected_names]
        missing = sorted(selected_names - {axis.joint_name for axis in selected})
        if missing:
            raise RuntimeError(f"Joints are not calibrated in {calibration_path}: {missing}")
    else:
        selected = available
    if not selected:
        raise RuntimeError(f"No calibrated axes were found in {calibration_path}.")
    selected.sort(key=lambda axis: axis.servo_id)
    names = [axis.joint_name for axis in selected]
    if len(set(names)) != len(names):
        raise RuntimeError("Selected servos map to duplicate semantic joints.")
    return calibration, tuple(selected)


# Compatibility for the existing MuJoCo-specific name.
raw_to_mujoco_position = raw_to_joint_position
