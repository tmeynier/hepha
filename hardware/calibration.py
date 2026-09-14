"""Shared Feetech calibration storage and bidirectional joint conversion."""

from __future__ import annotations

import json
import math
import os
import statistics
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


def _endpoint_slopes(
    axis: AxisDefinition,
    raw_min_delta: int,
    raw_max_delta: int,
) -> list[float]:
    slopes = []
    if axis.q_min != 0:
        slopes.append(raw_min_delta / axis.q_min)
    if axis.q_max != 0:
        slopes.append(raw_max_delta / axis.q_max)
    return slopes


def resolve_endpoint_deltas(
    axis: AxisDefinition,
    *,
    raw_min: int,
    raw_home: int,
    raw_max: int,
) -> tuple[int, int]:
    """Jointly unwrap endpoint deltas using their expected joint directions."""
    shortest_min = wrapped_encoder_delta(raw_min, raw_home)
    shortest_max = wrapped_encoder_delta(raw_max, raw_home)
    if axis.q_min == 0 or axis.q_max == 0:
        return shortest_min, shortest_max
    if shortest_min == 0 and shortest_max == 0:
        # Keep an all-identical capture at zero so validation can request a
        # recapture. Scoring wrapped alternatives would otherwise divide by a
        # zero mean scale or invent a full-revolution interval.
        return 0, 0

    candidates_min = tuple(
        shortest_min + turn * STEPS_PER_REVOLUTION for turn in (-1, 0, 1)
    )
    candidates_max = tuple(
        shortest_max + turn * STEPS_PER_REVOLUTION for turn in (-1, 0, 1)
    )

    def score(pair: tuple[int, int]) -> tuple[bool, bool, float, int]:
        min_delta, max_delta = pair
        slopes = _endpoint_slopes(axis, min_delta, max_delta)
        same_direction = slopes[0] * slopes[1] > 0
        fits_one_turn = abs(max_delta - min_delta) < STEPS_PER_REVOLUTION
        magnitudes = [abs(slope) for slope in slopes]
        mean_magnitude = statistics.mean(magnitudes)
        scale_difference = (
            abs(magnitudes[0] - magnitudes[1]) / mean_magnitude
            if mean_magnitude > 0
            else math.inf
        )
        return (
            not fits_one_turn,
            not same_direction,
            scale_difference,
            abs(min_delta) + abs(max_delta),
        )

    return min(
        (
            (min_delta, max_delta)
            for min_delta in candidates_min
            for max_delta in candidates_max
        ),
        key=score,
    )


def calibration_checks(
    axis: AxisDefinition,
    *,
    raw_min: int,
    raw_home: int,
    raw_max: int,
    spreads: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Return blocking calibration problems and non-blocking advisories."""
    raw_min_delta, raw_max_delta = resolve_endpoint_deltas(
        axis,
        raw_min=raw_min,
        raw_home=raw_home,
        raw_max=raw_max,
    )
    slopes = _endpoint_slopes(axis, raw_min_delta, raw_max_delta)
    problems = []
    advisories = []

    if any(spread > 4 for spread in spreads.values()):
        problems.append("an axis moved while samples were being captured")
    if axis.q_min == 0 and abs(raw_min_delta) > 4:
        problems.append("minimum and home represent q=0 but their readings differ")
    if any(abs(slope) < 1 for slope in slopes):
        problems.append("an endpoint is too close to home")
    if len(slopes) == 2 and slopes[0] * slopes[1] <= 0:
        problems.append("minimum and maximum are not on opposite sides of home")
    if len(slopes) == 2:
        magnitudes = [abs(slope) for slope in slopes]
        mean_magnitude = statistics.mean(magnitudes)
        relative_difference = (
            abs(magnitudes[0] - magnitudes[1]) / mean_magnitude
            if mean_magnitude > 0
            else 0.0
        )
        if mean_magnitude > 0 and relative_difference > 0.15:
            advisories.append("negative and positive encoder scales differ by more than 15%")
    return problems, advisories


def calibration_warnings(
    axis: AxisDefinition,
    *,
    raw_min: int,
    raw_home: int,
    raw_max: int,
    spreads: dict[str, int],
) -> list[str]:
    """Return all findings for callers that do not distinguish severity."""
    problems, advisories = calibration_checks(
        axis,
        raw_min=raw_min,
        raw_home=raw_home,
        raw_max=raw_max,
        spreads=spreads,
    )
    return problems + advisories


def build_axis_calibration(
    motor_id: int,
    axis: AxisDefinition,
    *,
    raw_min: int,
    raw_home: int,
    raw_max: int,
    spreads: dict[str, int],
) -> dict[str, object]:
    raw_min_delta, raw_max_delta = resolve_endpoint_deltas(
        axis,
        raw_min=raw_min,
        raw_home=raw_home,
        raw_max=raw_max,
    )
    slopes = _endpoint_slopes(axis, raw_min_delta, raw_max_delta)
    mean_steps_per_radian = statistics.mean(abs(slope) for slope in slopes)
    direction = 1 if statistics.mean(slopes) > 0 else -1

    return {
        "servo_id": motor_id,
        **asdict(axis),
        "joint_name": axis.joint_name,
        "q_home": 0.0,
        "raw_min": raw_min,
        "raw_home": raw_home,
        "raw_max": raw_max,
        "raw_min_delta": raw_min_delta,
        "raw_max_delta": raw_max_delta,
        "direction": direction,
        "mean_steps_per_radian": mean_steps_per_radian,
        "sample_spread": spreads,
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
