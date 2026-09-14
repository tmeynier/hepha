"""Reusable calibrated Feetech leader for MuJoCo teleoperation and recording."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .calibration import (
        CalibratedAxis,
        load_calibrated_axes,
        raw_to_joint_position,
    )
    from .feetech_bus import connect_and_ping, create_feetech_bus
except ImportError:  # Direct hardware script execution.
    from calibration import CalibratedAxis, load_calibrated_axes, raw_to_joint_position
    from feetech_bus import connect_and_ping, create_feetech_bus

CalibratedServo = CalibratedAxis


def load_calibrated_servos(
    calibration_path: Path,
    servo_ids: Sequence[int] | None,
) -> tuple[dict[str, object], tuple[CalibratedServo, ...]]:
    """Load selected calibrated axes, defaulting to every saved servo ID."""
    return load_calibrated_axes(calibration_path, servo_ids=servo_ids)


def configure_cnc_start_pose(
    action: np.ndarray,
    control_low: np.ndarray,
    actuator_names: Sequence[str],
) -> np.ndarray:
    """Set the fixed CNC pose used by physical-leader teleoperation."""
    action[actuator_names.index("cnc_x")] = 0.0
    action[actuator_names.index("cnc_y")] = 0.0
    head_z_index = actuator_names.index("head_z")
    action[head_z_index] = control_low[head_z_index]
    return action


def compose_action(
    base_action: np.ndarray,
    targets: dict[str, float],
    actuator_names: Sequence[str],
) -> np.ndarray:
    """Overlay calibrated leader targets on a complete MuJoCo action."""
    action = np.asarray(base_action, dtype=float).copy()
    for actuator, target in targets.items():
        try:
            index = actuator_names.index(actuator)
        except ValueError as exc:
            raise RuntimeError(f"MuJoCo actuator {actuator!r} does not exist.") from exc
        action[index] = target
    return action


class FeetechLeader:
    """Read-only physical leader whose selected servos have torque disabled."""

    def __init__(
        self,
        *,
        calibration_path: Path,
        servo_ids: Sequence[int] | None = None,
        joint_names: Sequence[str] | None = None,
        port: str | None = None,
        retries: int = 2,
        smoothing: float = 0.25,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be non-negative")
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be greater than 0 and at most 1")
        calibration, servos = load_calibrated_axes(
            calibration_path,
            servo_ids=servo_ids,
            joint_names=joint_names,
        )
        try:
            self.port = port or str(calibration["port"])
            self.baudrate = int(calibration["baudrate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Calibration has invalid bus metadata: {exc}") from exc
        self.calibration_path = calibration_path
        self.servos = servos
        self.retries = retries
        self.smoothing = smoothing
        self.motor_names = {servo.servo_id: f"servo_{servo.servo_id}" for servo in self.servos}
        self.bus: Any | None = None
        self.last_raw_positions: dict[int, int] = {}
        self.last_targets: dict[str, float] = {}

    @property
    def servo_ids(self) -> tuple[int, ...]:
        return tuple(servo.servo_id for servo in self.servos)

    def validate_mujoco_ranges(
        self,
        actuator_names: Sequence[str],
        control_low: np.ndarray,
        control_high: np.ndarray,
    ) -> None:
        """Reject stale calibration whose actuator or limits differ from MuJoCo."""
        for servo in self.servos:
            try:
                index = actuator_names.index(servo.actuator)
            except ValueError as exc:
                raise RuntimeError(
                    f"Calibration maps ID {servo.servo_id} to missing actuator {servo.actuator!r}."
                ) from exc
            q_min = float(servo.calibration["q_min"])
            q_max = float(servo.calibration["q_max"])
            if not np.allclose(
                (q_min, q_max),
                (control_low[index], control_high[index]),
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"Calibration limits for ID {servo.servo_id} ({q_min:+.6f}, "
                    f"{q_max:+.6f}) do not match MuJoCo ({control_low[index]:+.6f}, "
                    f"{control_high[index]:+.6f}); recalibrate this axis."
                )

    def connect(self) -> None:
        if self.bus is not None:
            raise RuntimeError("Feetech leader is already connected.")
        bus = create_feetech_bus(self.port, self.servo_ids)
        connect_and_ping(
            bus,
            baudrate=self.baudrate,
            motor_ids=self.servo_ids,
            retries=self.retries,
        )
        self.bus = bus

    def disable_torque(self) -> None:
        if self.bus is None:
            raise RuntimeError("Feetech leader is not connected.")
        self.bus.disable_torque(list(self.motor_names.values()), num_retry=self.retries)

    def reset_filter(self) -> None:
        """Make the next read use the physical pose without prior-episode smoothing."""
        self.last_raw_positions = {}
        self.last_targets = {}

    def read_targets(self) -> dict[str, float]:
        if self.bus is None:
            raise RuntimeError("Feetech leader is not connected.")
        positions = self.bus.sync_read(
            "Present_Position",
            list(self.motor_names.values()),
            normalize=False,
            num_retry=self.retries,
        )
        targets: dict[str, float] = {}
        raw_positions: dict[int, int] = {}
        for servo in self.servos:
            raw = int(positions[self.motor_names[servo.servo_id]])
            target = raw_to_joint_position(raw, servo.calibration)
            previous = self.last_targets.get(servo.actuator)
            if previous is not None:
                target = previous + self.smoothing * (target - previous)
            raw_positions[servo.servo_id] = raw
            targets[servo.actuator] = target
        self.last_raw_positions = raw_positions
        self.last_targets = targets
        return targets.copy()

    def read_joint_positions(self) -> dict[str, float]:
        """Return semantic joint positions in radians."""
        return self.read_targets()

    def read_action(
        self,
        base_action: np.ndarray,
        actuator_names: Sequence[str],
    ) -> np.ndarray:
        return compose_action(base_action, self.read_targets(), actuator_names)

    def disconnect(self) -> None:
        if self.bus is not None:
            if self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)
            self.bus = None

    def __enter__(self) -> FeetechLeader:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
