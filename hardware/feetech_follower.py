"""Safely command a calibrated Feetech follower arm in joint space."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from .calibration import (
        joint_to_raw_position,
        load_calibrated_axes,
        raw_to_joint_position,
    )
    from .feetech_bus import connect_and_ping, create_feetech_bus
except ImportError:  # Direct hardware script execution.
    from calibration import (
        joint_to_raw_position,
        load_calibrated_axes,
        raw_to_joint_position,
    )
    from feetech_bus import connect_and_ping, create_feetech_bus


class FeetechFollower:
    """Position-controlled follower with explicit seeding and torque arming."""

    def __init__(
        self,
        *,
        calibration_path: Path,
        servo_ids: Sequence[int] | None = None,
        joint_names: Sequence[str] | None = None,
        port: str | None = None,
        retries: int = 2,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be non-negative")
        calibration, axes = load_calibrated_axes(
            calibration_path,
            servo_ids=servo_ids,
            joint_names=joint_names,
        )
        if calibration.get("role") != "follower":
            raise RuntimeError(
                f"{calibration_path} is not a follower calibration. Run "
                "hardware/calibrate_feetech_follower.py for this arm."
            )
        try:
            self.port = port or str(calibration["port"])
            self.baudrate = int(calibration["baudrate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Follower calibration has invalid bus metadata: {exc}") from exc
        self.calibration_path = calibration_path
        self.axes = axes
        self.retries = retries
        self.motor_names = {axis.servo_id: f"servo_{axis.servo_id}" for axis in axes}
        self.bus: Any | None = None
        self.goals_seeded = False
        self.torque_enabled = False
        self.last_raw_positions: dict[int, int] = {}
        self.last_raw_goals: dict[int, int] = {}
        self.last_joint_goals: dict[str, float] = {}
        self._validate_saved_configuration()

    @property
    def servo_ids(self) -> tuple[int, ...]:
        return tuple(axis.servo_id for axis in self.axes)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(axis.joint_name for axis in self.axes)

    def _validate_saved_configuration(self) -> None:
        for axis in self.axes:
            data = axis.calibration
            missing = [
                field
                for field in ("homing_offset", "hardware_min", "hardware_max")
                if field not in data
            ]
            if missing:
                raise RuntimeError(
                    f"Follower ID {axis.servo_id} lacks commissioned fields: {missing}"
                )
            raw_values = (
                int(data["raw_min"]),
                int(data["raw_home"]),
                int(data["raw_max"]),
            )
            hardware_min = int(data["hardware_min"])
            hardware_max = int(data["hardware_max"])
            if not 0 <= hardware_min < hardware_max <= 4095:
                raise RuntimeError(f"Follower ID {axis.servo_id} has invalid hardware limits.")
            if not all(hardware_min <= raw <= hardware_max for raw in raw_values):
                raise RuntimeError(
                    f"Follower ID {axis.servo_id} calibration lies outside hardware limits."
                )

    def connect(self) -> None:
        if self.bus is not None:
            raise RuntimeError("Feetech follower is already connected.")
        bus = create_feetech_bus(self.port, self.servo_ids)
        connect_and_ping(
            bus,
            baudrate=self.baudrate,
            motor_ids=self.servo_ids,
            retries=self.retries,
        )
        self.bus = bus
        try:
            self.verify_servo_configuration()
        except Exception:
            self.disconnect()
            raise

    def verify_servo_configuration(self) -> None:
        """Ensure persistent position mode, offset, and hardware limits match the file."""
        if self.bus is None:
            raise RuntimeError("Feetech follower is not connected.")
        for axis in self.axes:
            motor = self.motor_names[axis.servo_id]
            expected = axis.calibration
            actual_mode = int(
                self.bus.read("Operating_Mode", motor, normalize=False, num_retry=self.retries)
            )
            actual_offset = int(
                self.bus.read("Homing_Offset", motor, normalize=False, num_retry=self.retries)
            )
            actual_min = int(
                self.bus.read("Min_Position_Limit", motor, normalize=False, num_retry=self.retries)
            )
            actual_max = int(
                self.bus.read("Max_Position_Limit", motor, normalize=False, num_retry=self.retries)
            )
            mismatches = []
            if actual_mode != 0:
                mismatches.append(f"mode={actual_mode}, expected 0")
            if actual_offset != int(expected["homing_offset"]):
                mismatches.append(
                    f"homing offset={actual_offset}, expected {int(expected['homing_offset'])}"
                )
            if actual_min != int(expected["hardware_min"]):
                mismatches.append(f"minimum={actual_min}, expected {int(expected['hardware_min'])}")
            if actual_max != int(expected["hardware_max"]):
                mismatches.append(f"maximum={actual_max}, expected {int(expected['hardware_max'])}")
            if mismatches:
                details = "; ".join(mismatches)
                raise RuntimeError(
                    f"Follower ID {axis.servo_id} does not match its calibration: {details}. "
                    "Re-run complete follower calibration."
                )

    def disable_torque(self) -> None:
        if self.bus is None:
            raise RuntimeError("Feetech follower is not connected.")
        self.bus.disable_torque(list(self.motor_names.values()), num_retry=self.retries)
        self.torque_enabled = False
        self.goals_seeded = False

    def configure_runtime(self, *, acceleration: int, torque_limit: int) -> None:
        if self.bus is None:
            raise RuntimeError("Feetech follower is not connected.")
        if not 0 <= acceleration <= 254:
            raise ValueError("acceleration must be between 0 and 254")
        if not 0 <= torque_limit <= 1000:
            raise ValueError("torque limit must be between 0 and 1000")
        names = list(self.motor_names.values())
        self.bus.sync_write(
            "Acceleration",
            {name: acceleration for name in names},
            normalize=False,
            num_retry=self.retries,
        )
        self.bus.sync_write(
            "Torque_Limit",
            {name: torque_limit for name in names},
            normalize=False,
            num_retry=self.retries,
        )

    def read_joint_positions(self) -> dict[str, float]:
        if self.bus is None:
            raise RuntimeError("Feetech follower is not connected.")
        positions = self.bus.sync_read(
            "Present_Position",
            list(self.motor_names.values()),
            normalize=False,
            num_retry=self.retries,
        )
        result = {}
        raw_positions = {}
        for axis in self.axes:
            raw = int(positions[self.motor_names[axis.servo_id]])
            raw_positions[axis.servo_id] = raw
            result[axis.joint_name] = raw_to_joint_position(raw, axis.calibration)
        self.last_raw_positions = raw_positions
        return result

    def seed_goals_from_present_position(self) -> dict[str, float]:
        """Set every goal to its current pose before torque can be enabled."""
        positions = self.read_joint_positions()
        assert self.bus is not None
        raw_goals = {
            self.motor_names[axis.servo_id]: self.last_raw_positions[axis.servo_id]
            for axis in self.axes
        }
        self.bus.sync_write(
            "Goal_Position",
            raw_goals,
            normalize=False,
            num_retry=self.retries,
        )
        self.last_raw_goals = {
            axis.servo_id: raw_goals[self.motor_names[axis.servo_id]] for axis in self.axes
        }
        self.last_joint_goals = positions.copy()
        self.goals_seeded = True
        return positions

    def enable_torque(self) -> None:
        if self.bus is None:
            raise RuntimeError("Feetech follower is not connected.")
        if not self.goals_seeded:
            raise RuntimeError("Seed follower goals from the present pose before enabling torque.")
        # Mark before the write so cleanup still attempts to disable torque if a
        # multi-servo enable operation succeeds only partially and then raises.
        self.torque_enabled = True
        self.bus.enable_torque(list(self.motor_names.values()), num_retry=self.retries)

    def write_joint_positions(self, targets: dict[str, float]) -> dict[str, float]:
        if self.bus is None:
            raise RuntimeError("Feetech follower is not connected.")
        if not self.torque_enabled:
            raise RuntimeError("Follower torque is not enabled.")
        missing = sorted(set(self.joint_names) - set(targets))
        if missing:
            raise ValueError(f"Follower command is missing joints: {missing}")

        raw_by_name = {}
        sent = {}
        raw_by_id = {}
        for axis in self.axes:
            q_min = float(axis.calibration["q_min"])
            q_max = float(axis.calibration["q_max"])
            q = min(q_max, max(q_min, float(targets[axis.joint_name])))
            raw = joint_to_raw_position(q, axis.calibration)
            raw_by_name[self.motor_names[axis.servo_id]] = raw
            raw_by_id[axis.servo_id] = raw
            sent[axis.joint_name] = q
        self.bus.sync_write(
            "Goal_Position",
            raw_by_name,
            normalize=False,
            num_retry=self.retries,
        )
        self.last_raw_goals = raw_by_id
        self.last_joint_goals = sent
        return sent.copy()

    def disconnect(self) -> None:
        if self.bus is None:
            return
        bus = self.bus
        self.bus = None
        try:
            if bus.is_connected and self.torque_enabled:
                bus.disable_torque(list(self.motor_names.values()), num_retry=self.retries)
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
            self.torque_enabled = False
            self.goals_seeded = False

    def __enter__(self) -> FeetechFollower:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
