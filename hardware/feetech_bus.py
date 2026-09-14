"""Small shared helpers for constructing and validating Feetech buses."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def create_feetech_bus(port: str, motor_ids: Iterable[int]) -> Any:
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError as exc:
        raise RuntimeError(
            "Install Feetech support with: .venv/bin/python -m pip install "
            "'lerobot[feetech]==0.6.1'"
        ) from exc

    motors = {
        f"servo_{motor_id}": Motor(
            id=motor_id,
            model="sts3215",
            norm_mode=MotorNormMode.DEGREES,
        )
        for motor_id in motor_ids
    }
    if not motors:
        raise ValueError("A Feetech bus needs at least one selected motor.")
    return FeetechMotorsBus(port=port, motors=motors)


def connect_and_ping(
    bus: Any,
    *,
    baudrate: int,
    motor_ids: Iterable[int],
    retries: int,
) -> None:
    """Connect without modifying torque and verify every selected motor."""
    try:
        bus.connect(handshake=False)
        bus.set_baudrate(baudrate)
        for motor_id in motor_ids:
            if bus.ping(motor_id, num_retry=retries) is None:
                raise RuntimeError(f"Servo ID {motor_id} did not respond.")
    except Exception:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        raise

