#!/usr/bin/env python3
"""Calibrate one complete Hepha arm by recording a simultaneous range sweep."""

from __future__ import annotations

import argparse
import select
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

try:
    from .axes import AXES
    from .calibration import (
        STEPS_PER_REVOLUTION,
        build_range_calibration,
        save_calibration,
        wrapped_encoder_delta,
    )
    from .feetech_bus import connect_and_ping, create_feetech_bus
    from .read_feetech_positions import scan_for_bus
except ImportError:  # Direct execution: python hardware/calibrate_feetech.py
    from axes import AXES
    from calibration import (
        STEPS_PER_REVOLUTION,
        build_range_calibration,
        save_calibration,
        wrapped_encoder_delta,
    )
    from feetech_bus import connect_and_ping, create_feetech_bus
    from read_feetech_positions import scan_for_bus

Role = Literal["leader", "follower"]
ALL_SERVO_IDS = tuple(sorted(AXES))
DEFAULT_OUTPUTS = {
    "leader": Path("hardware/feetech_calibration.json"),
    "follower": Path("hardware/feetech_follower_calibration.json"),
}


@dataclass
class RangeTracker:
    """Continuously unwrap one cyclic encoder and retain its observed extrema."""

    previous_raw: int | None = None
    unwrapped: int = 0
    minimum: int = 0
    maximum: int = 0
    samples: int = 0

    def observe(self, raw: int) -> None:
        raw = int(raw) % STEPS_PER_REVOLUTION
        if self.previous_raw is None:
            self.unwrapped = raw
            self.minimum = raw
            self.maximum = raw
        else:
            self.unwrapped += wrapped_encoder_delta(raw, self.previous_raw)
            self.minimum = min(self.minimum, self.unwrapped)
            self.maximum = max(self.maximum, self.unwrapped)
        self.previous_raw = raw
        self.samples += 1

    @property
    def span(self) -> int:
        return self.maximum - self.minimum


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(
    argv: list[str] | None = None,
    *,
    fixed_role: Role | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_role is None:
        parser.add_argument("--role", choices=("leader", "follower"), required=True)
    parser.add_argument(
        "--port",
        help="USB serial port. If omitted, connected USB serial ports are scanned.",
    )
    parser.add_argument("--baudrate", type=_positive_integer, default=1_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        help="Calibration JSON path; defaults to the selected role's standard file.",
    )
    parser.add_argument(
        "--frequency",
        type=_positive_float,
        default=20.0,
        help="Maximum range-sampling frequency in Hz (default: 20).",
    )
    parser.add_argument(
        "--minimum-span",
        type=_positive_integer,
        default=100,
        help="Minimum required encoder span for every servo (default: 100 steps).",
    )
    parser.add_argument("--retries", type=int, choices=range(0, 11), default=2)
    args = parser.parse_args(argv)
    if fixed_role is not None:
        args.role = fixed_role
    if args.output is None:
        args.output = DEFAULT_OUTPUTS[args.role]
    return args


def resolve_complete_bus(
    bus_class: type,
    *,
    port: str | None,
    baudrate: int,
    retries: int,
) -> tuple[str, int]:
    """Use a known port directly or discover exactly one complete Hepha bus."""

    if port:
        return port, baudrate

    discovered_port, discovered_baudrate, detected_ids = scan_for_bus(bus_class, None)
    missing = sorted(set(ALL_SERVO_IDS) - set(detected_ids))
    unexpected = sorted(set(detected_ids) - set(ALL_SERVO_IDS))
    if missing or unexpected:
        raise RuntimeError(
            "Calibration requires exactly Hepha servo IDs 1-12; "
            f"missing={missing}, unexpected={unexpected}."
        )
    return discovered_port, discovered_baudrate


def _enter_pressed() -> bool:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def record_ranges(
    bus: object,
    *,
    frequency: float,
    retries: int,
) -> dict[int, RangeTracker]:
    """Sample all servos together until Enter is pressed."""

    names = [f"servo_{motor_id}" for motor_id in ALL_SERVO_IDS]
    trackers = {motor_id: RangeTracker() for motor_id in ALL_SERVO_IDS}
    period = 1.0 / frequency
    started = time.monotonic()
    next_report = started + 1.0
    while True:
        sample_started = time.monotonic()
        positions = bus.sync_read(
            "Present_Position",
            names,
            normalize=False,
            num_retry=retries,
        )
        for motor_id in ALL_SERVO_IDS:
            trackers[motor_id].observe(int(positions[f"servo_{motor_id}"]))
        if _enter_pressed():
            break
        now = time.monotonic()
        if now >= next_report:
            elapsed = now - started
            smallest_span = min(tracker.span for tracker in trackers.values())
            print(
                f"\rRecording ranges: {elapsed:6.1f} s | "
                f"smallest observed span: {smallest_span:4d} steps",
                end="",
                flush=True,
            )
            next_report = now + 1.0
        time.sleep(max(0.0, period - (time.monotonic() - sample_started)))
    print()
    return trackers


def build_complete_calibration(
    *,
    role: Role,
    port: str,
    baudrate: int,
    trackers: dict[int, RangeTracker],
    minimum_span: int,
) -> dict[str, object]:
    """Validate a full sweep and construct the replacement calibration document."""

    insufficient = [
        motor_id
        for motor_id in ALL_SERVO_IDS
        if trackers[motor_id].span < minimum_span
    ]
    if insufficient:
        details = ", ".join(
            f"ID {motor_id}: {trackers[motor_id].span}" for motor_id in insufficient
        )
        raise RuntimeError(
            f"No calibration was saved. Move every servo through its complete range; "
            f"spans below {minimum_span} steps: {details}."
        )

    timestamp = datetime.now(UTC).isoformat()
    axes = {
        str(motor_id): build_range_calibration(
            motor_id,
            AXES[motor_id],
            encoder_low=trackers[motor_id].minimum,
            encoder_high=trackers[motor_id].maximum,
            sample_count=trackers[motor_id].samples,
        )
        for motor_id in ALL_SERVO_IDS
    }
    return {
        "schema_version": 1,
        "calibration_method": "simultaneous_range_sweep",
        "created_at": timestamp,
        "updated_at": timestamp,
        "motor_model": "sts3215",
        "encoder_resolution": STEPS_PER_REVOLUTION,
        "port": port,
        "baudrate": baudrate,
        "role": role,
        "axes": axes,
    }


def _centered_raw(raw: int, shift: int) -> int:
    return (int(raw) - shift) % STEPS_PER_REVOLUTION


def commission_follower(
    bus: object,
    calibration: dict[str, object],
    *,
    retries: int,
) -> None:
    """Center every follower midpoint and install non-wrapping hardware limits."""

    axes = calibration["axes"]
    assert isinstance(axes, dict)
    plans: list[tuple[str, int, int, int]] = []
    for motor_id in ALL_SERVO_IDS:
        motor = f"servo_{motor_id}"
        axis = axes[str(motor_id)]
        assert isinstance(axis, dict)
        old_offset = int(
            bus.read("Homing_Offset", motor, normalize=False, num_retry=retries)
        )
        raw_home = int(axis["raw_home"])
        shift = wrapped_encoder_delta(raw_home, STEPS_PER_REVOLUTION // 2)
        new_offset = wrapped_encoder_delta(old_offset + shift, 0)
        centered_values = {
            field: _centered_raw(int(axis[field]), shift)
            for field in ("raw_min", "raw_home", "raw_max")
        }
        hardware_min = min(centered_values.values())
        hardware_max = max(centered_values.values())
        if not 0 <= hardware_min < hardware_max < STEPS_PER_REVOLUTION:
            raise RuntimeError(
                f"ID {motor_id} does not form a safe non-wrapping follower interval."
            )
        axis.update(centered_values)
        axis.update(
            homing_offset=new_offset,
            hardware_min=hardware_min,
            hardware_max=hardware_max,
            operating_mode=0,
        )
        plans.append((motor, new_offset, hardware_min, hardware_max))

    for motor, new_offset, hardware_min, hardware_max in plans:
        bus.write("Min_Position_Limit", motor, 0, normalize=False, num_retry=retries)
        bus.write(
            "Max_Position_Limit",
            motor,
            STEPS_PER_REVOLUTION - 1,
            normalize=False,
            num_retry=retries,
        )
        bus.write("Operating_Mode", motor, 0, normalize=False, num_retry=retries)
        bus.write("Homing_Offset", motor, new_offset, normalize=False, num_retry=retries)
        bus.write(
            "Min_Position_Limit",
            motor,
            hardware_min,
            normalize=False,
            num_retry=retries,
        )
        bus.write(
            "Max_Position_Limit",
            motor,
            hardware_max,
            normalize=False,
            num_retry=retries,
        )


def _print_summary(calibration: dict[str, object]) -> None:
    axes = calibration["axes"]
    assert isinstance(axes, dict)
    print("\n ID  Robot element        Raw MIN  Raw HOME  Raw MAX  Span")
    print("---  ------------------  -------  --------  -------  ----")
    for motor_id in ALL_SERVO_IDS:
        axis = axes[str(motor_id)]
        assert isinstance(axis, dict)
        print(
            f"{motor_id:>3}  {axis['label']!s:<18}  "
            f"{int(axis['raw_min']):>7}  {int(axis['raw_home']):>8}  "
            f"{int(axis['raw_max']):>7}  {int(axis['encoder_span']):>4}"
        )


def run(role: Role, argv: list[str] | None = None) -> int:
    args = parse_args(argv, fixed_role=role)
    try:
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        print(
            'Install dependencies with: .venv/bin/python -m pip install "lerobot[feetech]==0.6.1"',
            file=sys.stderr,
        )
        return 2

    bus = None
    try:
        port, baudrate = resolve_complete_bus(
            FeetechMotorsBus,
            port=args.port,
            baudrate=args.baudrate,
            retries=args.retries,
        )
        bus = create_feetech_bus(port, ALL_SERVO_IDS)
        connect_and_ping(
            bus,
            baudrate=baudrate,
            motor_ids=ALL_SERVO_IDS,
            retries=args.retries,
        )
        names = [f"servo_{motor_id}" for motor_id in ALL_SERVO_IDS]
        print(f"Hepha {role} simultaneous range calibration")
        print(f"Port: {port}   Baud: {baudrate:,}   IDs: 1-12")
        print("Support both arms: torque on all 12 servos is being disabled.")
        bus.disable_torque(names, num_retry=args.retries)
        print("Move every servo through its complete physical range.")
        print("Recording starts immediately. Press ENTER once every range is covered.")
        trackers = record_ranges(bus, frequency=args.frequency, retries=args.retries)
        calibration = build_complete_calibration(
            role=role,
            port=port,
            baudrate=baudrate,
            trackers=trackers,
            minimum_span=args.minimum_span,
        )
        if role == "follower":
            print("Centering follower midpoints and installing hardware limits...")
            commission_follower(bus, calibration, retries=args.retries)
        save_calibration(args.output, calibration)
        _print_summary(calibration)
        print(f"\nCalibration complete: {args.output}")
        return 0
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nCalibration cancelled; the existing calibration file was not changed.")
        return 130
    except Exception as exc:
        print(f"\nCalibration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    role: Role = args.role
    forwarded = list(argv if argv is not None else sys.argv[1:])
    for index, argument in enumerate(forwarded):
        if argument == "--role":
            del forwarded[index : index + 2]
            break
        if argument.startswith("--role="):
            del forwarded[index]
            break
    return run(role, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
