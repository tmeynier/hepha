#!/usr/bin/env python3
"""Continuously print positions from STS3215 servos on a Feetech bus."""

from __future__ import annotations

import argparse
import sys
import time

from scan_feetech_ids import find_candidate_ports


STEPS_PER_REVOLUTION = 4096


def positive_rate(value: str) -> float:
    rate = float(value)
    if rate <= 0:
        raise argparse.ArgumentTypeError("rate must be greater than zero")
    return rate


def servo_id(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 252:
        raise argparse.ArgumentTypeError("servo IDs must be between 1 and 252")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously read all detected Feetech STS3215 positions."
    )
    parser.add_argument(
        "--port",
        help="USB serial port. If omitted, connected USB serial ports are probed.",
    )
    parser.add_argument(
        "--ids",
        type=servo_id,
        nargs="+",
        help="Expected servo IDs, for example: --ids 1 3",
    )
    parser.add_argument(
        "--rate",
        type=positive_rate,
        default=10.0,
        help="Reads per second (default: 10).",
    )
    return parser.parse_args()


def scan_for_bus(bus_class: type, requested_port: str | None) -> tuple[str, int, list[int]]:
    ports = [requested_port] if requested_port else find_candidate_ports()
    if not ports:
        raise RuntimeError("No USB serial ports were detected.")

    matches: list[tuple[str, int, list[int]]] = []
    for port in ports:
        print(f"Scanning {port}...", file=sys.stderr)
        try:
            results = bus_class.scan_port(port)
        except Exception as exc:
            print(f"  Skipped {port}: {exc}", file=sys.stderr)
            continue
        for baudrate, ids in results.items():
            if ids:
                matches.append((port, baudrate, sorted(ids)))

    if not matches:
        raise RuntimeError(
            "No servos responded. Check 12 V power, USB, and the 3-pin bus cables."
        )
    if len(matches) > 1:
        descriptions = ", ".join(
            f"{port} at {baudrate} baud (IDs {ids})"
            for port, baudrate, ids in matches
        )
        raise RuntimeError(
            f"More than one responding motor bus was found: {descriptions}. "
            "Select one with --port."
        )
    return matches[0]


def main() -> int:
    args = parse_args()

    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        print(
            'Install dependencies with: python -m pip install "lerobot[feetech]"',
            file=sys.stderr,
        )
        return 2

    try:
        port, baudrate, detected_ids = scan_for_bus(FeetechMotorsBus, args.port)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.ids:
        expected_ids = sorted(set(args.ids))
        missing = sorted(set(expected_ids) - set(detected_ids))
        unexpected = sorted(set(detected_ids) - set(expected_ids))
        if missing or unexpected:
            print(
                f"ID check failed: expected {expected_ids}, detected {detected_ids}."
                + (f" Missing: {missing}." if missing else "")
                + (f" Unexpected: {unexpected}." if unexpected else ""),
                file=sys.stderr,
            )
            return 1
        ids = expected_ids
    else:
        ids = detected_ids

    motors = {
        f"servo_{motor_id}": Motor(
            id=motor_id,
            model="sts3215",
            norm_mode=MotorNormMode.DEGREES,
        )
        for motor_id in ids
    }
    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        bus.connect(handshake=False)
        bus.set_baudrate(baudrate)
        for motor_id in ids:
            if bus.ping(motor_id) is None:
                raise RuntimeError(f"Servo ID {motor_id} stopped responding.")

        print(f"Reading IDs {ids} on {port} at {baudrate} baud.")
        print("Press Ctrl+C to stop.")
        print("time_s," + ",".join(f"id_{motor_id}_raw,id_{motor_id}_deg" for motor_id in ids))

        period = 1.0 / args.rate
        start = time.monotonic()
        next_read = start
        while True:
            positions = bus.sync_read("Present_Position", normalize=False)
            elapsed = time.monotonic() - start
            values = []
            for motor_id in ids:
                raw = int(positions[f"servo_{motor_id}"])
                degrees = raw * 360.0 / STEPS_PER_REVOLUTION
                values.extend((str(raw), f"{degrees:.3f}"))
            print(f"{elapsed:.3f}," + ",".join(values), flush=True)

            next_read += period
            time.sleep(max(0.0, next_read - time.monotonic()))
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        print(f"\nPosition reading failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if bus.is_connected:
            # This monitor never enables torque, so closing the port is sufficient.
            bus.disconnect(disable_torque=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
