#!/usr/bin/env python3
"""Continuously print positions from STS3215 servos on a Feetech bus."""

from __future__ import annotations

import argparse
import signal
import sys
import time

try:
    from .axes import SERVO_LABELS
    from .calibration import STEPS_PER_REVOLUTION
    from .scan_feetech_ids import find_candidate_ports
except ImportError:  # Direct execution: python hardware/read_feetech_positions.py
    from axes import SERVO_LABELS
    from calibration import STEPS_PER_REVOLUTION
    from scan_feetech_ids import find_candidate_ports


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


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("retries cannot be negative")
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
    parser.add_argument(
        "--retries",
        type=nonnegative_integer,
        default=2,
        help="Retries after a failed serial read (default: 2).",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("table", "csv"),
        default="table",
        help="Terminal output format (default: table).",
    )
    return parser.parse_args()


def position_degrees(raw: int) -> float:
    return raw * 360.0 / STEPS_PER_REVOLUTION


def format_live_table(
    *,
    port: str,
    baudrate: int,
    elapsed: float,
    ids: list[int],
    raw_positions: dict[int, int],
) -> str:
    rows = [
        "Feetech STS3215 bus monitor",
        f"Port: {port}",
        f"Baud: {baudrate:,}   Servos: {len(ids)}   Elapsed: {elapsed:8.1f} s",
        "",
        " ID  Robot element          Raw   Position",
        "---  ------------------  ------  ---------",
    ]
    for motor_id in ids:
        raw = raw_positions[motor_id]
        label = SERVO_LABELS.get(motor_id, "unmapped servo")
        rows.append(f"{motor_id:>3}  {label:<18}  {raw:>6}  {position_degrees(raw):>8.2f}°")
    rows.extend(("", "Press Ctrl+C to stop. Use --format csv for logging."))
    return "\n".join(rows)


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
            "No servos responded. Check the correct external servo power supply, "
            "USB, and the 3-pin bus cables."
        )
    if len(matches) > 1:
        descriptions = ", ".join(
            f"{port} at {baudrate} baud (IDs {ids})" for port, baudrate, ids in matches
        )
        raise RuntimeError(
            f"More than one responding motor bus was found: {descriptions}. Select one with --port."
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
            if bus.ping(motor_id, num_retry=args.retries) is None:
                raise RuntimeError(f"Servo ID {motor_id} stopped responding.")

        print(f"Reading IDs {ids} on {port} at {baudrate} baud.")
        if args.output_format == "csv":
            print("Press Ctrl+C to stop.")
            print("time_s," + ",".join(f"id_{motor_id}_raw,id_{motor_id}_deg" for motor_id in ids))

        period = 1.0 / args.rate
        start = time.monotonic()
        next_read = start
        while True:
            positions = bus.sync_read("Present_Position", normalize=False, num_retry=args.retries)
            elapsed = time.monotonic() - start
            raw_positions = {motor_id: int(positions[f"servo_{motor_id}"]) for motor_id in ids}
            if args.output_format == "table":
                table = format_live_table(
                    port=port,
                    baudrate=baudrate,
                    elapsed=elapsed,
                    ids=ids,
                    raw_positions=raw_positions,
                )
                print(f"\033[2J\033[H{table}", end="", flush=True)
            else:
                values = []
                for motor_id in ids:
                    raw = raw_positions[motor_id]
                    values.extend((str(raw), f"{position_degrees(raw):.3f}"))
                print(f"{elapsed:.3f}," + ",".join(values), flush=True)

            next_read += period
            time.sleep(max(0.0, next_read - time.monotonic()))
    except KeyboardInterrupt:
        # Ignore repeated Ctrl+C presses while the serial port is being closed.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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
