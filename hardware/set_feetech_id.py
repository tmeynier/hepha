#!/usr/bin/env python3
"""Set the ID of one Feetech STS3215 servo using LeRobot."""

from __future__ import annotations

import argparse
import sys

from scan_feetech_ids import find_candidate_ports


def servo_id(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 252:
        raise argparse.ArgumentTypeError("ID must be between 1 and 252")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find one connected STS3215 and change its stored ID."
    )
    parser.add_argument("new_id", type=servo_id, help="New servo ID (1-252)")
    parser.add_argument(
        "--port",
        help="USB serial port. If omitted, one unambiguous USB port is detected.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the safety confirmation prompt.",
    )
    return parser.parse_args()


def select_port(requested_port: str | None) -> str:
    if requested_port:
        return requested_port

    ports = find_candidate_ports()
    if not ports:
        raise RuntimeError("No USB serial adapter was detected.")
    if len(ports) > 1:
        choices = "\n".join(f"  {port}" for port in ports)
        raise RuntimeError(
            "More than one USB serial port was detected. Choose the motor adapter "
            f"with --port:\n{choices}"
        )
    return ports[0]


def main() -> int:
    args = parse_args()

    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        print(
            'Install the dependencies with: python -m pip install "lerobot[feetech]"',
            file=sys.stderr,
        )
        return 2

    try:
        port = select_port(args.port)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Port: {port}")
    print(f"New servo ID: {args.new_id}")
    print("WARNING: exactly ONE servo must be connected to the motor bus.")
    if not args.yes:
        answer = input("Type YES to write the new ID: ")
        if answer != "YES":
            print("Cancelled; no settings were changed.")
            return 1

    print("\nChecking how many servo IDs respond...")
    try:
        scan_results = FeetechMotorsBus.scan_port(port)
    except Exception as exc:
        print(f"Could not scan the motor bus: {exc}", file=sys.stderr)
        return 1
    responding_ids = {
        found_id for ids_at_baudrate in scan_results.values() for found_id in ids_at_baudrate
    }
    if len(responding_ids) != 1:
        print(
            "Refusing to write: expected exactly one responding servo ID, "
            f"but found {sorted(responding_ids)}.\n"
            "Power off, connect exactly one servo, and try again.",
            file=sys.stderr,
        )
        return 1

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "servo": Motor(
                id=args.new_id,
                model="sts3215",
                norm_mode=MotorNormMode.RANGE_M100_100,
            )
        },
    )

    try:
        # LeRobot finds the servo's current ID/baud rate, disables torque,
        # writes the requested ID, and sets the standard 1,000,000 baud rate.
        bus.setup_motor("servo")
        model_number = bus.ping(args.new_id)
        if model_number is None:
            raise RuntimeError(
                f"The write completed, but servo ID {args.new_id} did not respond."
            )
    except Exception as exc:
        print(f"\nCould not set the servo ID: {exc}", file=sys.stderr)
        return 1
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

    print(f"\nSuccess: the STS3215 now responds as ID {args.new_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
