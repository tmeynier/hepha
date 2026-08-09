#!/usr/bin/env python3
"""Find the IDs currently programmed into Feetech STS3215 servos."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a LeRobot Feetech motor bus and print every responding servo ID. "
            "The scan is read-only and does not change servo settings."
        )
    )
    parser.add_argument(
        "--port",
        help=(
            "Optional serial port to scan (for example /dev/ttyACM0). "
            "If omitted, all connected USB serial ports are detected and scanned."
        ),
    )
    return parser.parse_args()


def find_candidate_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        print(
            "PySerial is not installed.\n"
            'Install the dependencies with: python -m pip install "lerobot[feetech]"',
            file=sys.stderr,
        )
        return []

    ports = list(list_ports.comports())
    usb_ports = [
        port.device
        for port in ports
        if port.vid is not None
        or "usb" in port.device.lower()
        or "acm" in port.device.lower()
    ]
    return sorted(set(usb_ports))


def main() -> int:
    args = parse_args()

    try:
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        print(
            "LeRobot with Feetech support is not installed.\n"
            'Install it with: python -m pip install "lerobot[feetech]"',
            file=sys.stderr,
        )
        return 2

    ports = [args.port] if args.port else find_candidate_ports()
    if not ports:
        print(
            "No USB serial ports were detected.\n"
            "Connect the motor-bus adapter by USB and try again.",
            file=sys.stderr,
        )
        return 1

    print("Candidate serial port(s):")
    for port in ports:
        print(f"  {port}")

    found: list[tuple[str, int, list[int]]] = []
    for port in ports:
        print(f"\nScanning {port} (this can take a few seconds)...")
        try:
            results = FeetechMotorsBus.scan_port(port)
        except (ConnectionError, OSError) as exc:
            print(f"  Could not scan this port: {exc}")
            continue

        for baudrate, ids in sorted(results.items()):
            found.append((port, baudrate, sorted(ids)))

    if not found:
        print(
            "\nNo servos responded on any detected port.\n"
            "Check the 12 V power supply, USB connection, and 3-pin motor cable."
        )
        return 1

    print("\nDetected servo IDs:")
    total = 0
    for port, baudrate, ids in found:
        print(f"  port: {port}")
        print(f"  baud rate: {baudrate}")
        print(f"  IDs: {', '.join(map(str, ids))}")
        total += len(ids)
    print(f"\nFound {total} responding servo ID(s).")
    print(
        "To map an ID to a physical servo, connect only that servo and run this "
        "command again."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
