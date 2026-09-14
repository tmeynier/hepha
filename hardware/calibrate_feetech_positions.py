#!/usr/bin/env python3
"""Interactively capture min, home, and max positions for each Hepha servo."""

from __future__ import annotations

import argparse
import math
import signal
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    from .axes import AXES, AxisDefinition
    from .calibration import (
        STEPS_PER_REVOLUTION,
        build_axis_calibration,
        calibration_checks,
        calibration_warnings,
        load_calibration,
        raw_to_mujoco_position,
        resolve_endpoint_deltas,
        save_calibration,
        wrapped_encoder_delta,
    )
    from .feetech_bus import connect_and_ping, create_feetech_bus
    from .read_feetech_positions import (
        nonnegative_integer,
        scan_for_bus,
        servo_id,
    )
except ImportError:  # Direct execution: python hardware/calibrate_feetech_positions.py
    from axes import AXES, AxisDefinition
    from calibration import (
        STEPS_PER_REVOLUTION,
        build_axis_calibration,
        calibration_checks,
        calibration_warnings,
        load_calibration,
        raw_to_mujoco_position,
        resolve_endpoint_deltas,
        save_calibration,
        wrapped_encoder_delta,
    )
    from feetech_bus import connect_and_ping, create_feetech_bus
    from read_feetech_positions import (
        nonnegative_integer,
        scan_for_bus,
        servo_id,
    )

__all__ = [
    "AXES",
    "AxisDefinition",
    "build_axis_calibration",
    "calibration_checks",
    "calibration_warnings",
    "load_calibration",
    "raw_to_mujoco_position",
    "resolve_endpoint_deltas",
    "wrapped_encoder_delta",
]


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        help="USB serial port. If omitted, connected USB serial ports are probed.",
    )
    parser.add_argument(
        "--ids",
        type=servo_id,
        nargs="+",
        help="Axes to calibrate, for example: --ids 1 2 3. Default: all detected mapped IDs.",
    )
    parser.add_argument(
        "--baudrate",
        type=positive_integer,
        default=1_000_000,
        help="Baud rate used by the fast --port/--ids probe (default: 1000000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hardware/feetech_calibration.json"),
        help="Calibration JSON path (default: hardware/feetech_calibration.json).",
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=9,
        help="Encoder samples captured at each pose (default: 9).",
    )
    parser.add_argument(
        "--retries",
        type=nonnegative_integer,
        default=2,
        help="Retries after a failed serial operation (default: 2).",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Load an existing output file and skip axes already captured.",
    )
    return parser.parse_args()


def read_stable_position(bus: object, motor: str, samples: int, retries: int) -> tuple[int, int]:
    values = []
    for _ in range(samples):
        value = bus.read("Present_Position", motor, normalize=False, num_retry=retries)
        values.append(int(value))
        time.sleep(0.02)
    return round(statistics.median(values)), max(values) - min(values)


def capture_pose(
    bus: object,
    *,
    motor: str,
    pose_name: str,
    q_value: float,
    samples: int,
    retries: int,
) -> tuple[int, int]:
    degrees = math.degrees(q_value)
    response = input(
        f"  Move to {pose_name:<4} q={q_value:+.3f} rad ({degrees:+.1f} deg), "
        "then press ENTER to capture [q to quit]: "
    )
    if response.strip().lower() == "q":
        raise KeyboardInterrupt
    raw, spread = read_stable_position(bus, motor, samples, retries)
    print(f"    captured raw={raw} (sample spread={spread})")
    return raw, spread


def select_ids(requested_ids: list[int] | None, detected_ids: list[int]) -> list[int]:
    if requested_ids:
        ids = sorted(set(requested_ids))
        missing = sorted(set(ids) - set(detected_ids))
        if missing:
            raise RuntimeError(f"Requested servo IDs did not respond: {missing}")
    else:
        ids = [motor_id for motor_id in detected_ids if motor_id in AXES]

    unmapped = [motor_id for motor_id in ids if motor_id not in AXES]
    if unmapped:
        raise RuntimeError(f"No MuJoCo axis mapping is defined for servo IDs: {unmapped}")
    if not ids:
        raise RuntimeError("No mapped Hepha servo IDs were selected.")
    return ids


def resolve_calibration_bus(
    bus_class: type,
    *,
    requested_port: str | None,
    requested_ids: list[int] | None,
    baudrate: int,
    retries: int,
) -> tuple[str, int, list[int]]:
    """Quickly ping explicit IDs, falling back to exhaustive discovery on failure."""
    if requested_port and requested_ids:
        ids = sorted(set(requested_ids))
        print(
            f"Checking {requested_port} at {baudrate:,} baud for IDs {ids}...",
            file=sys.stderr,
        )
        probe = create_feetech_bus(requested_port, ids)
        try:
            connect_and_ping(
                probe,
                baudrate=baudrate,
                motor_ids=ids,
                retries=retries,
            )
        except Exception as exc:
            print(
                f"Quick ping failed ({exc}); falling back to exhaustive bus scan.",
                file=sys.stderr,
            )
        else:
            print("Quick ping succeeded; exhaustive bus scan skipped.", file=sys.stderr)
            return requested_port, baudrate, ids
        finally:
            if probe.is_connected:
                probe.disconnect(disable_torque=False)

    return scan_for_bus(bus_class, requested_port)


def prepare_calibration(
    path: Path,
    *,
    overwrite: bool,
    resume: bool,
    requested_ids: list[int] | None,
    selected_ids: list[int],
    port: str,
    baudrate: int,
    role: str = "leader",
) -> tuple[dict[str, object], list[int], list[int]]:
    """Load/merge calibration state and choose axes to capture in this run."""
    if resume and not path.exists():
        raise RuntimeError(f"Cannot resume because calibration does not exist: {path}")

    if path.exists() and not overwrite:
        calibration = load_calibration(path)
        saved_role = str(calibration.get("role", "leader"))
        if saved_role != role:
            raise RuntimeError(
                f"Calibration role mismatch: {path} is {saved_role!r}, requested {role!r}."
            )
        calibration["role"] = role
        axes = calibration["axes"]
        assert isinstance(axes, dict)
        try:
            completed_ids = {int(motor_id) for motor_id in axes}
        except ValueError as exc:
            raise RuntimeError(f"Calibration contains an invalid servo ID: {path}") from exc

        replacing_ids = sorted(set(selected_ids) & completed_ids)
        if resume or requested_ids is None:
            selected_ids = [motor_id for motor_id in selected_ids if motor_id not in completed_ids]
            replacing_ids = []
        calibration["updated_at"] = datetime.now(UTC).isoformat()
        calibration["port"] = port
        calibration["baudrate"] = baudrate
        return calibration, selected_ids, replacing_ids

    calibration: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "motor_model": "sts3215",
        "encoder_resolution": STEPS_PER_REVOLUTION,
        "port": port,
        "baudrate": baudrate,
        "role": role,
        "axes": {},
    }
    return calibration, selected_ids, []


def main() -> int:
    args = parse_args()

    try:
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        print(
            'Install dependencies with: .venv/bin/python -m pip install "lerobot[feetech]==0.6.1"',
            file=sys.stderr,
        )
        return 2

    try:
        port, baudrate, detected_ids = resolve_calibration_bus(
            FeetechMotorsBus,
            requested_port=args.port,
            requested_ids=args.ids,
            baudrate=args.baudrate,
            retries=args.retries,
        )
        ids = select_ids(args.ids, detected_ids)
        calibration, ids, replacing_ids = prepare_calibration(
            args.output,
            overwrite=args.overwrite,
            resume=args.resume,
            requested_ids=args.ids,
            selected_ids=ids,
            port=port,
            baudrate=baudrate,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.output.exists() and not args.overwrite:
        axes = calibration["axes"]
        assert isinstance(axes, dict)
        preserved_ids = sorted(
            int(motor_id) for motor_id in axes if int(motor_id) not in replacing_ids
        )
        print(f"Loaded {args.output}; preserving calibrated IDs: {preserved_ids}")
        if replacing_ids:
            print(f"This run will replace only calibrated IDs: {replacing_ids}")
        if not ids:
            print(f"All selected axes are already calibrated in {args.output}.")
            return 0
        print(f"Axes selected for this run: {ids}")

    bus = create_feetech_bus(port, ids)

    try:
        connect_and_ping(bus, baudrate=baudrate, motor_ids=ids, retries=args.retries)

        print("\nHepha per-axis calibration")
        print(f"Port: {port}   Baud: {baudrate:,}   IDs: {ids}")
        print("This utility never commands a position.")
        print("Support the arm before disabling torque; calibrate one axis at a time.")
        input("Press ENTER to begin calibration: ")

        axes = calibration["axes"]
        assert isinstance(axes, dict)
        for index, motor_id in enumerate(ids, start=1):
            axis = AXES[motor_id]
            motor = f"servo_{motor_id}"
            print(f"\n[{index}/{len(ids)}] ID {motor_id}: {axis.label} ({axis.mujoco_actuator})")
            input("  Support this axis, then press ENTER to disable its torque: ")
            bus.disable_torque(motor, num_retry=args.retries)

            while True:
                raw_min, spread_min = capture_pose(
                    bus,
                    motor=motor,
                    pose_name="MIN",
                    q_value=axis.q_min,
                    samples=args.samples,
                    retries=args.retries,
                )
                raw_home, spread_home = capture_pose(
                    bus,
                    motor=motor,
                    pose_name="HOME",
                    q_value=0.0,
                    samples=args.samples,
                    retries=args.retries,
                )
                raw_max, spread_max = capture_pose(
                    bus,
                    motor=motor,
                    pose_name="MAX",
                    q_value=axis.q_max,
                    samples=args.samples,
                    retries=args.retries,
                )
                spreads = {
                    "min": spread_min,
                    "home": spread_home,
                    "max": spread_max,
                }
                problems, advisories = calibration_checks(
                    axis,
                    raw_min=raw_min,
                    raw_home=raw_home,
                    raw_max=raw_max,
                    spreads=spreads,
                )
                preview = build_axis_calibration(
                    motor_id,
                    axis,
                    raw_min=raw_min,
                    raw_home=raw_home,
                    raw_max=raw_max,
                    spreads=spreads,
                )
                resolved_min = int(preview["raw_min_delta"])
                resolved_max = int(preview["raw_max_delta"])
                if any(
                    abs(delta) > STEPS_PER_REVOLUTION // 2 for delta in (resolved_min, resolved_max)
                ):
                    print(
                        "  Resolved encoder wrap: "
                        f"MIN delta={resolved_min:+d}, MAX delta={resolved_max:+d}"
                    )
                if advisories:
                    print("  Calibration advisory (saved without requiring recapture):")
                    for advisory in advisories:
                        print(f"    - {advisory}")
                if not problems:
                    break
                print("  Calibration check requires attention:")
                for problem in problems:
                    print(f"    - {problem}")
                choice = input("  Press ENTER to recapture this axis, or type ACCEPT: ")
                if choice.strip() == "ACCEPT":
                    break

            axes[str(motor_id)] = build_axis_calibration(
                motor_id,
                axis,
                raw_min=raw_min,
                raw_home=raw_home,
                raw_max=raw_max,
                spreads=spreads,
            )
            save_calibration(args.output, calibration)
            print(f"  Saved progress to {args.output}")

        print(f"\nCalibration complete: {args.output}")
        return 0
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(f"\nStopped. Completed axes remain saved in {args.output}.")
        return 130
    except Exception as exc:
        print(f"\nCalibration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if bus.is_connected:
            # Calibrated axes were already disabled individually. Do not alter other axes.
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    raise SystemExit(main())
