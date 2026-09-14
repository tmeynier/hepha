#!/usr/bin/env python3
"""Calibrate and safely commission a position-controlled Feetech follower arm."""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from .axes import AXES, AxisDefinition
    from .calibrate_feetech_positions import (
        capture_pose,
        positive_integer,
        prepare_calibration,
        read_stable_position,
        resolve_calibration_bus,
        select_ids,
    )
    from .calibration import (
        STEPS_PER_REVOLUTION,
        build_axis_calibration,
        calibration_checks,
        save_calibration,
        wrapped_encoder_delta,
    )
    from .feetech_bus import connect_and_ping, create_feetech_bus
    from .read_feetech_positions import nonnegative_integer, servo_id
except ImportError:  # Direct hardware script execution.
    from axes import AXES, AxisDefinition
    from calibrate_feetech_positions import (
        capture_pose,
        positive_integer,
        prepare_calibration,
        read_stable_position,
        resolve_calibration_bus,
        select_ids,
    )
    from calibration import (
        STEPS_PER_REVOLUTION,
        build_axis_calibration,
        calibration_checks,
        save_calibration,
        wrapped_encoder_delta,
    )
    from feetech_bus import connect_and_ping, create_feetech_bus
    from read_feetech_positions import nonnegative_integer, servo_id

FOLLOWER_HOME_RAW = STEPS_PER_REVOLUTION // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        help="Follower USB serial port. If omitted, connected adapters are probed.",
    )
    parser.add_argument(
        "--ids",
        type=servo_id,
        nargs="+",
        help="Follower axes to calibrate. Default: all detected mapped IDs.",
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
        default=Path("hardware/feetech_follower_calibration.json"),
    )
    parser.add_argument("--samples", type=positive_integer, default=9)
    parser.add_argument("--retries", type=nonnegative_integer, default=2)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def centered_raw(raw: int, applied_shift: int) -> int:
    """Transform a reading captured before the new follower homing offset."""
    return (raw - applied_shift) % STEPS_PER_REVOLUTION


def center_follower_home(
    bus: object,
    *,
    motor: str,
    raw_home: int,
    target_home: int = FOLLOWER_HOME_RAW,
    retries: int,
) -> tuple[int, int]:
    """Persistently shift captured home and return (offset, applied shift)."""
    bus.write("Min_Position_Limit", motor, 0, normalize=False, num_retry=retries)
    bus.write(
        "Max_Position_Limit",
        motor,
        STEPS_PER_REVOLUTION - 1,
        normalize=False,
        num_retry=retries,
    )
    bus.write("Operating_Mode", motor, 0, normalize=False, num_retry=retries)
    old_offset = int(
        bus.read("Homing_Offset", motor, normalize=False, num_retry=retries)
    )
    applied_shift = wrapped_encoder_delta(raw_home, target_home)
    new_offset = wrapped_encoder_delta(old_offset + applied_shift, 0)
    bus.write("Homing_Offset", motor, new_offset, normalize=False, num_retry=retries)
    return new_offset, applied_shift


def follower_home_target(axis_calibration: dict[str, object]) -> int:
    """Choose the nearest-to-center home that keeps the full raw interval contiguous."""
    deltas = (
        int(axis_calibration["raw_min_delta"]),
        0,
        int(axis_calibration["raw_max_delta"]),
    )
    lower_home = -min(deltas)
    upper_home = STEPS_PER_REVOLUTION - 1 - max(deltas)
    if lower_home > upper_home:
        raise RuntimeError("Follower calibrated motion spans a complete encoder revolution.")
    return min(upper_home, max(lower_home, FOLLOWER_HOME_RAW))


def follower_hardware_limits(axis_calibration: dict[str, object]) -> tuple[int, int]:
    """Return non-wrapping raw limits after follower home has been centered."""
    raw_values = (
        int(axis_calibration["raw_min"]),
        int(axis_calibration["raw_home"]),
        int(axis_calibration["raw_max"]),
    )
    lower = min(raw_values)
    upper = max(raw_values)
    if lower == upper or not 0 <= lower < upper < STEPS_PER_REVOLUTION:
        raise RuntimeError(
            "Follower calibration does not form a safe non-wrapping hardware interval."
        )
    return lower, upper


def capture_follower_reference_poses(
    bus: object,
    *,
    motor: str,
    axis: AxisDefinition,
    samples: int,
    retries: int,
) -> tuple[int, int, int, dict[str, int]]:
    """Capture MIN, MAX, then HOME so follower HOME is requested only once."""
    raw_min, spread_min = capture_pose(
        bus,
        motor=motor,
        pose_name="MIN",
        q_value=axis.q_min,
        samples=samples,
        retries=retries,
    )
    raw_max, spread_max = capture_pose(
        bus,
        motor=motor,
        pose_name="MAX",
        q_value=axis.q_max,
        samples=samples,
        retries=retries,
    )
    raw_home, spread_home = capture_pose(
        bus,
        motor=motor,
        pose_name="HOME",
        q_value=0.0,
        samples=samples,
        retries=retries,
    )
    spreads = {
        "min": spread_min,
        "home": spread_home,
        "max": spread_max,
    }
    return raw_min, raw_home, raw_max, spreads


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
            role="follower",
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
        print(f"Loaded {args.output}; preserving calibrated follower IDs: {preserved_ids}")
        if replacing_ids:
            print(f"This run will replace only follower IDs: {replacing_ids}")
        if not ids:
            print(f"All selected follower axes are already calibrated in {args.output}.")
            return 0
        print(f"Follower axes selected for this run: {ids}")

    bus = create_feetech_bus(port, ids)
    try:
        connect_and_ping(bus, baudrate=baudrate, motor_ids=ids, retries=args.retries)
        print("\nHepha follower per-axis calibration")
        print(f"Port: {port}   Baud: {baudrate:,}   IDs: {ids}")
        print("Capture order: MIN, MAX, HOME. HOME is requested only once.")
        print("This follower utility also writes homing offsets, position mode, and limits.")
        print("It never commands a position during calibration; all motion is manual.")
        print("Support the follower before disabling torque; calibrate one axis at a time.")
        input("Press ENTER to begin follower calibration: ")

        axes = calibration["axes"]
        assert isinstance(axes, dict)
        for index, motor_id in enumerate(ids, start=1):
            axis = AXES[motor_id]
            motor = f"servo_{motor_id}"
            print(f"\n[{index}/{len(ids)}] ID {motor_id}: {axis.label} ({axis.joint_name})")
            input("  Support this axis, then press ENTER to disable its torque: ")
            bus.disable_torque(motor, num_retry=args.retries)

            while True:
                raw_min, raw_home, raw_max, spreads = capture_follower_reference_poses(
                    bus,
                    motor=motor,
                    axis=axis,
                    samples=args.samples,
                    retries=args.retries,
                )
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
                try:
                    target_home = follower_home_target(preview)
                except RuntimeError as exc:
                    problems.append(str(exc))
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
                    target_home = follower_home_target(preview)
                    break

            print("  HOME captured; verifying it before writing the homing offset and limits.")
            current_home, verification_spread = read_stable_position(
                bus,
                motor,
                args.samples,
                args.retries,
            )
            home_error = abs(wrapped_encoder_delta(current_home, raw_home))
            if home_error > 8 or verification_spread > 4:
                raise RuntimeError(
                    f"ID {motor_id} was not stable at the captured HOME "
                    f"(encoder error={home_error}, spread={verification_spread}); rerun this axis."
                )
            homing_offset, shift = center_follower_home(
                bus,
                motor=motor,
                raw_home=raw_home,
                target_home=target_home,
                retries=args.retries,
            )
            raw_min = centered_raw(raw_min, shift)
            raw_home = centered_raw(raw_home, shift)
            raw_max = centered_raw(raw_max, shift)
            preview = build_axis_calibration(
                motor_id,
                axis,
                raw_min=raw_min,
                raw_home=raw_home,
                raw_max=raw_max,
                spreads=spreads,
            )
            verified_home, verification_spread = read_stable_position(
                bus,
                motor,
                args.samples,
                args.retries,
            )
            centered_error = abs(wrapped_encoder_delta(verified_home, raw_home))
            if centered_error > 4 or verification_spread > 4:
                raise RuntimeError(
                    f"ID {motor_id} homing-offset verification failed "
                    f"(encoder error={centered_error}, spread={verification_spread}); "
                    "rerun this axis."
                )
            hardware_min, hardware_max = follower_hardware_limits(preview)
            bus.write(
                "Min_Position_Limit",
                motor,
                hardware_min,
                normalize=False,
                num_retry=args.retries,
            )
            bus.write(
                "Max_Position_Limit",
                motor,
                hardware_max,
                normalize=False,
                num_retry=args.retries,
            )
            preview.update(
                {
                    "homing_offset": homing_offset,
                    "hardware_min": hardware_min,
                    "hardware_max": hardware_max,
                    "operating_mode": 0,
                }
            )
            axes[str(motor_id)] = preview
            calibration["updated_at"] = datetime.now(UTC).isoformat()
            save_calibration(args.output, calibration)
            print(
                f"  Centered HOME at raw={raw_home} with offset={homing_offset:+d}.\n"
                f"  Saved follower ID {motor_id}: raw limits "
                f"[{hardware_min}, {hardware_max}] to {args.output}"
            )

        print(f"\nFollower calibration complete: {args.output}")
        return 0
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(
            f"\nStopped. Completed axes remain saved in {args.output}. "
            "Re-run the interrupted axis before teleoperation."
        )
        return 130
    except Exception as exc:
        print(f"\nFollower calibration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    raise SystemExit(main())
