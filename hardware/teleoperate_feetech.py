#!/usr/bin/env python3
"""Teleoperate a calibrated physical follower from a calibrated Feetech leader."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from .calibration import CalibratedAxis, load_calibrated_axes
    from .feetech_follower import FeetechFollower
    from .feetech_leader import FeetechLeader
    from .read_feetech_positions import nonnegative_integer
except ImportError:  # Direct hardware script execution.
    from calibration import CalibratedAxis, load_calibrated_axes
    from feetech_follower import FeetechFollower
    from feetech_leader import FeetechLeader
    from read_feetech_positions import nonnegative_integer


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def bounded_integer(minimum: int, maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"value must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leader-calibration",
        type=Path,
        default=Path("hardware/feetech_calibration.json"),
    )
    parser.add_argument(
        "--follower-calibration",
        type=Path,
        default=Path("hardware/feetech_follower_calibration.json"),
    )
    parser.add_argument("--leader-port")
    parser.add_argument("--follower-port")
    parser.add_argument(
        "--joints",
        nargs="+",
        help="Semantic joints to mirror (default: every joint shared by both files).",
    )
    parser.add_argument("--fps", type=positive_float, default=30.0)
    parser.add_argument(
        "--max-velocity-deg",
        type=positive_float,
        default=30.0,
        help="Maximum follower target change per second (default: 30 degrees/s).",
    )
    parser.add_argument(
        "--startup-velocity-deg",
        type=positive_float,
        default=10.0,
        help="Follower speed while automatically approaching the leader (default: 10 degrees/s).",
    )
    parser.add_argument(
        "--startup-tolerance-deg",
        "--max-start-error-deg",
        dest="startup_tolerance_deg",
        type=positive_float,
        default=5.0,
        help="Error at which normal teleoperation begins (default: 5 degrees).",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=positive_float,
        default=60.0,
        help="Stop if automatic startup alignment does not converge (default: 60 seconds).",
    )
    parser.add_argument("--smoothing", type=positive_float, default=0.25)
    parser.add_argument("--acceleration", type=bounded_integer(0, 254), default=50)
    parser.add_argument("--torque-limit", type=bounded_integer(0, 1000), default=500)
    parser.add_argument("--retries", type=nonnegative_integer, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and compare both arms without enabling torque or writing follower goals.",
    )
    return parser.parse_args()


def resolve_joint_names(
    leader_axes: tuple[CalibratedAxis, ...],
    follower_axes: tuple[CalibratedAxis, ...],
    requested: list[str] | None,
) -> tuple[str, ...]:
    leader_names = {axis.joint_name for axis in leader_axes}
    follower_names = {axis.joint_name for axis in follower_axes}
    shared = leader_names & follower_names
    if requested:
        selected = tuple(dict.fromkeys(requested))
        missing = sorted(set(selected) - shared)
        if missing:
            raise RuntimeError(f"Joints are not calibrated on both arms: {missing}")
        return selected
    selected = tuple(axis.joint_name for axis in leader_axes if axis.joint_name in shared)
    if not selected:
        raise RuntimeError("Leader and follower calibrations have no joints in common.")
    return selected


@dataclass
class JointSafetyLimiter:
    limits: dict[str, tuple[float, float]]
    max_velocity: float
    previous: dict[str, float]

    def apply(self, requested: dict[str, float], dt: float) -> dict[str, float]:
        maximum_step = self.max_velocity * max(0.0, dt)
        result = {}
        for joint, (lower, upper) in self.limits.items():
            bounded = min(upper, max(lower, float(requested[joint])))
            previous = self.previous[joint]
            result[joint] = previous + min(
                maximum_step,
                max(-maximum_step, bounded - previous),
            )
        self.previous = result.copy()
        return result


def start_pose_errors(
    leader: dict[str, float],
    follower: dict[str, float],
) -> dict[str, float]:
    return {joint: abs(leader[joint] - follower[joint]) for joint in leader}


def startup_alignment_complete(
    leader: dict[str, float],
    follower: dict[str, float],
    tolerance: float,
) -> bool:
    """Return whether every follower joint is close enough to its leader target."""
    return all(error <= tolerance for error in start_pose_errors(leader, follower).values())


def print_joint_status(
    joints: tuple[str, ...],
    requested: dict[str, float],
    limited: dict[str, float],
    actual: dict[str, float],
    previous_width: int,
    *,
    prefix: str = "",
) -> int:
    status = " | ".join(
        f"{joint}: L={math.degrees(requested[joint]):+6.1f} "
        f"T={math.degrees(limited[joint]):+6.1f} "
        f"F={math.degrees(actual[joint]):+6.1f} deg"
        for joint in joints
    )
    line = f"{prefix}{status}"
    print(f"\r{line:<{previous_width}}", end="", flush=True)
    return max(previous_width, len(line))


def run_teleoperation(args: argparse.Namespace) -> None:
    if not 0 < args.smoothing <= 1:
        raise ValueError("--smoothing must be greater than 0 and at most 1")
    leader_calibration, leader_available = load_calibrated_axes(args.leader_calibration)
    follower_calibration, follower_available = load_calibrated_axes(args.follower_calibration)
    if follower_calibration.get("role") != "follower":
        raise RuntimeError("The follower calibration file is not marked as role=follower.")
    joints = resolve_joint_names(leader_available, follower_available, args.joints)
    leader_port = args.leader_port or str(leader_calibration["port"])
    follower_port = args.follower_port or str(follower_calibration["port"])
    if leader_port == follower_port:
        raise RuntimeError("Leader and follower must use different USB serial ports.")

    leader = FeetechLeader(
        calibration_path=args.leader_calibration,
        joint_names=joints,
        port=leader_port,
        retries=args.retries,
        smoothing=args.smoothing,
    )
    follower = FeetechFollower(
        calibration_path=args.follower_calibration,
        joint_names=joints,
        port=follower_port,
        retries=args.retries,
    )

    with leader, follower:
        print("\nPhysical Feetech leader -> physical Feetech follower")
        print(f"Leader:   {leader.port}   IDs: {list(leader.servo_ids)}")
        print(f"Follower: {follower.port}   IDs: {list(follower.servo_ids)}")
        print(f"Joints: {', '.join(joints)}")
        print("The next step disables torque on both arms. Support both arms securely.")
        input("Press ENTER to disable torque on both selected arms: ")
        leader.disable_torque()
        follower.disable_torque()

        leader.reset_filter()
        leader_positions = leader.read_joint_positions()
        follower_positions = follower.read_joint_positions()
        errors = start_pose_errors(leader_positions, follower_positions)
        print("Initial leader/follower difference:")
        for joint in joints:
            print(
                f"  {joint:<12} leader={math.degrees(leader_positions[joint]):+7.2f} deg  "
                f"follower={math.degrees(follower_positions[joint]):+7.2f} deg  "
                f"error={math.degrees(errors[joint]):5.2f} deg"
            )

        limits = {
            axis.joint_name: (
                float(axis.calibration["q_min"]),
                float(axis.calibration["q_max"]),
            )
            for axis in follower.axes
        }
        period = 1.0 / args.fps
        previous_status_width = 0

        if args.dry_run:
            print("Dry run: follower torque remains disabled and no goals will be written.")
            limiter = JointSafetyLimiter(
                limits=limits,
                max_velocity=math.radians(args.max_velocity_deg),
                previous=follower_positions.copy(),
            )
        else:
            input(
                "Clear the follower workspace, keep an emergency power cutoff ready, "
                "then press ENTER to enable torque and approach the leader slowly: "
            )
            follower.configure_runtime(
                acceleration=args.acceleration,
                torque_limit=args.torque_limit,
            )
            follower_positions = follower.seed_goals_from_present_position()
            follower.enable_torque()
            print("Follower torque enabled.")
            limiter = JointSafetyLimiter(
                limits=limits,
                max_velocity=math.radians(args.startup_velocity_deg),
                previous=follower_positions.copy(),
            )
            tolerance = math.radians(args.startup_tolerance_deg)
            deadline = time.monotonic() + args.startup_timeout_seconds
            last_tick = time.monotonic()
            print(
                "Hold the leader steady. Automatically aligning follower at "
                f"{args.startup_velocity_deg:.1f} degrees/s. Press Ctrl+C to stop."
            )
            while True:
                started = time.monotonic()
                dt = started - last_tick
                last_tick = started
                requested = leader.read_joint_positions()
                limited = limiter.apply(requested, dt)
                follower.write_joint_positions(limited)
                actual = follower.read_joint_positions()
                previous_status_width = print_joint_status(
                    joints,
                    requested,
                    limited,
                    actual,
                    previous_status_width,
                    prefix="ALIGNING | ",
                )
                if startup_alignment_complete(requested, actual, tolerance):
                    print(
                        f"\nFollower is within {args.startup_tolerance_deg:.1f} degrees "
                        "of the leader. Normal teleoperation started."
                    )
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Automatic startup alignment did not converge within "
                        f"{args.startup_timeout_seconds:.1f} seconds."
                    )
                time.sleep(max(0.0, period - (time.monotonic() - started)))
            limiter.max_velocity = math.radians(args.max_velocity_deg)

        last_tick = time.monotonic()
        print("Move the leader slowly. Press Ctrl+C to stop and disable follower torque.")
        while True:
            started = time.monotonic()
            dt = started - last_tick
            last_tick = started
            requested = leader.read_joint_positions()
            limited = limiter.apply(requested, dt)
            if not args.dry_run:
                follower.write_joint_positions(limited)
            actual = follower.read_joint_positions()
            previous_status_width = print_joint_status(
                joints,
                requested,
                limited,
                actual,
                previous_status_width,
            )
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def main() -> int:
    args = parse_args()
    try:
        run_teleoperation(args)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nStopped. Follower torque was disabled.")
        return 130
    except Exception as exc:
        print(f"\nPhysical teleoperation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
