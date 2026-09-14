#!/usr/bin/env python3
"""Guarded FMC4030 CLI for status, timed calibration, and preset moves."""

from __future__ import annotations

import argparse
import itertools
import math
import socket as socket
from collections.abc import Sequence
from pathlib import Path

from hardware.cnc_positions import load_limit_calibration, resolve_named_position
from hardware.fmc4030 import (
    DEFAULT_CALIBRATION_STATUS_READ_TIMEOUT_SECONDS,
    DEFAULT_IP,
    DEFAULT_MOTION_SETTLE_SECONDS,
    DEFAULT_PORT,
    DEFAULT_STATUS_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    HOME_DONE,
    MOVE_PACKET,
    Axis,
    CNCStatus,
    FMC4030Client,
    HomeDirection,
    Mode,
    build_move_payload,
    calibrate_axis_limits_timed,
    expected_motion_seconds,
    home_axes,
    load_position_config,
    move_to_absolute_targets_timed,
    read_status_with_retries,
    save_limit_calibrations,
    wait_for_move,
)
from hardware.fmc4030 import (
    CommandOutcomeUnknownError as CommandOutcomeUnknownError,
)

DEFAULT_CONFIG = Path("hardware/fmc4030_positions.json")
DEFAULT_LIMIT_CALIBRATION = Path("hardware/fmc4030_limit_calibration.json")


def check_connection(ip: str, port: int, timeout: float) -> None:
    client = FMC4030Client(ip, port, timeout)
    try:
        client.check_connection()
    finally:
        client.close()


def send_move(ip: str, port: int, timeout: float, payload: bytes) -> bytes:
    if len(payload) != MOVE_PACKET.size:
        raise ValueError(f"Move payload must be exactly {MOVE_PACKET.size} bytes.")
    client = FMC4030Client(ip, port, timeout)
    try:
        return client.send_raw_command(payload)
    finally:
        client.close()


def parse_axis(value: str) -> Axis:
    try:
        return Axis[value.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("axis must be x, y, or z") from exc


def parse_mode(value: str) -> Mode:
    try:
        return Mode[value.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("mode must be relative or absolute") from exc


def parse_home_direction(value: str) -> HomeDirection:
    try:
        return HomeDirection[value.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "direction must be positive, negative, or current"
        ) from exc


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def _add_motion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--speed", type=positive_float, default=10.0)
    parser.add_argument("--acceleration", type=positive_float, default=20.0)
    parser.add_argument("--deceleration", type=positive_float, default=20.0)
    parser.add_argument("--motion-timeout", type=positive_float, default=120.0)


def _add_calibration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-travel",
        type=positive_float,
        default=600.0,
        help="bounded travel commanded toward each endpoint; default: 600 mm",
    )
    parser.add_argument("--speed", type=positive_float, default=20.0)
    parser.add_argument("--acceleration", type=positive_float, default=200.0)
    parser.add_argument("--deceleration", type=positive_float, default=200.0)
    parser.add_argument("--backoff", type=positive_float, default=5.0)
    parser.add_argument(
        "--settle-seconds",
        type=nonnegative_float,
        default=DEFAULT_MOTION_SETTLE_SECONDS,
        help="extra wait after calculated move duration; default: 2 seconds",
    )
    parser.add_argument(
        "--status-read-timeout",
        type=positive_float,
        default=DEFAULT_CALIBRATION_STATUS_READ_TIMEOUT_SECONDS,
        help="total time allowed for each slow status read; default: 120 seconds",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_LIMIT_CALIBRATION,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded FMC4030 CNC controller")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=positive_float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--status-timeout",
        type=positive_float,
        default=DEFAULT_STATUS_TIMEOUT_SECONDS,
        help="maximum wait for one status response window; default: 3 seconds",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check", help="test TCP connectivity without sending a command")
    commands.add_parser("status", help="read positions, speeds, and limit switches")

    move_parser = commands.add_parser("move", help="send one unrestricted motion command")
    move_parser.add_argument("--axis", required=True, type=parse_axis)
    move_parser.add_argument("--position", required=True, type=float)
    move_parser.add_argument("--mode", type=parse_mode, default=Mode.RELATIVE)
    _add_motion_options(move_parser)

    home_parser = commands.add_parser("home", help="home selected axes sequentially")
    home_parser.add_argument("--axes", nargs="+", type=parse_axis, default=list(Axis))
    home_parser.add_argument(
        "--direction", type=parse_home_direction, default=HomeDirection.NEGATIVE
    )
    home_parser.add_argument("--backoff", type=nonnegative_float, default=5.0)
    _add_motion_options(home_parser)

    calibration_parser = commands.add_parser(
        "calibrate-axis",
        help="capture one axis MIN and MAX using timed bounded moves",
    )
    calibration_parser.add_argument("--axis", required=True, type=parse_axis)
    _add_calibration_options(calibration_parser)

    all_calibration_parser = commands.add_parser(
        "calibrate-all-axes",
        help="calibrate X, Y, and Z sequentially with one confirmation",
    )
    all_calibration_parser.add_argument(
        "--order",
        nargs=3,
        type=parse_axis,
        default=list(Axis),
        help="axis calibration order; default: x y z",
    )
    _add_calibration_options(all_calibration_parser)

    goto_parser = commands.add_parser(
        "goto", help="move to one of the 27 saved min/mid/max combinations"
    )
    goto_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for axis_name in ("x", "y", "z"):
        goto_parser.add_argument(f"--{axis_name}", required=True, choices=("min", "mid", "max"))
    goto_parser.add_argument(
        "--order",
        nargs=3,
        type=parse_axis,
        default=[Axis.Z, Axis.X, Axis.Y],
        help="sequential movement order; default: z x y",
    )
    _add_motion_options(goto_parser)

    named_parser = commands.add_parser(
        "goto-position",
        help="move to named CNC position A or drawer-dependent B",
    )
    named_parser.add_argument(
        "--position",
        required=True,
        type=str.upper,
        choices=("A", "B"),
    )
    named_parser.add_argument("--drawer", type=int, choices=range(1, 10))
    named_parser.add_argument("--config", type=Path, default=DEFAULT_LIMIT_CALIBRATION)
    named_parser.add_argument(
        "--order",
        nargs=3,
        type=parse_axis,
        default=[Axis.Z, Axis.X, Axis.Y],
        help="command transmission order; axes move in parallel; default: z x y",
    )
    named_parser.add_argument("--speed", type=positive_float, default=200.0)
    named_parser.add_argument("--acceleration", type=positive_float, default=200.0)
    named_parser.add_argument("--deceleration", type=positive_float, default=200.0)
    named_parser.add_argument(
        "--settle-seconds",
        type=nonnegative_float,
        default=DEFAULT_MOTION_SETTLE_SECONDS,
    )
    named_parser.add_argument(
        "--status-read-timeout",
        type=positive_float,
        default=DEFAULT_CALIBRATION_STATUS_READ_TIMEOUT_SECONDS,
    )
    named_parser.add_argument(
        "--position-tolerance",
        type=positive_float,
        default=1.0,
        help="maximum final absolute position error; default: 1 mm",
    )
    named_parser.add_argument(
        "--read-initial-status",
        action="store_true",
        help="opt in to a slow preflight status read; skipped by default",
    )
    named_parser.add_argument(
        "--read-final-status",
        action="store_true",
        help="opt in to slow final position verification; skipped by default",
    )
    return parser


def _print_status(status: CNCStatus) -> None:
    home_names = {0x08: "DONE", 0x0A: "STARTING", 0x0B: "NOT FINISHED"}
    print("\nFMC4030 status")
    print("Axis  Position     Speed    Negative limit  Positive limit  Axis status")
    print("----  --------  ---------  --------------  --------------  -----------")
    for axis in Axis:
        negative = "TRIGGERED" if status.negative_limit(axis) else "off"
        positive = "TRIGGERED" if status.positive_limit(axis) else "off"
        print(
            f" {axis.name}   {status.position(axis):8.3f}  "
            f"{status.speed(axis):8.3f}  {negative:>14}  {positive:>14}  "
            f"{status.axis_statuses[int(axis)]:11d}"
        )
    print(f"Run status: {status.run_status}")
    print(f"Home status: {home_names.get(status.home_status, str(status.home_status))}")
    print(f"Inputs: 0b{status.input_mask:b}   Outputs: 0b{status.output_mask:b}")


def _pause(prompt: str) -> bool:
    try:
        input(prompt)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. No next command was sent.")
        return False
    return True


def _move_and_wait(
    client: FMC4030Client,
    axis: Axis,
    target: float,
    speed: float,
    acceleration: float,
    deceleration: float,
    timeout: float,
) -> CNCStatus:
    initial = read_status_with_retries(client)
    if abs(target - initial.position(axis)) <= 0.01:
        return initial
    client.move(axis, target, speed, acceleration, deceleration, Mode.ABSOLUTE)
    return wait_for_move(client, axis, initial.position(axis), timeout)


def _run_move(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        payload = build_move_payload(
            args.axis,
            args.position,
            args.speed,
            args.acceleration,
            args.deceleration,
            args.mode,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Controller: {args.ip}:{args.port}")
    print(f"Move: {args.axis.name} {args.mode.name} {args.position:+.3f} mm")
    print(f"Speed: {args.speed:.3f} mm/s")
    print(f"Payload: {payload.hex(' ')}")
    print("Ensure the machine is clear and you can remove power immediately.")
    if not _pause("Press ENTER to transmit this move: "):
        return 1
    response = send_move(args.ip, args.port, args.timeout, payload)
    print(f"Received: {response.hex(' ')}")
    return 0


def _run_home(args: argparse.Namespace, client: FMC4030Client) -> int:
    if len(set(args.axes)) != len(args.axes):
        raise ValueError("Each axis may be listed only once.")
    print("Homing plan:")
    print(f"  Axes: {' '.join(axis.name for axis in args.axes)}")
    print(f"  Direction: {args.direction.name}")
    print(f"  Speed: {args.speed:g} mm/s   Backoff: {args.backoff:g} mm")
    print("Axes are homed sequentially. Ensure every limit switch works.")
    if not _pause("Press ENTER to start homing: "):
        return 1
    results = home_axes(
        client,
        args.axes,
        args.direction,
        args.speed,
        args.acceleration,
        args.backoff,
        args.motion_timeout,
    )
    for axis, status in results.items():
        print(f"{axis.name} home complete at {status.position(axis):.3f} mm")
    return 0


def _run_axis_calibration(args: argparse.Namespace, client: FMC4030Client) -> int:
    long_wait = expected_motion_seconds(
        args.max_travel,
        args.speed,
        args.acceleration,
        args.deceleration,
    ) + args.settle_seconds

    print("FMC4030 timed single-axis calibration")
    print(f"  Axis: {args.axis.name}")
    print(f"  Bounded move toward each limit: {args.max_travel:g} mm")
    print(f"  Speed: {args.speed:g} mm/s")
    print(f"  Calculated wait after each long move: {long_wait:.1f} s")
    print(f"  Safe backoff from each switch: {args.backoff:g} mm")
    print(f"  Status response window: {args.status_timeout:g} s")
    print(f"  Total deadline for each status read: {args.status_read_timeout:g} s")
    print("Sequence: MIN move -> status -> backoff -> MAX move -> status -> backoff.")
    print("No status request is sent while a bounded move is running.")
    print("Nothing is saved unless both stopped limit states are verified.")
    print("Clear the complete axis travel and keep emergency power removal within reach.")
    if not _pause("Press ENTER to start this axis calibration: "):
        return 1

    def show_progress(phase: str, wait_seconds: float) -> None:
        descriptions = {
            "move_min": f"Moving {args.axis.name} toward MIN",
            "backoff_min": f"Backing {args.axis.name} away from MIN",
            "move_max": f"Moving {args.axis.name} toward MAX",
            "backoff_max": f"Backing {args.axis.name} away from MAX",
            "read_min": (
                "Reading and verifying MIN status "
                f"(allowing up to {args.status_read_timeout:g} s)"
            ),
            "read_max": (
                "Reading and verifying MAX status "
                f"(allowing up to {args.status_read_timeout:g} s)"
            ),
        }
        suffix = f"; waiting {wait_seconds:.1f} s" if wait_seconds else ""
        print(f"\n{descriptions[phase]}{suffix}...")

    calibration = calibrate_axis_limits_timed(
        client,
        args.axis,
        args.max_travel,
        args.speed,
        args.acceleration,
        args.deceleration,
        args.backoff,
        settle_seconds=args.settle_seconds,
        status_read_timeout=args.status_read_timeout,
        progress=show_progress,
    )
    save_limit_calibrations(
        args.config,
        args.ip,
        args.port,
        [calibration.minimum, calibration.maximum],
    )
    midpoint = (
        calibration.minimum.safe_position_mm + calibration.maximum.safe_position_mm
    ) / 2.0
    print(f"\n{args.axis.name} calibration complete")
    print(
        f"  MIN switch={calibration.minimum.switch_position_mm:.3f} mm, "
        f"safe={calibration.minimum.safe_position_mm:.3f} mm"
    )
    print(f"  MID={midpoint:.3f} mm")
    print(
        f"  MAX switch={calibration.maximum.switch_position_mm:.3f} mm, "
        f"safe={calibration.maximum.safe_position_mm:.3f} mm"
    )
    print(f"Saved to {args.config}")
    return 0


def _run_all_axes_calibration(args: argparse.Namespace, client: FMC4030Client) -> int:
    if len(set(args.order)) != 3:
        raise ValueError("--order must contain x, y, and z exactly once.")
    long_wait = expected_motion_seconds(
        args.max_travel,
        args.speed,
        args.acceleration,
        args.deceleration,
    ) + args.settle_seconds

    print("FMC4030 timed all-axis calibration")
    print(f"  Order: {' '.join(axis.name for axis in args.order)}")
    print(f"  Bounded move toward each limit: {args.max_travel:g} mm")
    print(f"  Speed: {args.speed:g} mm/s")
    print(f"  Calculated wait after each long move: {long_wait:.1f} s")
    print(f"  Safe backoff from every switch: {args.backoff:g} mm")
    print(f"  Status response window: {args.status_timeout:g} s")
    print(f"  Total deadline for each status read: {args.status_read_timeout:g} s")
    print("Each axis completes MIN -> status -> backoff -> MAX -> status -> backoff.")
    print("Axes are calibrated sequentially; no status is requested during motion.")
    print("All six endpoints are saved together only after every axis succeeds.")
    print("Clear the complete 3D workspace and keep emergency power removal within reach.")
    if not _pause("Press ENTER to start calibration of all axes: "):
        return 1

    endpoints = []
    calibrations = []
    for index, axis in enumerate(args.order, start=1):
        print(f"\n[{index}/3] Calibrating {axis.name}")

        def show_progress(
            phase: str,
            wait_seconds: float,
            *,
            selected_axis: Axis = axis,
        ) -> None:
            descriptions = {
                "move_min": f"Moving {selected_axis.name} toward MIN",
                "backoff_min": f"Backing {selected_axis.name} away from MIN",
                "move_max": f"Moving {selected_axis.name} toward MAX",
                "backoff_max": f"Backing {selected_axis.name} away from MAX",
                "read_min": (
                    "Reading and verifying MIN status "
                    f"(allowing up to {args.status_read_timeout:g} s)"
                ),
                "read_max": (
                    "Reading and verifying MAX status "
                    f"(allowing up to {args.status_read_timeout:g} s)"
                ),
            }
            suffix = f"; waiting {wait_seconds:.1f} s" if wait_seconds else ""
            print(f"{descriptions[phase]}{suffix}...")

        calibration = calibrate_axis_limits_timed(
            client,
            axis,
            args.max_travel,
            args.speed,
            args.acceleration,
            args.deceleration,
            args.backoff,
            settle_seconds=args.settle_seconds,
            status_read_timeout=args.status_read_timeout,
            progress=show_progress,
        )
        calibrations.append(calibration)
        endpoints.extend((calibration.minimum, calibration.maximum))
        print(
            f"{axis.name} captured: safe MIN={calibration.minimum.safe_position_mm:.3f}, "
            f"safe MAX={calibration.maximum.safe_position_mm:.3f} mm"
        )

    save_limit_calibrations(args.config, args.ip, args.port, endpoints)
    print("\nAll-axis calibration complete")
    for calibration in calibrations:
        midpoint = (
            calibration.minimum.safe_position_mm
            + calibration.maximum.safe_position_mm
        ) / 2.0
        print(
            f"  {calibration.axis.name}: "
            f"MIN={calibration.minimum.safe_position_mm:.3f}, "
            f"MID={midpoint:.3f}, MAX={calibration.maximum.safe_position_mm:.3f} mm"
        )
    print(f"Saved to {args.config}")
    return 0


def _run_goto(args: argparse.Namespace, client: FMC4030Client) -> int:
    config = load_position_config(args.config)
    if (args.ip, args.port) != (config.ip, config.port):
        raise ValueError(f"Config belongs to {config.ip}:{config.port}, not {args.ip}:{args.port}.")
    if len(set(args.order)) != 3:
        raise ValueError("--order must contain x, y, and z exactly once.")
    status = client.read_status()
    if status.home_status != HOME_DONE:
        raise RuntimeError(
            "The controller does not report homing complete. Run the home command first."
        )
    selected = {axis: getattr(args, axis.name.lower()) for axis in Axis}
    targets = {axis: config.axes[axis.name.lower()].preset(selected[axis]) for axis in Axis}
    print("Discrete CNC move:")
    for axis in Axis:
        print(f"  {axis.name}: {selected[axis]} = {targets[axis]:.3f} mm")
    print(f"  Order: {' '.join(axis.name for axis in args.order)}")
    print("Only stored min/mid/max values are accepted by this command.")
    if not _pause("Press ENTER to move to this saved position: "):
        return 1
    for axis in args.order:
        status = _move_and_wait(
            client,
            axis,
            targets[axis],
            args.speed,
            args.acceleration,
            args.deceleration,
            args.motion_timeout,
        )
        print(f"{axis.name} reached {status.position(axis):.3f} mm")
    return 0


def _run_named_position(args: argparse.Namespace, client: FMC4030Client) -> int:
    calibration = load_limit_calibration(args.config)
    if (args.ip, args.port) != (calibration.ip, calibration.port):
        raise ValueError(
            f"Calibration belongs to {calibration.ip}:{calibration.port}, "
            f"not {args.ip}:{args.port}."
        )
    if len(set(args.order)) != 3:
        raise ValueError("--order must contain x, y, and z exactly once.")
    targets = resolve_named_position(calibration, args.position, args.drawer)

    label = args.position if args.drawer is None else f"{args.position} (drawer {args.drawer})"
    print(f"FMC4030 named position: {label}")
    for axis in Axis:
        axis_range = calibration.axes[axis]
        print(
            f"  {axis.name}: {targets[axis]:+9.3f} mm  "
            f"safe [{axis_range.minimum_mm:+.3f}, {axis_range.maximum_mm:+.3f}]"
        )
    print(f"  Command order: {' '.join(axis.name for axis in args.order)}")
    print(
        f"  Speed: {args.speed:g} mm/s   acceleration: {args.acceleration:g} mm/s²   "
        f"deceleration: {args.deceleration:g} mm/s²"
    )
    print("Targets were validated against the measured safe calibration interval.")
    print("Absolute axis commands are sent back-to-back and execute in parallel.")
    if args.read_initial_status:
        print("Initial status: enabled; stopped state and current positions will be checked.")
    elif args.read_final_status:
        print("Initial status: skipped; verification timing uses full calibrated spans.")
    else:
        print("Initial status: skipped.")
    if args.read_final_status:
        print("Final status: enabled; reached positions will be verified.")
    else:
        print("Final status: skipped; the command returns without waiting for motion.")
    print("Starting immediately. Keep emergency power removal within reach.")

    def show_progress(phase: str, axis: Axis | None, wait_seconds: float) -> None:
        if phase == "read_initial":
            print(f"Reading initial status (allowing up to {wait_seconds:g} s)...")
        elif phase == "read_final":
            print(f"Reading final status (allowing up to {wait_seconds:g} s)...")
        elif phase == "skip" and axis is not None:
            print(f"{axis.name} is already at its target.")
        elif phase == "command" and axis is not None:
            print(f"Commanding {axis.name} absolute target {targets[axis]:+.3f} mm...")
        elif phase == "wait_parallel":
            print(f"Axes moving in parallel; waiting {wait_seconds:.1f} s...")

    final = move_to_absolute_targets_timed(
        client,
        targets,
        args.order,
        args.speed,
        args.acceleration,
        args.deceleration,
        settle_seconds=args.settle_seconds,
        status_read_timeout=args.status_read_timeout,
        read_initial_status=args.read_initial_status,
        read_final_status=args.read_final_status,
        travel_distance_bounds_mm={
            axis: calibration.axes[axis].maximum_mm
            - calibration.axes[axis].minimum_mm
            for axis in Axis
        },
        position_tolerance_mm=args.position_tolerance,
        progress=show_progress,
    )
    if final is None:
        print(
            f"\nNamed position {label} commands accepted; returning immediately. "
            "Controller motion may still be in progress."
        )
        return 0
    print(f"\nReached named position {label}:")
    for axis in Axis:
        print(f"  {axis.name}: {final.position(axis):+.3f} mm")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    client = FMC4030Client(
        args.ip,
        args.port,
        args.timeout,
        status_timeout=args.status_timeout,
    )

    try:
        if args.command == "check":
            client.check_connection()
            print(f"Connected successfully to {args.ip}:{args.port}. No command was sent.")
            return 0
        if args.command == "status":
            _print_status(read_status_with_retries(client))
            return 0
        if args.command == "move":
            return _run_move(args, parser)
        if args.command == "home":
            return _run_home(args, client)
        if args.command == "calibrate-axis":
            return _run_axis_calibration(args, client)
        if args.command == "calibrate-all-axes":
            return _run_all_axes_calibration(args, client)
        if args.command == "goto":
            return _run_goto(args, client)
        if args.command == "goto-position":
            return _run_named_position(args, client)
    except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
        print(f"FMC4030 error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user. Any active guarded motion requested an immediate stop.")
        return 130
    finally:
        client.close()
    parser.error("unknown command")


def all_discrete_position_names() -> list[tuple[str, str, str]]:
    """Return the 27 possible X/Y/Z preset-name combinations."""
    return list(itertools.product(("min", "mid", "max"), repeat=3))


if __name__ == "__main__":
    raise SystemExit(main())
