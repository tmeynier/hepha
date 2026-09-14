#!/usr/bin/env python3
"""Teleoperate a Feetech follower through one interactive A/B/A CNC episode."""

from __future__ import annotations

import argparse
import math
import select
import signal
import sys
import termios
import time
import tty
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import TracebackType

try:
    from .calibration import load_calibrated_axes
    from .feetech_follower import FeetechFollower
    from .feetech_leader import FeetechLeader
    from .fmc4030 import Axis
    from .read_feetech_positions import nonnegative_integer
    from .teleop_episode import (
        CNCMove,
        EpisodeFrame,
        EpisodePhase,
        EpisodeSink,
        NamedCNCController,
        NullEpisodeSink,
        TeleopEpisodeSequence,
        choose_drawer,
    )
    from .teleoperate_feetech import (
        JointSafetyLimiter,
        bounded_integer,
        print_joint_status,
        resolve_joint_names,
        startup_alignment_complete,
    )
except ImportError:  # Direct hardware script execution.
    from calibration import load_calibrated_axes
    from feetech_follower import FeetechFollower
    from feetech_leader import FeetechLeader
    from fmc4030 import Axis
    from read_feetech_positions import nonnegative_integer
    from teleop_episode import (
        CNCMove,
        EpisodeFrame,
        EpisodePhase,
        EpisodeSink,
        NamedCNCController,
        NullEpisodeSink,
        TeleopEpisodeSequence,
        choose_drawer,
    )
    from teleoperate_feetech import (
        JointSafetyLimiter,
        bounded_integer,
        print_joint_status,
        resolve_joint_names,
        startup_alignment_complete,
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite nonnegative number")
    return parsed


def parse_axis(value: str) -> Axis:
    try:
        return Axis[value.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("axis must be x, y, or z") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--cnc-calibration",
        type=Path,
        default=Path("hardware/fmc4030_limit_calibration.json"),
    )
    parser.add_argument("--leader-port")
    parser.add_argument("--follower-port")
    parser.add_argument("--cnc-ip")
    parser.add_argument("--cnc-port", type=int)
    parser.add_argument(
        "--joints",
        nargs="+",
        help="Semantic joints to mirror (default: every joint calibrated on both arms).",
    )
    parser.add_argument(
        "--drawer",
        type=bounded_integer(1, 9),
        help="Use a fixed drawer for testing (default: choose 1 through 9 randomly).",
    )
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible random drawers.")
    parser.add_argument("--fps", type=positive_float, default=30.0)
    parser.add_argument("--max-velocity-deg", type=positive_float, default=30.0)
    parser.add_argument("--startup-velocity-deg", type=positive_float, default=10.0)
    parser.add_argument("--startup-tolerance-deg", type=positive_float, default=5.0)
    parser.add_argument("--startup-timeout-seconds", type=positive_float, default=60.0)
    parser.add_argument(
        "--startup-countdown",
        type=nonnegative_float,
        default=3.0,
        help="Warning delay before torque is enabled; no confirmation is requested.",
    )
    parser.add_argument("--smoothing", type=positive_float, default=0.25)
    parser.add_argument("--acceleration", type=bounded_integer(0, 254), default=50)
    parser.add_argument("--torque-limit", type=bounded_integer(0, 1000), default=500)
    parser.add_argument("--retries", type=nonnegative_integer, default=2)
    parser.add_argument("--cnc-speed", type=positive_float, default=200.0)
    parser.add_argument("--cnc-acceleration", type=positive_float, default=200.0)
    parser.add_argument("--cnc-deceleration", type=positive_float, default=200.0)
    parser.add_argument("--cnc-timeout", type=positive_float, default=3.0)
    parser.add_argument("--cnc-settle-seconds", type=nonnegative_float, default=2.0)
    parser.add_argument(
        "--cnc-order",
        nargs=3,
        type=parse_axis,
        default=[Axis.Z, Axis.X, Axis.Y],
        help="CNC packet order; all axes execute in parallel (default: z x y).",
    )
    return parser.parse_args(argv)


class SpaceKeyReader:
    """Read SPACE immediately from a POSIX terminal and restore it on every exit."""

    def __init__(self) -> None:
        self.stream = sys.stdin
        self.fd: int | None = None
        self.saved_attributes: list[object] | None = None

    def __enter__(self) -> SpaceKeyReader:
        if not self.stream.isatty():
            raise RuntimeError("SPACE-key episode control requires an interactive terminal.")
        self.fd = self.stream.fileno()
        self.saved_attributes = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def pressed(self) -> bool:
        assert self.fd is not None
        pressed = False
        while select.select([self.fd], [], [], 0.0)[0]:
            pressed = self.stream.read(1) == " " or pressed
        return pressed

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self.fd is not None and self.saved_attributes is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved_attributes)


def _countdown(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print(
            f"\rStarting automatically in {remaining:4.1f} s; Ctrl+C cancels.", end="", flush=True
        )
        time.sleep(min(0.1, remaining))
    if seconds:
        print("\rStarting now.                                      ")


def _print_transition(move: CNCMove) -> None:
    targets = ", ".join(f"{axis.name}={move.targets_mm[axis]:+.3f} mm" for axis in Axis)
    print(f"CNC command accepted: {move.transition.label} | {targets}")
    print(
        "Follower teleoperation remains active while the CNC moves; "
        f"conservative readiness delay is {move.conservative_duration_seconds:.1f} s."
    )


def _align_follower(
    *,
    args: argparse.Namespace,
    leader: FeetechLeader,
    follower: FeetechFollower,
    joints: tuple[str, ...],
    follower_positions: dict[str, float],
) -> JointSafetyLimiter:
    limits = {
        axis.joint_name: (
            float(axis.calibration["q_min"]),
            float(axis.calibration["q_max"]),
        )
        for axis in follower.axes
    }
    limiter = JointSafetyLimiter(
        limits=limits,
        max_velocity=math.radians(args.startup_velocity_deg),
        previous=follower_positions.copy(),
    )
    tolerance = math.radians(args.startup_tolerance_deg)
    deadline = time.monotonic() + args.startup_timeout_seconds
    period = 1.0 / args.fps
    last_tick = time.monotonic()
    status_width = 0
    print(
        "Hold the leader steady. Aligning follower automatically at "
        f"{args.startup_velocity_deg:.1f} degrees/s."
    )
    while True:
        started = time.monotonic()
        dt = started - last_tick
        last_tick = started
        requested = leader.read_joint_positions()
        limited = limiter.apply(requested, dt)
        follower.write_joint_positions(limited)
        actual = follower.read_joint_positions()
        status_width = print_joint_status(
            joints,
            requested,
            limited,
            actual,
            status_width,
            prefix="ALIGNING | ",
        )
        if startup_alignment_complete(requested, actual, tolerance):
            print(f"\nFollower is within {args.startup_tolerance_deg:.1f} degrees of the leader.")
            limiter.max_velocity = math.radians(args.max_velocity_deg)
            return limiter
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Automatic startup alignment did not converge within "
                f"{args.startup_timeout_seconds:.1f} seconds."
            )
        time.sleep(max(0.0, period - (time.monotonic() - started)))


def run_episode(args: argparse.Namespace, sink: EpisodeSink | None = None) -> None:
    if not 0 < args.smoothing <= 1:
        raise ValueError("--smoothing must be greater than 0 and at most 1")
    if args.cnc_port is not None and not 1 <= args.cnc_port <= 65535:
        raise ValueError("--cnc-port must be between 1 and 65535")
    if len(set(args.cnc_order)) != 3:
        raise ValueError("--cnc-order must contain x, y, and z exactly once")

    leader_calibration, leader_available = load_calibrated_axes(args.leader_calibration)
    follower_calibration, follower_available = load_calibrated_axes(args.follower_calibration)
    if follower_calibration.get("role") != "follower":
        raise RuntimeError("The follower calibration file is not marked as role=follower.")
    joints = resolve_joint_names(leader_available, follower_available, args.joints)
    leader_port = args.leader_port or str(leader_calibration["port"])
    follower_port = args.follower_port or str(follower_calibration["port"])
    if leader_port == follower_port:
        raise RuntimeError("Leader and follower must use different USB serial ports.")

    drawer = choose_drawer(drawer=args.drawer, seed=args.seed)
    sequence = TeleopEpisodeSequence(drawer)
    data_sink = sink or NullEpisodeSink()
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
    cnc = NamedCNCController(
        calibration_path=args.cnc_calibration,
        ip=args.cnc_ip,
        port=args.cnc_port,
        timeout=args.cnc_timeout,
        order=tuple(args.cnc_order),
        speed=args.cnc_speed,
        acceleration=args.cnc_acceleration,
        deceleration=args.cnc_deceleration,
        settle_seconds=args.cnc_settle_seconds,
    )

    completed = False
    data_started = False
    print("\nPhysical Feetech teleoperation episode")
    print(f"Selected drawer: {drawer}")
    print("Sequence: A -> B (open) -> A (pick) -> B (place/close) -> A -> finish")
    print(f"Leader:   {leader_port}")
    print(f"Follower: {follower_port}")
    print(f"Joints: {', '.join(joints)}")
    print(f"CNC: {cnc.ip}:{cnc.port}")
    print("No data will be saved by this test command.")
    print("Support both arms and clear the complete robot/CNC workspace.")
    _countdown(args.startup_countdown)

    try:
        with leader, follower:
            leader.disable_torque()
            follower.disable_torque()
            leader.reset_filter()
            leader_positions = leader.read_joint_positions()
            follower_positions = follower.read_joint_positions()
            print("Initial leader/follower joint difference:")
            for joint in joints:
                error = abs(leader_positions[joint] - follower_positions[joint])
                print(
                    f"  {joint:<12} leader={math.degrees(leader_positions[joint]):+7.2f} deg  "
                    f"follower={math.degrees(follower_positions[joint]):+7.2f} deg  "
                    f"error={math.degrees(error):5.2f} deg"
                )

            follower.configure_runtime(
                acceleration=args.acceleration,
                torque_limit=args.torque_limit,
            )
            follower_positions = follower.seed_goals_from_present_position()
            follower.enable_torque()
            print("Follower torque enabled.")
            limiter = _align_follower(
                args=args,
                leader=leader,
                follower=follower,
                joints=joints,
                follower_positions=follower_positions,
            )

            data_sink.start(drawer=drawer, joints=joints)
            data_started = True
            episode_started = time.monotonic()
            frame_index = 0
            status_width = 0
            last_tick = time.monotonic()
            current_cnc_position = "A"
            cnc_ready_at: float | None = None
            ready_announced = False
            pending_transition = sequence.initial_transition

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnc-episode") as executor:
                pending: Future[CNCMove] | None = executor.submit(
                    cnc.command,
                    pending_transition,
                )
                print("Commanding initial CNC position A in the background...")

                with SpaceKeyReader() as keys:
                    while sequence.phase is not EpisodePhase.COMPLETE:
                        started = time.monotonic()
                        dt = started - last_tick
                        last_tick = started

                        if pending is not None and pending.done():
                            move = pending.result()
                            pending = None
                            current_cnc_position = move.transition.position
                            cnc_ready_at = time.monotonic() + move.conservative_duration_seconds
                            ready_announced = False
                            data_sink.cnc_command(move)
                            print()
                            _print_transition(move)

                        cnc_busy = pending is not None or (
                            cnc_ready_at is not None and time.monotonic() < cnc_ready_at
                        )
                        if not cnc_busy and not ready_announced:
                            ready_announced = True
                            print(f"\nCNC {current_cnc_position} readiness delay complete.")
                            print(sequence.next_instruction)

                        requested = leader.read_joint_positions()
                        limited = limiter.apply(requested, dt)
                        follower.write_joint_positions(limited)
                        actual = follower.read_joint_positions()
                        status_width = print_joint_status(
                            joints,
                            requested,
                            limited,
                            actual,
                            status_width,
                            prefix=f"{sequence.phase.value.upper()} | ",
                        )
                        data_sink.append(
                            EpisodeFrame(
                                index=frame_index,
                                elapsed_seconds=started - episode_started,
                                drawer=drawer,
                                phase=sequence.phase,
                                cnc_position=current_cnc_position,
                                cnc_motion_pending=cnc_busy,
                                leader_positions_rad=requested.copy(),
                                follower_targets_rad=limited.copy(),
                                follower_positions_rad=actual.copy(),
                            )
                        )
                        frame_index += 1

                        if keys.pressed():
                            print()
                            if cnc_busy:
                                print("CNC movement is not ready yet; SPACE was ignored.")
                            else:
                                transition = sequence.advance()
                                if transition is None:
                                    completed = True
                                    break
                                pending_transition = transition
                                pending = executor.submit(cnc.command, transition)
                                cnc_ready_at = None
                                ready_announced = False
                                print(f"Commanding CNC {transition.label} in the background...")

                        period = 1.0 / args.fps
                        time.sleep(max(0.0, period - (time.monotonic() - started)))

            print(f"\nEpisode complete for drawer {drawer}. Recorded data: disabled.")
    finally:
        if data_started and not completed:
            cnc.stop_all()
        cnc.close()
        if data_started:
            data_sink.finish(completed=completed)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_episode(args)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nStopped. Follower torque was disabled; the episode was not completed.")
        return 130
    except Exception as exc:
        print(f"\nTeleoperation episode failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
