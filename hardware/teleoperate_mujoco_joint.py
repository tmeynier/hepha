#!/usr/bin/env python3
"""Drive one or more MuJoCo joints from calibrated physical Feetech servos."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

try:
    from .feetech_leader import FeetechLeader, configure_cnc_start_pose
    from .read_feetech_positions import nonnegative_integer, servo_id
except ImportError:  # Direct execution: python hardware/teleoperate_mujoco_joint.py
    from feetech_leader import FeetechLeader, configure_cnc_start_pose
    from read_feetech_positions import nonnegative_integer, servo_id


def smoothing_factor(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("smoothing must be greater than 0 and at most 1")
    return parsed


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def selected_servo_ids(single_id: int | None, multiple_ids: list[int] | None) -> list[int]:
    """Resolve the backward-compatible --id and multi-axis --ids options."""
    if multiple_ids:
        return sorted(set(multiple_ids))
    return [2 if single_id is None else single_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    ids = parser.add_mutually_exclusive_group()
    ids.add_argument(
        "--id",
        type=servo_id,
        help="One calibrated servo ID (default: 2, left shoulder).",
    )
    ids.add_argument(
        "--ids",
        type=servo_id,
        nargs="+",
        help="One or more calibrated servo IDs, for example: --ids 2 4.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("hardware/feetech_calibration.json"),
        help="Calibration JSON path (default: hardware/feetech_calibration.json).",
    )
    parser.add_argument("--port", help="Override the serial port saved in the calibration file.")
    parser.add_argument("--fps", type=positive_integer, default=30)
    parser.add_argument(
        "--smoothing",
        type=smoothing_factor,
        default=0.25,
        help="New-sample weight for exponential smoothing (default: 0.25; 1 disables smoothing).",
    )
    parser.add_argument(
        "--retries",
        type=nonnegative_integer,
        default=2,
        help="Retries after a failed serial operation (default: 2).",
    )
    parser.add_argument("--debug", action="store_true", help="Show MuJoCo debug geometry.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    servo_ids = selected_servo_ids(args.id, args.ids)

    from simulation.view import _ensure_mjpython_on_macos

    _ensure_mjpython_on_macos("hardware.teleoperate_mujoco_joint")

    try:
        import mujoco

        from simulation.backends.mujoco import ACTUATOR_NAMES, MujocoBackend
        from simulation.base import SimulationConfig

        leader = FeetechLeader(
            calibration_path=args.calibration,
            servo_ids=servo_ids,
            port=args.port,
            retries=args.retries,
            smoothing=args.smoothing,
        )
        config = SimulationConfig(fps=args.fps, render=False, viewer=False, debug=args.debug)
        with MujocoBackend(config) as backend:
            leader.validate_mujoco_ranges(
                ACTUATOR_NAMES,
                backend.control_low,
                backend.control_high,
            )
            with leader:
                print("\nPhysical servos -> MuJoCo multi-axis test")
                for servo in leader.servos:
                    print(f"ID {servo.servo_id}: {servo.label} -> {servo.actuator}")
                print(f"Port: {leader.port}   Baud: {leader.baudrate:,}")
                print("The physical servos will only be read; no commands are sent to them.")
                print("Selected arm actuators follow the servos; others remain at home.")
                print("MuJoCo CNC start pose: cnc_x=0 m, cnc_y=0 m, head_z=-0.1 m (minimum).")
                input(
                    "Support the selected physical axes, then press ENTER to disable "
                    "their torque and open MuJoCo: "
                )
                leader.disable_torque()

                base_action = backend.home_action()
                configure_cnc_start_pose(base_action, backend.control_low, ACTUATOR_NAMES)
                leader.reset_filter()
                action = leader.read_action(base_action, ACTUATOR_NAMES)
                backend.data.qpos[backend.qpos_ids] = action
                dof_ids = backend.model.jnt_dofadr[backend.joint_ids].astype(int)
                backend.data.qvel[dof_ids] = 0.0
                backend.send_action(action)
                mujoco.mj_forward(backend.model, backend.data)
                backend.open_viewer(debug=args.debug)

                actuator_indices = {
                    servo.servo_id: ACTUATOR_NAMES.index(servo.actuator) for servo in leader.servos
                }
                print("Move the physical axes slowly. Close the viewer or press Ctrl+C to stop.")
                period = 1.0 / args.fps
                previous_line_width = 0
                while backend.viewer_is_running():
                    started = time.perf_counter()
                    action = leader.read_action(base_action, ACTUATOR_NAMES)
                    backend.send_action(action)
                    backend.step()
                    simulated_positions = backend.joint_positions()
                    status = " | ".join(
                        f"ID {servo.servo_id:>2} "
                        f"raw={leader.last_raw_positions[servo.servo_id]:>4} "
                        f"target={leader.last_targets[servo.actuator]:+.3f} "
                        "sim="
                        f"{float(simulated_positions[actuator_indices[servo.servo_id]]):+.3f} "
                        "rad"
                        for servo in leader.servos
                    )
                    print(f"\r{status:<{previous_line_width}}", end="", flush=True)
                    previous_line_width = max(previous_line_width, len(status))
                    time.sleep(max(0.0, period - (time.perf_counter() - started)))
                print()
        return 0
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nStopped.")
        return 130
    except Exception as exc:
        print(f"\nTeleoperation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
