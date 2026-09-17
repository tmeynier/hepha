"""Run a trained LeRobot ACT policy on the calibrated physical Hepha follower."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from hepha_lerobot.evaluation.conditioned_rollout import _load_policy
from hepha_lerobot.recording.physical_teleop import (
    RecordingInterface,
    camera_frame,
    positive_float,
    positive_integer,
)
from hepha_lerobot.training.train import resolve_device
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from hardware.axes import AXES
from hardware.feetech_follower import FeetechFollower
from hardware.read_feetech_positions import nonnegative_integer
from hardware.read_usb_camera import open_camera, read_first_frame
from hardware.teleoperate_feetech import JointSafetyLimiter, bounded_integer

CANONICAL_JOINTS = tuple(AXES[servo_id].joint_name for servo_id in sorted(AXES))
DEFAULT_TASK = "Teleoperate the Hepha robot"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Local pretrained_model directory or Hugging Face model repo ID.",
    )
    parser.add_argument(
        "--follower-calibration",
        type=Path,
        default=Path("hardware/feetech_follower_calibration.json"),
    )
    parser.add_argument("--follower-port")
    parser.add_argument("--camera-index", type=nonnegative_integer, default=0)
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--width", type=positive_integer, default=256)
    parser.add_argument("--height", type=positive_integer, default=256)
    parser.add_argument("--fps", type=positive_integer, default=30)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this duration; zero runs until Q, Escape, or Ctrl+C.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--n-action-steps",
        type=positive_integer,
        default=None,
        help="Override actions consumed from each ACT chunk before replanning.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Optional ACT temporal ensembling; requires --n-action-steps 1.",
    )
    parser.add_argument(
        "--max-velocity-deg",
        type=positive_float,
        default=10.0,
        help="Maximum commanded joint velocity in degrees/s (default: 10).",
    )
    parser.add_argument("--acceleration", type=bounded_integer(0, 254), default=30)
    parser.add_argument("--torque-limit", type=bounded_integer(0, 1000), default=300)
    parser.add_argument("--retries", type=nonnegative_integer, default=2)
    parser.add_argument(
        "--watchdog-seconds",
        type=positive_float,
        default=2.0,
        help="Disable torque if one camera/inference/control cycle exceeds this time.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Keep follower torque disabled and display policy predictions only.",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--preview-width", type=positive_integer, default=1280)
    parser.add_argument("--preview-height", type=positive_integer, default=720)
    parser.add_argument(
        "--preview-fullscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def _policy_reference(value: str) -> str | Path:
    local = Path(value).expanduser()
    return local.resolve() if local.exists() else value


def _feature_shape(features: dict[str, Any] | None, key: str) -> tuple[int, ...] | None:
    if not features or key not in features:
        return None
    return tuple(int(value) for value in features[key].shape)


def validate_physical_policy_features(
    policy_config: Any,
    *,
    camera_name: str,
    width: int,
    height: int,
    joint_count: int = len(CANONICAL_JOINTS),
) -> None:
    """Reject checkpoints whose physical input/output schema cannot match the recorder."""
    if policy_config.type != "act":
        raise ValueError(
            f"Physical rollout currently supports an original ACT policy, not "
            f"{policy_config.type!r}."
        )
    state_shape = _feature_shape(policy_config.input_features, OBS_STATE)
    if state_shape != (joint_count,):
        raise ValueError(
            f"Policy expects {OBS_STATE} shape {state_shape}; physical Hepha requires "
            f"({joint_count},)."
        )
    image_key = f"{OBS_IMAGES}.{camera_name}"
    image_shape = _feature_shape(policy_config.input_features, image_key)
    accepted_image_shapes = {(3, height, width), (height, width, 3)}
    if image_shape not in accepted_image_shapes:
        raise ValueError(
            f"Policy expects {image_key} shape {image_shape}; this rollout supplies "
            f"RGB {width}x{height}."
        )
    action_shape = _feature_shape(policy_config.output_features, ACTION)
    if action_shape != (joint_count,):
        raise ValueError(
            f"Policy produces {ACTION} shape {action_shape}; physical Hepha requires "
            f"({joint_count},)."
        )


def build_physical_policy_observation(
    positions: dict[str, float],
    rgb: np.ndarray,
    *,
    camera_name: str,
    joints: tuple[str, ...] = CANONICAL_JOINTS,
) -> dict[str, np.ndarray]:
    missing = sorted(set(joints) - set(positions))
    if missing:
        raise ValueError(f"Follower observation is missing joints: {missing}")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Physical camera observation must be HWC uint8 RGB.")
    return {
        OBS_STATE: np.asarray([positions[joint] for joint in joints], dtype=np.float32),
        f"{OBS_IMAGES}.{camera_name}": rgb,
    }


def action_targets(
    action: torch.Tensor | np.ndarray,
    *,
    joints: tuple[str, ...] = CANONICAL_JOINTS,
) -> dict[str, float]:
    if isinstance(action, torch.Tensor):
        values = action.squeeze(0).detach().cpu().numpy()
    else:
        values = np.asarray(action).squeeze()
    if values.shape != (len(joints),):
        raise ValueError(
            f"Policy returned action shape {values.shape}; expected ({len(joints)},)."
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Policy returned a non-finite physical joint target.")
    return {joint: float(value) for joint, value in zip(joints, values, strict=True)}


def reset_policy_state(policy: Any, preprocessor: Any, postprocessor: Any) -> None:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


def predict_targets(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    positions: dict[str, float],
    rgb: np.ndarray,
    camera_name: str,
    device: str,
    task: str,
) -> dict[str, float]:
    observation = build_physical_policy_observation(
        positions,
        rgb,
        camera_name=camera_name,
    )
    batch = prepare_observation_for_inference(
        observation,
        torch.device(device),
        task=task,
        robot_type="hepha_physical_feetech",
    )
    with torch.inference_mode():
        batch = preprocessor(batch)
        action = postprocessor(policy.select_action(batch))
    return action_targets(action)


def _maximum_error_degrees(
    first: dict[str, float], second: dict[str, float]
) -> float:
    return max(math.degrees(abs(first[joint] - second[joint])) for joint in CANONICAL_JOINTS)


def run_physical_rollout(args: argparse.Namespace) -> None:
    if args.duration_seconds < 0:
        raise ValueError("--duration-seconds cannot be negative")
    if not args.camera_name:
        raise ValueError("--camera-name cannot be empty")

    device = resolve_device(args.device)
    policy_ref = _policy_reference(args.policy_path)
    print(f"Loading policy {args.policy_path} on {device} before connecting hardware...")
    policy, policy_config, preprocessor, postprocessor = _load_policy(
        policy_ref,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    validate_physical_policy_features(
        policy_config,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
    )
    print(
        "Policy ready: "
        f"chunk_size={policy_config.chunk_size}, "
        f"n_action_steps={policy_config.n_action_steps}, "
        f"camera={args.camera_name}, joints={len(CANONICAL_JOINTS)}"
    )

    follower = FeetechFollower(
        calibration_path=args.follower_calibration,
        joint_names=CANONICAL_JOINTS,
        port=args.follower_port,
        retries=args.retries,
    )
    capture = open_camera(
        args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    first_frame = read_first_frame(capture)
    interface = RecordingInterface(
        capture=capture,
        window_name=f"Hepha ACT physical rollout camera {args.camera_index}",
        record_width=args.width,
        record_height=args.height,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        enabled=args.preview,
        fullscreen=args.preview_fullscreen,
    )
    interface.open(first_frame)

    try:
        with follower:
            if follower.joint_names != CANONICAL_JOINTS:
                raise RuntimeError(
                    "Follower calibration joint order does not match the physical training "
                    f"dataset: {follower.joint_names}"
                )
            interface.confirm(
                title="Support the follower arms",
                message="Support both arms. SPACE disables torque for the policy check.",
            )
            follower.disable_torque()
            positions = follower.read_joint_positions()
            preview, rgb = camera_frame(capture, width=args.width, height=args.height)
            reset_policy_state(policy, preprocessor, postprocessor)
            warmup_started = time.perf_counter()
            warmup_targets = predict_targets(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                positions=positions,
                rgb=rgb,
                camera_name=args.camera_name,
                device=device,
                task=args.task,
            )
            warmup_seconds = time.perf_counter() - warmup_started
            initial_error = _maximum_error_degrees(warmup_targets, positions)
            print(
                f"Torque-disabled policy check completed in {warmup_seconds * 1000:.1f} ms; "
                f"largest raw target change is {initial_error:.1f} degrees."
            )

            limits = {
                axis.joint_name: (
                    float(axis.calibration["q_min"]),
                    float(axis.calibration["q_max"]),
                )
                for axis in follower.axes
            }
            limiter = JointSafetyLimiter(
                limits=limits,
                max_velocity=math.radians(args.max_velocity_deg),
                previous=positions.copy(),
            )

            if args.dry_run:
                print(
                    "DRY RUN: follower torque remains disabled. Support the unpowered arms. "
                    "Press Q, Escape, or Ctrl+C to stop."
                )
            else:
                interface.confirm(
                    title="ACT rollout safety confirmation",
                    message=(
                        "Clear the workspace and keep power removal ready. "
                        "SPACE enables follower torque."
                    ),
                )
                follower.configure_runtime(
                    acceleration=args.acceleration,
                    torque_limit=args.torque_limit,
                )
                positions = follower.seed_goals_from_present_position()
                limiter.previous = positions.copy()
                follower.enable_torque()
                print(
                    "Follower torque enabled. "
                    f"Velocity limit={args.max_velocity_deg:.1f} deg/s, "
                    f"torque limit={args.torque_limit}."
                )

            reset_policy_state(policy, preprocessor, postprocessor)
            period = 1.0 / args.fps
            started_at = time.monotonic()
            last_tick = started_at
            previous_status_width = 0
            while (
                args.duration_seconds == 0
                or time.monotonic() - started_at < args.duration_seconds
            ):
                cycle_started = time.monotonic()
                dt = min(max(0.0, cycle_started - last_tick), 2.0 * period)
                last_tick = cycle_started
                positions = follower.read_joint_positions()
                preview, rgb = camera_frame(capture, width=args.width, height=args.height)
                requested = predict_targets(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    positions=positions,
                    rgb=rgb,
                    camera_name=args.camera_name,
                    device=device,
                    task=args.task,
                )
                if args.dry_run:
                    limiter.previous = positions.copy()
                commanded = limiter.apply(requested, dt)
                if not args.dry_run:
                    follower.write_joint_positions(commanded)

                cycle_seconds = time.monotonic() - cycle_started
                if cycle_seconds > args.watchdog_seconds:
                    raise RuntimeError(
                        f"Physical rollout watchdog expired: one cycle took "
                        f"{cycle_seconds:.3f} s (limit {args.watchdog_seconds:.3f} s)."
                    )
                target_error = _maximum_error_degrees(requested, positions)
                command_step = _maximum_error_degrees(commanded, positions)
                mode = "DRY RUN" if args.dry_run else "ACTIVE"
                status = (
                    f"{mode} | cycle={cycle_seconds * 1000:6.1f} ms | "
                    f"policy delta={target_error:6.1f} deg | "
                    f"safe step={command_step:5.2f} deg"
                )
                print(f"\r{status:<{previous_status_width}}", end="", flush=True)
                previous_status_width = max(previous_status_width, len(status))
                action = interface.present(
                    preview,
                    title=f"Hepha ACT rollout — {mode}",
                    message=(
                        f"Cycle {cycle_seconds * 1000:.0f} ms | "
                        f"policy delta {target_error:.1f} deg | "
                        f"safe step {command_step:.2f} deg"
                    ),
                    controls="Q / ESC: stop and disable follower torque",
                    recording=not args.dry_run,
                )
                if action == "quit":
                    break
                time.sleep(max(0.0, period - (time.monotonic() - cycle_started)))
            print()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_physical_rollout(args)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nStopped. Follower torque was disabled.")
        return 130
    except Exception as exc:
        print(f"\nPhysical policy rollout failed: {exc}", file=sys.stderr)
        return 1
    print("Physical policy rollout stopped. Follower torque was disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
