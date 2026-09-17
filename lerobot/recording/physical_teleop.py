"""Record physical leader-to-follower teleoperation as a native LeRobot dataset."""

from __future__ import annotations

import argparse
import math
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)

from hardware.calibration import load_calibrated_axes
from hardware.feetech_follower import FeetechFollower
from hardware.feetech_leader import FeetechLeader
from hardware.read_feetech_positions import nonnegative_integer
from hardware.read_usb_camera import open_camera, read_first_frame
from hardware.teleoperate_feetech import (
    JointSafetyLimiter,
    bounded_integer,
    print_joint_status,
    resolve_joint_names,
    startup_alignment_complete,
)
from lerobot.datasets import LeRobotDataset

EPISODE_DECISIONS = ("SAVE", "DISCARD", "RETRY", "QUIT")
KEY_ACTIONS = {
    27: "quit",
    ord("q"): "quit",
    ord("Q"): "quit",
    32: "space",
    ord("s"): "save",
    ord("S"): "save",
    ord("d"): "discard",
    ord("D"): "discard",
    ord("r"): "retry",
    ord("R"): "retry",
}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def smoothing_factor(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be greater than zero and at most one")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo ID.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/hepha_physical_teleop"),
    )
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
        help="Joints to record and command (default: every joint shared by both arms).",
    )
    parser.add_argument("--camera-index", type=nonnegative_integer, default=0)
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--width", type=positive_integer, default=256)
    parser.add_argument("--height", type=positive_integer, default=256)
    parser.add_argument("--fps", type=positive_integer, default=30)
    parser.add_argument("--episodes", type=positive_integer, default=20)
    parser.add_argument("--episode-seconds", type=positive_float, default=60.0)
    parser.add_argument("--task", default="Teleoperate the Hepha robot")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--smoothing", type=smoothing_factor, default=0.25)
    parser.add_argument(
        "--max-velocity-deg",
        type=positive_float,
        default=30.0,
    )
    parser.add_argument(
        "--startup-velocity-deg",
        type=positive_float,
        default=10.0,
    )
    parser.add_argument(
        "--startup-tolerance-deg",
        type=positive_float,
        default=5.0,
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=positive_float,
        default=60.0,
    )
    parser.add_argument("--acceleration", type=bounded_integer(0, 254), default=50)
    parser.add_argument("--torque-limit", type=bounded_integer(0, 1000), default=500)
    parser.add_argument("--retries", type=nonnegative_integer, default=2)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the physical camera while aligning and recording (default: true).",
    )
    parser.add_argument(
        "--preview-width",
        type=positive_integer,
        default=1280,
        help="Camera preview window width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--preview-height",
        type=positive_integer,
        default=720,
        help="Camera preview window height in pixels (default: 720).",
    )
    parser.add_argument(
        "--preview-fullscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the camera preview fullscreen (default: false).",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.countdown < 0:
        raise ValueError("--countdown must be non-negative")
    if not args.camera_name:
        raise ValueError("--camera-name cannot be empty")
    if args.root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Dataset root already exists: {args.root}. Use --overwrite to replace it."
        )


def prepare_dataset_root(root: Path, *, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset root already exists: {root}.")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)


def joint_feature_names(joints: tuple[str, ...]) -> dict[str, type]:
    return {f"{joint}.pos": float for joint in joints}


def create_physical_dataset(
    *,
    repo_id: str,
    root: Path,
    joints: tuple[str, ...],
    camera_name: str,
    width: int,
    height: int,
    fps: int,
    use_videos: bool,
) -> LeRobotDataset:
    """Create a camera-and-arm dataset with no CNC features."""
    joint_features = joint_feature_names(joints)
    observation_features = {
        **joint_features,
        camera_name: (height, width, 3),
    }
    features = combine_feature_dicts(
        hw_to_dataset_features(joint_features, ACTION, use_video=use_videos),
        hw_to_dataset_features(observation_features, OBS_STR, use_video=use_videos),
    )
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type="hepha_physical_feetech",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=2,
        batch_encoding_size=1,
        streaming_encoding=False,
        encoder_threads=2,
    )


def vector_values(positions: dict[str, float], joints: tuple[str, ...]) -> dict[str, float]:
    return {f"{joint}.pos": float(positions[joint]) for joint in joints}


def add_physical_frame(
    dataset: LeRobotDataset,
    *,
    joints: tuple[str, ...],
    follower_positions: dict[str, float],
    follower_commands: dict[str, float],
    camera_name: str,
    image: np.ndarray,
    task: str,
) -> None:
    observation_values = {
        **vector_values(follower_positions, joints),
        camera_name: image,
    }
    action_values = vector_values(follower_commands, joints)
    dataset.add_frame(
        {
            **build_dataset_frame(dataset.features, observation_values, prefix=OBS_STR),
            **build_dataset_frame(dataset.features, action_values, prefix=ACTION),
            "task": task,
        }
    )


def camera_frame(capture: Any, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    ok, bgr = capture.read()
    if not ok or bgr is None or not bgr.size:
        raise RuntimeError("USB camera stopped returning frames.")
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def configure_preview_window(
    window_name: str,
    *,
    width: int,
    height: int,
    fullscreen: bool,
) -> None:
    """Create a large, resizable preview without changing recorded image dimensions."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, width, height)


def fit_preview_frame(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Letterbox a camera frame into a fixed-size interface canvas."""
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    resized_width = max(1, round(frame_width * scale))
    resized_height = max(1, round(frame_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def draw_interface_overlay(
    frame: np.ndarray,
    *,
    title: str,
    message: str,
    controls: str,
    recording: bool = False,
) -> np.ndarray:
    """Draw readable recording state and controls over a camera canvas."""
    output = frame.copy()
    overlay = output.copy()
    height, width = output.shape[:2]
    cv2.rectangle(overlay, (0, 0), (width, 108), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 64), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0, output)
    accent = (40, 40, 230) if recording else (40, 190, 80)
    if recording:
        cv2.circle(output, (31, 34), 11, accent, -1)
    cv2.putText(
        output,
        title,
        (54 if recording else 24, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        message,
        (24, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        controls,
        (24, height - 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


@dataclass
class RecordingInterface:
    """Large live-camera UI used for every interactive recording decision."""

    capture: Any
    window_name: str
    record_width: int
    record_height: int
    preview_width: int
    preview_height: int
    enabled: bool = True
    fullscreen: bool = False

    def open(self, first_frame: np.ndarray) -> None:
        if not self.enabled:
            return
        configure_preview_window(
            self.window_name,
            width=self.preview_width,
            height=self.preview_height,
            fullscreen=self.fullscreen,
        )
        self.present(
            first_frame,
            title="Hepha physical recording",
            message="Camera and recording controls are ready.",
            controls="SPACE: continue    Q / ESC: quit",
        )
        # macOS needs one rendered frame before a window can enter fullscreen.
        if self.fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        return camera_frame(
            self.capture,
            width=self.record_width,
            height=self.record_height,
        )

    def present(
        self,
        frame: np.ndarray,
        *,
        title: str,
        message: str,
        controls: str,
        recording: bool = False,
    ) -> str | None:
        if not self.enabled:
            return None
        canvas = fit_preview_frame(
            frame,
            width=self.preview_width,
            height=self.preview_height,
        )
        canvas = draw_interface_overlay(
            canvas,
            title=title,
            message=message,
            controls=controls,
            recording=recording,
        )
        cv2.imshow(self.window_name, canvas)
        return KEY_ACTIONS.get(cv2.waitKey(1) & 0xFF)

    def confirm(self, *, title: str, message: str) -> None:
        if not self.enabled:
            input(f"{message} Press ENTER to continue: ")
            return
        while True:
            frame, _ = self.read()
            action = self.present(
                frame,
                title=title,
                message=message,
                controls="SPACE: continue    Q / ESC: quit",
            )
            if action == "space":
                return
            if action == "quit":
                raise KeyboardInterrupt("recording interface closed")

    def countdown(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                return
            frame, _ = self.read()
            action = self.present(
                frame,
                title="Recording starts soon",
                message=f"Starting in {remaining:.1f} seconds",
                controls="Q / ESC: quit",
            )
            if action == "quit":
                raise KeyboardInterrupt("recording interface closed")
            time.sleep(min(0.01, remaining))

    def review_episode(self, frame_count: int) -> str:
        if not self.enabled:
            return prompt_episode_decision()
        while True:
            frame, _ = self.read()
            action = self.present(
                frame,
                title="Episode complete",
                message=f"Captured {frame_count} frames. Choose what to do with this episode.",
                controls="S: save    D: discard    R: retry    Q / ESC: quit",
            )
            if action in {"save", "discard", "retry", "quit"}:
                return action.upper()


def prompt_episode_decision() -> str:
    while True:
        decision = input("Save this episode? Type SAVE, DISCARD, RETRY, or QUIT: ").strip().upper()
        if decision in EPISODE_DECISIONS:
            return decision
        print(f"Choose one of: {', '.join(EPISODE_DECISIONS)}")


def clear_unsaved_episode(dataset: Any) -> None:
    buffer = getattr(getattr(dataset, "writer", None), "episode_buffer", None)
    if isinstance(buffer, dict) and int(buffer.get("size", 0)) > 0:
        dataset.clear_episode_buffer()


def align_follower(
    *,
    args: argparse.Namespace,
    leader: FeetechLeader,
    follower: FeetechFollower,
    joints: tuple[str, ...],
    interface: RecordingInterface,
) -> JointSafetyLimiter:
    leader.reset_filter()
    follower_positions = follower.read_joint_positions()
    interface.confirm(
        title="Prepare follower alignment",
        message="Keep an emergency cutoff ready and hold the leader steady.",
    )
    follower.configure_runtime(
        acceleration=args.acceleration,
        torque_limit=args.torque_limit,
    )
    follower_positions = follower.seed_goals_from_present_position()
    follower.enable_torque()
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
    previous_status_width = 0
    last_tick = time.monotonic()
    print(
        f"Aligning follower at {args.startup_velocity_deg:.1f} degrees/s. "
        "Hold the leader steady."
    )
    while True:
        started = time.monotonic()
        dt = started - last_tick
        last_tick = started
        requested = leader.read_joint_positions()
        commanded = limiter.apply(requested, dt)
        follower.write_joint_positions(commanded)
        actual = follower.read_joint_positions()
        preview, _ = interface.read()
        max_error_degrees = max(
            math.degrees(abs(requested[joint] - actual[joint])) for joint in joints
        )
        action = interface.present(
            preview,
            title="Aligning follower",
            message=(
                f"Maximum joint error: {max_error_degrees:.1f} deg. "
                f"Target: {args.startup_tolerance_deg:.1f} deg."
            ),
            controls="Hold leader steady    Q / ESC: emergency stop",
        )
        if action == "quit":
            raise KeyboardInterrupt("recording interface closed")
        previous_status_width = print_joint_status(
            joints,
            requested,
            commanded,
            actual,
            previous_status_width,
            prefix="ALIGNING | ",
        )
        if startup_alignment_complete(requested, actual, tolerance):
            print("\nFollower aligned. Physical dataset recording is ready.")
            limiter.max_velocity = math.radians(args.max_velocity_deg)
            limiter.previous = commanded.copy()
            return limiter
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Automatic startup alignment did not converge within "
                f"{args.startup_timeout_seconds:.1f} seconds."
            )
        time.sleep(max(0.0, period - (time.monotonic() - started)))


def record_episode(
    *,
    args: argparse.Namespace,
    leader: FeetechLeader,
    follower: FeetechFollower,
    limiter: JointSafetyLimiter,
    dataset: LeRobotDataset,
    joints: tuple[str, ...],
    interface: RecordingInterface,
) -> int:
    frame_count = max(1, round(args.episode_seconds * args.fps))
    period = 1.0 / args.fps
    last_tick = time.monotonic()
    previous_status_width = 0
    for frame_index in range(frame_count):
        started = time.monotonic()
        dt = started - last_tick
        last_tick = started
        leader_positions = leader.read_joint_positions()
        commanded = limiter.apply(leader_positions, dt)
        sent = follower.write_joint_positions(commanded)
        follower_positions = follower.read_joint_positions()
        preview, rgb = interface.read()
        add_physical_frame(
            dataset,
            joints=joints,
            follower_positions=follower_positions,
            follower_commands=sent,
            camera_name=args.camera_name,
            image=rgb,
            task=args.task,
        )
        elapsed = (frame_index + 1) / args.fps
        action = interface.present(
            preview,
            title=f"RECORDING  Episode frame {frame_index + 1}/{frame_count}",
            message=f"Elapsed: {elapsed:.1f} / {args.episode_seconds:.1f} seconds",
            controls="SPACE: finish episode now    Q / ESC: stop recording session",
            recording=True,
        )
        status = (
            f"frame {frame_index + 1:>5}/{frame_count} | "
            f"elapsed={elapsed:6.1f}/{args.episode_seconds:.1f} s"
        )
        print(f"\r{status:<{previous_status_width}}", end="", flush=True)
        previous_status_width = max(previous_status_width, len(status))
        if action == "space":
            print("\nEpisode finished early from the camera interface.")
            return frame_index + 1
        if action == "quit":
            raise KeyboardInterrupt("recording interface closed")
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    print()
    return frame_count


def record_dataset(args: argparse.Namespace) -> Path:
    validate_args(args)
    leader_calibration, leader_axes = load_calibrated_axes(args.leader_calibration)
    follower_calibration, follower_axes = load_calibrated_axes(args.follower_calibration)
    if follower_calibration.get("role") != "follower":
        raise RuntimeError("The follower calibration file is not marked as role=follower.")
    joints = resolve_joint_names(leader_axes, follower_axes, args.joints)
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
    capture = open_camera(
        args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    first_frame = read_first_frame(capture)
    window_name = f"Hepha recording camera {args.camera_index}"
    interface = RecordingInterface(
        capture=capture,
        window_name=window_name,
        record_width=args.width,
        record_height=args.height,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        enabled=args.preview,
        fullscreen=args.preview_fullscreen,
    )
    interface.open(first_frame)
    dataset = None
    saved_episodes = 0
    try:
        with leader, follower:
            print("\nPhysical Feetech leader -> follower LeRobot recording")
            print(f"Leader:   {leader.port}")
            print(f"Follower: {follower.port}")
            print(f"Joints: {', '.join(joints)}")
            print(f"Camera: index {args.camera_index} -> observation.images.{args.camera_name}")
            if args.preview:
                preview_mode = (
                    "fullscreen"
                    if args.preview_fullscreen
                    else f"{args.preview_width}x{args.preview_height} resizable window"
                )
                print(f"Camera preview: {preview_mode}; press Q or Escape to stop.")
            print("Dataset contains arm joints and camera RGB only; CNC data is excluded.")
            interface.confirm(
                title="Arm safety confirmation",
                message="Support both arms. SPACE disables torque on the selected servos.",
            )
            leader.disable_torque()
            follower.disable_torque()
            prepare_dataset_root(args.root, overwrite=args.overwrite)
            dataset = create_physical_dataset(
                repo_id=args.repo_id,
                root=args.root,
                joints=joints,
                camera_name=args.camera_name,
                width=args.width,
                height=args.height,
                fps=args.fps,
                use_videos=not args.no_video,
            )
            limiter = align_follower(
                args=args,
                leader=leader,
                follower=follower,
                joints=joints,
                interface=interface,
            )

            attempt = 0
            while saved_episodes < args.episodes:
                print(
                    f"\nEpisode {saved_episodes + 1}/{args.episodes}, attempt {attempt + 1}. "
                    "Place the scene and leader at the desired starting state."
                )
                interface.confirm(
                    title=f"Episode {saved_episodes + 1} ready",
                    message="Arrange the scene and leader, then press SPACE to start.",
                )
                interface.countdown(args.countdown)
                frames = record_episode(
                    args=args,
                    leader=leader,
                    follower=follower,
                    limiter=limiter,
                    dataset=dataset,
                    joints=joints,
                    interface=interface,
                )
                decision = interface.review_episode(frames)
                if decision == "SAVE":
                    dataset.save_episode()
                    saved_episodes += 1
                    attempt += 1
                    print(f"Saved episode {saved_episodes} ({frames} frames).")
                else:
                    clear_unsaved_episode(dataset)
                    if decision == "QUIT":
                        break
                    if decision == "DISCARD":
                        attempt += 1
                        print("Discarded episode.")
                    else:
                        print("Discarded episode; retrying.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if dataset is not None:
            active_error = sys.exception()
            try:
                clear_unsaved_episode(dataset)
                dataset.finalize()
            except Exception:
                if active_error is None:
                    raise

    if args.push_to_hub and saved_episodes:
        dataset.push_to_hub(tags=["hepha", "physical", "feetech", "teleoperation"])
        print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")
    print(f"Recorded {saved_episodes} episode(s) at {args.root}.")
    return args.root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        dataset_root = record_dataset(args)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nStopped. Follower torque was disabled and the unsaved episode discarded.")
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Physical recording failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"LeRobot physical teleoperation dataset ready at {dataset_root}")


if __name__ == "__main__":
    main()
