#!/usr/bin/env python3
"""Discover and continuously read a USB camera with OpenCV."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class CameraInfo:
    index: int
    width: int
    height: int
    fps: float


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=nonnegative_integer,
        help="OpenCV camera index. If omitted, the first readable camera is selected.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List readable camera indices and exit.",
    )
    parser.add_argument(
        "--max-index",
        type=nonnegative_integer,
        default=5,
        help="Highest camera index checked during discovery (default: 5).",
    )
    parser.add_argument("--width", type=positive_integer, default=640)
    parser.add_argument("--height", type=positive_integer, default=480)
    parser.add_argument("--fps", type=positive_integer, default=30)
    parser.add_argument(
        "--frames",
        type=nonnegative_integer,
        default=0,
        help="Stop after this many frames; zero means run until Ctrl+C or Q.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Read frames and print statistics without opening a preview window.",
    )
    return parser.parse_args(argv)


def _require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed. Install the repository dependencies in .venv first."
        )


def open_camera(index: int, *, width: int, height: int, fps: int):
    """Open one camera, preferring the native AVFoundation backend on macOS."""
    _require_opencv()
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Camera index {index} could not be opened.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def read_first_frame(capture, *, timeout: float = 2.0):
    """Wait briefly for a newly opened camera to produce its first frame."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            return frame
        time.sleep(0.05)
    raise RuntimeError("Camera opened but did not return a frame.")


def camera_info(index: int, capture, frame) -> CameraInfo:
    height, width = frame.shape[:2]
    return CameraInfo(
        index=index,
        width=int(width),
        height=int(height),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
    )


def discover_cameras(*, max_index: int, width: int, height: int, fps: int) -> list[CameraInfo]:
    """Return every camera index that can provide a frame."""
    cameras = []
    for index in range(max_index + 1):
        capture = None
        try:
            capture = open_camera(index, width=width, height=height, fps=fps)
            frame = read_first_frame(capture)
            cameras.append(camera_info(index, capture, frame))
        except RuntimeError:
            pass
        finally:
            if capture is not None:
                capture.release()
    return cameras


def print_cameras(cameras: list[CameraInfo]) -> None:
    if not cameras:
        print("No readable cameras found.")
        return
    print("Readable cameras")
    print("Index  Resolution  Reported FPS")
    print("-----  ----------  ------------")
    for camera in cameras:
        print(
            f"{camera.index:>5}  {camera.width:>4}x{camera.height:<4}  "
            f"{camera.fps:>12.1f}"
        )


def read_camera(args: argparse.Namespace) -> int:
    if args.index is None:
        cameras = discover_cameras(
            max_index=args.max_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        print_cameras(cameras)
        if not cameras:
            raise RuntimeError(
                "macOS reported a USB camera, but OpenCV could not read it. Allow camera "
                "access for Terminal or Codex in System Settings > Privacy & Security > Camera."
            )
        index = cameras[0].index
        print(f"\nSelecting camera index {index}.")
    else:
        index = args.index

    capture = open_camera(index, width=args.width, height=args.height, fps=args.fps)
    window_name = f"Hepha USB camera {index}"
    frames = 0
    report_started = time.monotonic()
    report_frames = 0
    try:
        first_frame = read_first_frame(capture)
        info = camera_info(index, capture, first_frame)
        print(
            f"Reading camera {index}: {info.width}x{info.height}, "
            f"reported {info.fps:.1f} FPS"
        )
        if args.no_display:
            print("Press Ctrl+C to stop.")
        else:
            print("Press Q or Escape in the preview window, or Ctrl+C, to stop.")

        frame = first_frame
        while True:
            frames += 1
            report_frames += 1
            if not args.no_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.frames and frames >= args.frames:
                break

            now = time.monotonic()
            if now - report_started >= 1.0:
                measured_fps = report_frames / (now - report_started)
                print(
                    f"\rFrames: {frames:>8} | measured rate: {measured_fps:>5.1f} FPS",
                    end="",
                    flush=True,
                )
                report_started = now
                report_frames = 0

            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError("Camera stopped returning frames.")
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"\nStopped after {frames} frames.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _require_opencv()
        if args.list:
            cameras = discover_cameras(
                max_index=args.max_index,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            print_cameras(cameras)
            if not cameras and sys.platform == "darwin":
                print(
                    "Allow camera access for Terminal or Codex in System Settings > "
                    "Privacy & Security > Camera, then run this command again.",
                    file=sys.stderr,
                )
            return 0 if cameras else 1
        return read_camera(args)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nStopped.")
        return 130
    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
