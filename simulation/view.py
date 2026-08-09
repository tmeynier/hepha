"""Open any registered Hepha simulation backend in its interactive viewer."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .base import SimulationConfig, parse_backend_options
from .registry import available_backends, create_backend

MJPYTHON_GUARD = "HEPHA_MJPYTHON_ACTIVE"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _find_environment_executable(name: str) -> str:
    sibling = Path(sys.executable).parent / name
    if sibling.is_file():
        return str(sibling)
    return shutil.which(name) or name


def _ensure_mjpython_on_macos() -> None:
    if (
        sys.platform != "darwin"
        or Path(sys.executable).name == "mjpython"
        or os.environ.get(MJPYTHON_GUARD) == "1"
    ):
        return
    launcher = _find_environment_executable("mjpython")
    if Path(launcher).is_file():
        os.environ[MJPYTHON_GUARD] = "1"
        os.execv(launcher, [launcher, "-m", "simulation.view", *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="mujoco", help="Simulation backend name")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Show collision geometry and the head-camera coordinate axes",
    )
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Backend-specific option; may be repeated and accepts JSON values",
    )
    parser.add_argument("--list-backends", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_backends:
        print("\n".join(available_backends()))
        return
    _ensure_mjpython_on_macos()
    config = SimulationConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        render=False,
        viewer=False,
        options=parse_backend_options(args.backend_option),
    )
    try:
        with create_backend(args.backend, config) as backend:
            backend.run_interactive(debug=args.debug)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
