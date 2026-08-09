"""Helpers shared by thin wrappers around official LeRobot commands."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def find_environment_executable(name: str) -> str:
    sibling = Path(sys.executable).parent / name
    if sibling.is_file():
        return str(sibling)
    return shutil.which(name) or name


def passthrough_arguments(values: list[str] | None) -> list[str]:
    values = list(values or [])
    return values[1:] if values[:1] == ["--"] else values
