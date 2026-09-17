#!/usr/bin/env python3
"""Compatibility command for simultaneous leader-arm range calibration."""

from __future__ import annotations

try:
    from .calibrate_feetech import run
except ImportError:  # Direct execution: python hardware/calibrate_feetech_positions.py
    from calibrate_feetech import run


def main() -> int:
    return run("leader")


if __name__ == "__main__":
    raise SystemExit(main())
