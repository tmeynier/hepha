"""Named CNC work positions derived from measured safe axis limits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from hardware.fmc4030 import Axis
from hepha_lerobot.workspaces import drawer_grid_offsets

CM_TO_MM = 10.0
DRAWER_SPACING_CM = 5.0


@dataclass(frozen=True)
class AxisRange:
    minimum_mm: float
    midpoint_mm: float
    maximum_mm: float

    def validate(self, axis: Axis) -> None:
        values = (self.minimum_mm, self.midpoint_mm, self.maximum_mm)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{axis.name} calibration contains a non-finite position.")
        if not self.minimum_mm < self.midpoint_mm < self.maximum_mm:
            raise ValueError(
                f"{axis.name} calibration must satisfy safe MIN < MID < safe MAX."
            )


@dataclass(frozen=True)
class LimitCalibration:
    ip: str
    port: int
    axes: dict[Axis, AxisRange]

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("Calibration controller port must be between 1 and 65535.")
        if set(self.axes) != set(Axis):
            raise ValueError("Calibration must contain complete X, Y, and Z limits.")
        for axis, axis_range in self.axes.items():
            axis_range.validate(axis)


def drawer_offsets_mm(drawer: int) -> tuple[float, float]:
    """Return the original drawer-column X and drawer-row Y offsets."""
    delta_x_cm, delta_y_cm = drawer_grid_offsets(drawer, spacing=DRAWER_SPACING_CM)
    return delta_x_cm * CM_TO_MM, delta_y_cm * CM_TO_MM


def load_limit_calibration(path: Path) -> LimitCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("units") != "mm":
        raise ValueError("Unsupported CNC limit-calibration file.")

    controller = payload.get("controller")
    axes_payload = payload.get("axes")
    if not isinstance(controller, dict) or not isinstance(axes_payload, dict):
        raise ValueError("CNC limit calibration is missing controller or axes data.")

    axes: dict[Axis, AxisRange] = {}
    for axis in Axis:
        axis_payload = axes_payload.get(axis.name.lower())
        if not isinstance(axis_payload, dict):
            raise ValueError(f"Calibration is missing the {axis.name} axis.")
        minimum = axis_payload.get("min")
        maximum = axis_payload.get("max")
        if not isinstance(minimum, dict) or not isinstance(maximum, dict):
            raise ValueError(f"Calibration is missing {axis.name} MIN or MAX.")
        minimum_mm = float(minimum["position_mm"])
        maximum_mm = float(maximum["position_mm"])
        midpoint_mm = (minimum_mm + maximum_mm) / 2.0
        axes[axis] = AxisRange(minimum_mm, midpoint_mm, maximum_mm)

    config = LimitCalibration(
        ip=str(controller["ip"]),
        port=int(controller["port"]),
        axes=axes,
    )
    config.validate()
    return config


def resolve_named_position(
    calibration: LimitCalibration,
    position: str,
    drawer: int | None = None,
) -> dict[Axis, float]:
    """Resolve A or drawer-dependent B into absolute millimetres."""
    name = position.upper()
    x = calibration.axes[Axis.X]
    y = calibration.axes[Axis.Y]
    z = calibration.axes[Axis.Z]

    if name == "A":
        if drawer is not None:
            raise ValueError("--drawer is accepted only for position B.")
        targets = {Axis.X: x.midpoint_mm, Axis.Y: y.minimum_mm, Axis.Z: z.maximum_mm}
    elif name == "B":
        if drawer is None:
            raise ValueError("Position B requires --drawer 1 through 9.")
        delta_x_mm, delta_y_mm = drawer_offsets_mm(drawer)
        targets = {
            Axis.X: x.midpoint_mm - delta_x_mm,
            Axis.Y: y.minimum_mm + DRAWER_SPACING_CM * CM_TO_MM + delta_y_mm,
            Axis.Z: z.minimum_mm,
        }
    else:
        raise ValueError("Position must be A or B.")

    for axis, target in targets.items():
        axis_range = calibration.axes[axis]
        if not axis_range.minimum_mm <= target <= axis_range.maximum_mm:
            raise ValueError(
                f"Position {name} targets {axis.name}={target:.3f} mm, outside its "
                f"safe calibrated interval [{axis_range.minimum_mm:.3f}, "
                f"{axis_range.maximum_mm:.3f}] mm."
            )
    return targets
