"""State and data interfaces for a physical Feetech/CNC teleoperation episode."""

from __future__ import annotations

import random
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from hardware.cnc_positions import LimitCalibration, load_limit_calibration, resolve_named_position
from hardware.fmc4030 import (
    DEFAULT_MOTION_SETTLE_SECONDS,
    Axis,
    FMC4030Client,
    StopMode,
    expected_motion_seconds,
    move_to_absolute_targets_timed,
)
from hepha_lerobot.workspaces import Workspace


class EpisodePhase(StrEnum):
    """Discrete task phase associated with every future recorded frame."""

    START_A = "start_a"
    OPEN_DRAWER_B = "open_drawer_b"
    PICK_CUBE_A = "pick_cube_a"
    PLACE_CLOSE_B = "place_close_b"
    FINAL_A = "final_a"
    COMPLETE = "complete"


@dataclass(frozen=True)
class CNCTransition:
    """One absolute CNC position command in the episode sequence."""

    phase: EpisodePhase
    position: str
    drawer: int | None

    @property
    def label(self) -> str:
        return self.position if self.drawer is None else f"{self.position} (drawer {self.drawer})"


class TeleopEpisodeSequence:
    """Deterministic A -> B -> A -> B -> A -> complete task sequence."""

    def __init__(self, drawer: int) -> None:
        if not 1 <= drawer <= 9:
            raise ValueError("Drawer must be between 1 and 9.")
        self.drawer = drawer
        self.phase = EpisodePhase.START_A

    @property
    def initial_transition(self) -> CNCTransition:
        return CNCTransition(EpisodePhase.START_A, "A", None)

    def advance(self) -> CNCTransition | None:
        """Advance on SPACE and return a CNC move, or None when complete."""
        if self.phase is EpisodePhase.START_A:
            self.phase = EpisodePhase.OPEN_DRAWER_B
            return CNCTransition(self.phase, Workspace.DRAWERS, self.drawer)
        if self.phase is EpisodePhase.OPEN_DRAWER_B:
            self.phase = EpisodePhase.PICK_CUBE_A
            return CNCTransition(self.phase, Workspace.STORAGE, None)
        if self.phase is EpisodePhase.PICK_CUBE_A:
            self.phase = EpisodePhase.PLACE_CLOSE_B
            return CNCTransition(self.phase, Workspace.DRAWERS, self.drawer)
        if self.phase is EpisodePhase.PLACE_CLOSE_B:
            self.phase = EpisodePhase.FINAL_A
            return CNCTransition(self.phase, Workspace.STORAGE, None)
        if self.phase is EpisodePhase.FINAL_A:
            self.phase = EpisodePhase.COMPLETE
            return None
        raise RuntimeError("The episode is already complete.")

    @property
    def next_instruction(self) -> str:
        if self.phase is EpisodePhase.START_A:
            return f"Press SPACE to move to B and open drawer {self.drawer}."
        if self.phase is EpisodePhase.OPEN_DRAWER_B:
            return "Press SPACE to return to A and pick the cube."
        if self.phase is EpisodePhase.PICK_CUBE_A:
            return f"Press SPACE to move to B, place the cube, and close drawer {self.drawer}."
        if self.phase is EpisodePhase.PLACE_CLOSE_B:
            return "Press SPACE to return to A."
        if self.phase is EpisodePhase.FINAL_A:
            return "Press SPACE to finish the episode."
        return "Episode complete."


def choose_drawer(*, drawer: int | None, seed: int | None) -> int:
    """Use a requested drawer or make one reproducible from an optional seed."""
    if drawer is not None:
        if not 1 <= drawer <= 9:
            raise ValueError("Drawer must be between 1 and 9.")
        return drawer
    return random.Random(seed).randint(1, 9)


@dataclass(frozen=True)
class CNCMove:
    transition: CNCTransition
    targets_mm: dict[Axis, float]
    conservative_duration_seconds: float


class NamedCNCController:
    """Validate and asynchronously-friendly command named absolute CNC positions."""

    def __init__(
        self,
        *,
        calibration_path: Path,
        ip: str | None = None,
        port: int | None = None,
        timeout: float = 3.0,
        order: tuple[Axis, ...] = (Axis.Z, Axis.X, Axis.Y),
        speed: float = 200.0,
        acceleration: float = 200.0,
        deceleration: float = 200.0,
        settle_seconds: float = DEFAULT_MOTION_SETTLE_SECONDS,
        client: FMC4030Client | None = None,
    ) -> None:
        self.calibration: LimitCalibration = load_limit_calibration(calibration_path)
        self.ip = ip or self.calibration.ip
        self.port = port or self.calibration.port
        if (self.ip, self.port) != (self.calibration.ip, self.calibration.port):
            raise ValueError(
                f"Calibration belongs to {self.calibration.ip}:{self.calibration.port}, "
                f"not {self.ip}:{self.port}."
            )
        if len(order) != 3 or set(order) != set(Axis):
            raise ValueError("CNC command order must contain X, Y, and Z exactly once.")
        if min(speed, acceleration, deceleration) <= 0:
            raise ValueError("CNC speed, acceleration, and deceleration must be positive.")
        if settle_seconds < 0:
            raise ValueError("CNC settle time cannot be negative.")
        self.order = list(order)
        self.speed = speed
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.settle_seconds = settle_seconds
        self.client = client or FMC4030Client(self.ip, self.port, timeout)

    def command(self, transition: CNCTransition) -> CNCMove:
        targets = resolve_named_position(
            self.calibration,
            transition.position,
            transition.drawer,
        )
        spans = {
            axis: axis_range.maximum_mm - axis_range.minimum_mm
            for axis, axis_range in self.calibration.axes.items()
        }
        duration = max(
            expected_motion_seconds(
                spans[axis],
                self.speed,
                self.acceleration,
                self.deceleration,
            )
            + self.settle_seconds
            for axis in Axis
        )
        move_to_absolute_targets_timed(
            self.client,
            targets,
            self.order,
            self.speed,
            self.acceleration,
            self.deceleration,
            settle_seconds=self.settle_seconds,
            read_initial_status=False,
            read_final_status=False,
            travel_distance_bounds_mm=spans,
        )
        return CNCMove(transition, targets, duration)

    def close(self) -> None:
        self.client.close()

    def stop_all(self) -> None:
        """Best-effort emergency cleanup after an interrupted episode."""
        self.client.close()
        for axis in self.order:
            with suppress(OSError, RuntimeError):
                self.client.stop(axis, StopMode.IMMEDIATE)
            self.client.close()


@dataclass(frozen=True)
class EpisodeFrame:
    """One control-cycle sample ready for a future LeRobot dataset sink."""

    index: int
    elapsed_seconds: float
    drawer: int
    phase: EpisodePhase
    cnc_position: str
    cnc_motion_pending: bool
    leader_positions_rad: dict[str, float]
    follower_targets_rad: dict[str, float]
    follower_positions_rad: dict[str, float]


class EpisodeSink(Protocol):
    """Storage boundary; a future LeRobot writer can implement these callbacks."""

    def start(self, *, drawer: int, joints: tuple[str, ...]) -> None: ...

    def cnc_command(self, move: CNCMove) -> None: ...

    def append(self, frame: EpisodeFrame) -> None: ...

    def finish(self, *, completed: bool) -> None: ...


class NullEpisodeSink:
    """Current no-storage implementation of the episode data interface."""

    def start(self, *, drawer: int, joints: tuple[str, ...]) -> None:
        del drawer, joints

    def cnc_command(self, move: CNCMove) -> None:
        del move

    def append(self, frame: EpisodeFrame) -> None:
        del frame

    def finish(self, *, completed: bool) -> None:
        del completed
