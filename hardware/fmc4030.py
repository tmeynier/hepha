"""FMC4030 protocol, status, homing, and discrete-position support.

The binary layouts in this module follow the controller packets supplied for
this machine.  Keep protocol handling separate from the interactive CLI so it
can be tested without opening a socket or moving hardware.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path

DEFAULT_IP = "192.168.0.30"
DEFAULT_PORT = 8088
DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_STATUS_TIMEOUT_SECONDS = 3.0
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_DELAY_SECONDS = 0.25
STOP_SETTLE_SECONDS = 0.1
OPTIONAL_RESPONSE_TIMEOUT_SECONDS = 0.1
DEFAULT_MOTION_SETTLE_SECONDS = 2.0
DEFAULT_CALIBRATION_STATUS_READ_TIMEOUT_SECONDS = 120.0

CONTROLLER_ADDRESS = 0x01
MOVE_FUNCTION = 0x04
HOME_FUNCTION = 0x06
STOP_FUNCTION = 0x07
STATUS_FUNCTION = 0x03
PARAMETERS_FUNCTION = 0x12

MOVE_PACKET = struct.Struct("<BBBffffB")
HOME_PACKET = struct.Struct("<BBBfffB")
STOP_PACKET = struct.Struct("<BBBB")
STATUS_BODY = struct.Struct("<ffffff9I")
STATUS_PREFIX_SIZE = 4
STATUS_PAYLOAD_SIZE = 0x0294
PARAMETERS_RESPONSE_SIZE = 94

HOME_DONE = 0x08
HOME_STARTING = 0x0A
HOME_NOT_FINISHED = 0x0B


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


class Mode(IntEnum):
    RELATIVE = 1
    ABSOLUTE = 2


class HomeDirection(IntEnum):
    POSITIVE = 1
    NEGATIVE = 2
    CURRENT = 3


class StopMode(IntEnum):
    DECELERATED = 1
    IMMEDIATE = 2


class CommandOutcomeUnknownError(RuntimeError):
    """A command was sent, but no acknowledgement was received."""


@dataclass(frozen=True)
class CNCStatus:
    positions_mm: tuple[float, float, float]
    speeds_mm_s: tuple[float, float, float]
    input_mask: int
    output_mask: int
    negative_limit_mask: int
    positive_limit_mask: int
    run_status: int
    axis_statuses: tuple[int, int, int]
    home_status: int

    def position(self, axis: Axis) -> float:
        return self.positions_mm[int(axis)]

    def speed(self, axis: Axis) -> float:
        return self.speeds_mm_s[int(axis)]

    def negative_limit(self, axis: Axis) -> bool:
        return bool(self.negative_limit_mask & (1 << int(axis)))

    def positive_limit(self, axis: Axis) -> bool:
        return bool(self.positive_limit_mask & (1 << int(axis)))


@dataclass(frozen=True)
class LimitSeekResult:
    axis: Axis
    limit: str
    switch_position_mm: float
    safe_position_mm: float


@dataclass(frozen=True)
class AxisLimitCalibration:
    axis: Axis
    minimum: LimitSeekResult
    maximum: LimitSeekResult


@dataclass(frozen=True)
class AxisPositions:
    min: float
    mid: float
    max: float
    positive_limit_mm: float

    def validate(self, axis_name: str) -> None:
        values = (self.min, self.mid, self.max, self.positive_limit_mm)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{axis_name} positions must be finite numbers.")
        if not self.min < self.mid < self.max <= self.positive_limit_mm:
            raise ValueError(f"{axis_name} must satisfy min < mid < max <= positive_limit_mm.")

    def preset(self, name: str) -> float:
        if name not in {"min", "mid", "max"}:
            raise ValueError("Preset must be min, mid, or max.")
        return float(getattr(self, name))


@dataclass(frozen=True)
class CNCPositionConfig:
    ip: str
    port: int
    axes: dict[str, AxisPositions]
    created_at: str
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported config schema {self.schema_version}.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Config port must be between 1 and 65535.")
        if set(self.axes) != {"x", "y", "z"}:
            raise ValueError("Config must contain exactly x, y, and z axes.")
        for axis_name, positions in self.axes.items():
            positions.validate(axis_name)


def _validate_motion_values(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Motion parameters must be finite numbers.")


def build_move_payload(
    axis: Axis,
    position_mm: float,
    speed_mm_s: float,
    acceleration_mm_s2: float,
    deceleration_mm_s2: float,
    mode: Mode,
) -> bytes:
    _validate_motion_values(position_mm, speed_mm_s, acceleration_mm_s2, deceleration_mm_s2)
    if speed_mm_s <= 0:
        raise ValueError("Speed must be greater than zero.")
    if acceleration_mm_s2 <= 0 or deceleration_mm_s2 <= 0:
        raise ValueError("Acceleration and deceleration must be greater than zero.")
    return MOVE_PACKET.pack(
        CONTROLLER_ADDRESS,
        MOVE_FUNCTION,
        int(axis),
        position_mm,
        speed_mm_s,
        acceleration_mm_s2,
        deceleration_mm_s2,
        int(mode),
    )


def build_home_payload(
    axis: Axis,
    direction: HomeDirection,
    speed_mm_s: float,
    acceleration_mm_s2: float,
    backoff_mm: float,
) -> bytes:
    _validate_motion_values(speed_mm_s, acceleration_mm_s2, backoff_mm)
    if speed_mm_s <= 0 or acceleration_mm_s2 <= 0:
        raise ValueError("Homing speed and acceleration must be greater than zero.")
    if backoff_mm < 0:
        raise ValueError("Homing backoff cannot be negative.")
    return HOME_PACKET.pack(
        CONTROLLER_ADDRESS,
        HOME_FUNCTION,
        int(axis),
        speed_mm_s,
        acceleration_mm_s2,
        backoff_mm,
        int(direction),
    )


def build_stop_payload(axis: Axis, mode: StopMode = StopMode.IMMEDIATE) -> bytes:
    return STOP_PACKET.pack(
        CONTROLLER_ADDRESS,
        STOP_FUNCTION,
        int(axis),
        int(mode),
    )


def parse_status_response(data: bytes) -> CNCStatus:
    required = STATUS_PREFIX_SIZE + STATUS_BODY.size
    if len(data) < required:
        raise ValueError(f"Status response is {len(data)} bytes; expected at least {required}.")
    values = STATUS_BODY.unpack_from(data, STATUS_PREFIX_SIZE)
    return CNCStatus(
        positions_mm=(float(values[0]), float(values[1]), float(values[2])),
        speeds_mm_s=(float(values[3]), float(values[4]), float(values[5])),
        input_mask=int(values[6]),
        output_mask=int(values[7]),
        negative_limit_mask=int(values[8]),
        positive_limit_mask=int(values[9]),
        run_status=int(values[10]),
        axis_statuses=(int(values[11]), int(values[12]), int(values[13])),
        home_status=int(values[14]),
    )


class FMC4030Client:
    def __init__(
        self,
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        status_timeout: float = DEFAULT_STATUS_TIMEOUT_SECONDS,
    ) -> None:
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.status_timeout = status_timeout
        self._connection: socket.socket | None = None
        self._receive_buffer = bytearray()
        self._status_request_pending = False
        self._pending_status_header: bytes | None = None
        self._status_session_initialized = False
        self._status_initialization_pending = False

    def open(
        self,
        attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        """Open one reusable TCP connection to the controller."""
        if self._connection is not None:
            return
        if attempts < 1:
            raise ValueError("Connection attempts must be at least one.")
        last_error: OSError | None = None
        for attempt in range(1, attempts + 1):
            try:
                connection = socket.create_connection(
                    (self.ip, self.port), timeout=self.timeout
                )
                connection.settimeout(self.timeout)
                with suppress(OSError):
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                with suppress(OSError):
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._connection = connection
                self._receive_buffer.clear()
                return
            except OSError as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(retry_delay)
        raise ConnectionError(
            f"Could not connect to {self.ip}:{self.port} after {attempts} attempts: {last_error}"
        ) from last_error

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._receive_buffer.clear()
        self._status_request_pending = False
        self._pending_status_header = None
        self._status_session_initialized = False
        self._status_initialization_pending = False
        if connection is None:
            return
        # This embedded single-client server can keep a gracefully closed
        # session internally even after macOS has removed its TCP socket. An
        # abortive close makes the peer observe teardown immediately so the
        # next CLI process can become the active client.
        with suppress(OSError):
            connection.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_LINGER,
                struct.pack("ii", 1, 0),
            )
        with suppress(OSError):
            connection.close()

    def __enter__(self) -> FMC4030Client:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _connected_socket(self, attempts: int = DEFAULT_RETRY_ATTEMPTS) -> socket.socket:
        self.open(attempts=attempts)
        if self._connection is None:  # pragma: no cover - guarded by open()
            raise ConnectionError("FMC4030 connection was not established.")
        return self._connection

    def _recv_exact(self, size: int) -> bytes:
        connection = self._connected_socket(attempts=1)
        while len(self._receive_buffer) < size:
            chunk = connection.recv(max(1024, size - len(self._receive_buffer)))
            if not chunk:
                raise ConnectionError("Controller closed the TCP connection.")
            self._receive_buffer.extend(chunk)
        result = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return result

    def _discard_exact(self, size: int, timeout: float) -> None:
        connection = self._connected_socket(attempts=1)
        connection.settimeout(timeout)
        try:
            remaining = size
            while remaining:
                chunk_size = min(remaining, 1024)
                chunk = self._recv_exact(chunk_size)
                remaining -= len(chunk)
        finally:
            if self._connection is connection:
                connection.settimeout(self.timeout)

    def check_connection(self) -> None:
        self.open()

    def send_raw_command(
        self,
        payload: bytes,
        response_timeout: float | None = None,
    ) -> bytes:
        connection = self._connected_socket()
        effective_timeout = self.timeout if response_timeout is None else response_timeout
        connection.settimeout(effective_timeout)
        try:
            connection.sendall(payload)
            response = self._recv_exact(len(payload))
        except OSError as exc:
            self.close()
            raise CommandOutcomeUnknownError(
                "The command was sent, but no complete response was received. "
                "The machine may still have moved; its state is unknown."
            ) from exc
        finally:
            if self._connection is connection:
                connection.settimeout(self.timeout)
        if response != payload:
            self.close()
            raise CommandOutcomeUnknownError(
                "The controller response did not echo the command. "
                "The machine may still have moved; its state is unknown."
            )
        return response

    def send_raw_without_response(
        self,
        payload: bytes,
        connection_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    ) -> None:
        """Transmit a command whose controller endpoint sends no acknowledgement."""
        connection = self._connected_socket(attempts=connection_attempts)
        try:
            connection.sendall(payload)
        except OSError:
            self.close()
            raise

    def _skip_delayed_command_echo(self, header: bytes) -> bytes:
        """Discard a delayed home/stop/move echo before a status frame."""
        response_sizes = {
            MOVE_FUNCTION: MOVE_PACKET.size,
            HOME_FUNCTION: HOME_PACKET.size,
            STOP_FUNCTION: STOP_PACKET.size,
        }
        function = header[1] if len(header) >= 2 and header[0] == CONTROLLER_ADDRESS else -1
        response_size = response_sizes.get(function)
        if response_size is None:
            return header
        self._discard_exact(response_size - len(header), self.status_timeout)
        return self._recv_exact(STATUS_PREFIX_SIZE)

    def _initialize_status_session(self, connection: socket.socket) -> None:
        """Prime firmware that ignores status until parameters are read once."""
        if self._status_session_initialized:
            return
        request = bytes((CONTROLLER_ADDRESS, PARAMETERS_FUNCTION))
        if not self._status_initialization_pending:
            connection.sendall(request)
            self._status_initialization_pending = True
        response = self._recv_exact(PARAMETERS_RESPONSE_SIZE)
        if response[0:2] != request:
            raise ValueError(
                f"Unexpected parameters response command: {response[0:2].hex(' ')}."
            )
        self._status_initialization_pending = False
        self._status_session_initialized = True

    def _query_status_frame_once(self) -> bytes:
        request = bytes((CONTROLLER_ADDRESS, STATUS_FUNCTION))
        connection = self._connected_socket(attempts=1)
        connection.settimeout(self.status_timeout)
        try:
            self._initialize_status_session(connection)
            if not self._status_request_pending:
                connection.sendall(request)
                self._status_request_pending = True
            header = self._pending_status_header
            if header is None:
                header = self._recv_exact(STATUS_PREFIX_SIZE)
                for _ in range(4):
                    if header[0:2] == request:
                        break
                    header = self._skip_delayed_command_echo(header)
                if header[0:2] != request:
                    raise ValueError(
                        f"Unexpected status response command: {header[0:2].hex(' ')}."
                    )
                self._pending_status_header = header
            payload_size = int.from_bytes(header[2:4], byteorder="big")
            if payload_size != STATUS_PAYLOAD_SIZE:
                raise ValueError(
                    f"Status payload is {payload_size} bytes; expected {STATUS_PAYLOAD_SIZE}."
                )
            body = self._recv_exact(STATUS_BODY.size)
            self._pending_status_header = None
            self._status_request_pending = False
        except TimeoutError:
            # Keep this socket, any partial bytes, and the single outstanding
            # request. The next retry continues receiving instead of opening a
            # new session that can strand a single-client embedded TCP server.
            raise
        except (OSError, ValueError):
            self.close()
            raise
        finally:
            if self._connection is connection:
                connection.settimeout(self.timeout)

        # The remaining 600 bytes are a file-name table and are irrelevant to
        # motion monitoring. Drain it briefly to keep a persistent stream in
        # sync, but never fail an otherwise complete status sample because the
        # optional table is delayed.
        trailer_size = payload_size - STATUS_BODY.size
        try:
            self._discard_exact(trailer_size, OPTIONAL_RESPONSE_TIMEOUT_SECONDS)
        except OSError:
            self.close()
        return header + body

    def read_status(
        self,
        attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> CNCStatus:
        """Read status, with one retry layer covering connect and transaction."""
        if attempts < 1:
            raise ValueError("Status attempts must be at least one.")
        last_error: OSError | ValueError | None = None
        for attempt in range(1, attempts + 1):
            try:
                self.open(attempts=1)
                return parse_status_response(self._query_status_frame_once())
            except TimeoutError as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(retry_delay)
            except (OSError, ValueError) as exc:
                last_error = exc
                self.close()
                if attempt < attempts:
                    time.sleep(retry_delay)
        raise TimeoutError(
            f"Status query failed after {attempts} attempts: {last_error}"
        ) from last_error

    def _drain_optional_stop_response(self, payload: bytes) -> None:
        connection = self._connected_socket(attempts=1)
        connection.settimeout(OPTIONAL_RESPONSE_TIMEOUT_SECONDS)
        try:
            response = self._recv_exact(len(payload))
        except TimeoutError:
            # The documented stop endpoint may not acknowledge the packet.
            # A partial packet would make the stream ambiguous, so reconnect.
            if self._receive_buffer:
                self.close()
            return
        except OSError:
            self.close()
            return
        finally:
            if self._connection is connection:
                connection.settimeout(self.timeout)
        if response != payload:
            self.close()

    def move(
        self,
        axis: Axis,
        position_mm: float,
        speed_mm_s: float,
        acceleration_mm_s2: float,
        deceleration_mm_s2: float,
        mode: Mode,
    ) -> bytes:
        return self.send_raw_command(
            build_move_payload(
                axis,
                position_mm,
                speed_mm_s,
                acceleration_mm_s2,
                deceleration_mm_s2,
                mode,
            )
        )

    def home(
        self,
        axis: Axis,
        direction: HomeDirection,
        speed_mm_s: float,
        acceleration_mm_s2: float,
        backoff_mm: float,
    ) -> None:
        self.send_raw_without_response(
            build_home_payload(axis, direction, speed_mm_s, acceleration_mm_s2, backoff_mm)
        )

    def stop(self, axis: Axis, mode: StopMode = StopMode.IMMEDIATE) -> None:
        payload = build_stop_payload(axis, mode)
        last_error: OSError | None = None
        for attempt in range(1, DEFAULT_RETRY_ATTEMPTS + 1):
            try:
                self.send_raw_without_response(payload, connection_attempts=1)
                time.sleep(STOP_SETTLE_SECONDS)
                self._drain_optional_stop_response(payload)
                return
            except OSError as exc:
                last_error = exc
                self.close()
                if attempt < DEFAULT_RETRY_ATTEMPTS:
                    time.sleep(DEFAULT_RETRY_DELAY_SECONDS)
        raise ConnectionError(
            f"Could not send {axis.name} stop after {DEFAULT_RETRY_ATTEMPTS} attempts: {last_error}"
        ) from last_error


def read_status_with_retries(
    client: FMC4030Client,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> CNCStatus:
    """Retry a read-only status transaction without repeating a motion command."""
    return client.read_status(attempts=attempts, retry_delay=retry_delay)


def read_status_until_deadline(
    client: FMC4030Client,
    timeout: float,
    retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> CNCStatus:
    """Keep receiving one read-only status request until a total deadline."""
    _validate_motion_values(timeout, retry_delay)
    if timeout <= 0:
        raise ValueError("Status-read timeout must be greater than zero.")
    if retry_delay < 0:
        raise ValueError("Status retry delay cannot be negative.")

    deadline = time.monotonic() + timeout
    original_status_timeout = client.status_timeout
    attempts = 0
    last_error: TimeoutError | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempts += 1
            client.status_timeout = min(original_status_timeout, remaining)
            try:
                return client.read_status(attempts=1)
            except TimeoutError as exc:
                last_error = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(retry_delay, remaining))
    finally:
        client.status_timeout = original_status_timeout

    raise TimeoutError(
        f"Status did not arrive within {timeout:g} seconds after {attempts} "
        f"receive windows. Last error: {last_error}"
    ) from last_error


def wait_for_home(
    client: FMC4030Client,
    axis: Axis,
    timeout: float,
    initial_position_mm: float | None = None,
    poll_interval: float = 0.1,
) -> CNCStatus:
    deadline = time.monotonic() + timeout
    saw_activity = False
    stable_reads = 0
    previous_position: float | None = None
    while time.monotonic() < deadline:
        try:
            status = client.read_status(attempts=1)
        except TimeoutError as exc:
            with suppress(OSError):
                client.stop(axis, StopMode.IMMEDIATE)
            raise TimeoutError(
                f"Lost status while {axis.name} was homing; immediate stop was requested."
            ) from exc
        saw_activity |= status.home_status in {HOME_STARTING, HOME_NOT_FINISHED}
        position = status.position(axis)
        moved = initial_position_mm is not None and abs(position - initial_position_mm) > 0.01
        position_stable = (
            previous_position is not None and abs(position - previous_position) <= 0.01
        )
        if moved and position_stable and abs(status.speed(axis)) <= 0.01:
            stable_reads += 1
        else:
            stable_reads = 0
        home_complete = status.home_status == HOME_DONE
        completion_is_fresh = saw_activity or stable_reads >= 3
        if home_complete and completion_is_fresh:
            return status
        previous_position = position
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {axis.name} homing to finish.")


def wait_for_move(
    client: FMC4030Client,
    axis: Axis,
    initial_position_mm: float,
    timeout: float,
    poll_interval: float = 0.1,
    speed_tolerance: float = 0.01,
    position_tolerance: float = 0.01,
) -> CNCStatus:
    deadline = time.monotonic() + timeout
    stable_reads = 0
    previous_position: float | None = None
    while time.monotonic() < deadline:
        try:
            status = client.read_status(attempts=1)
        except TimeoutError as exc:
            with suppress(OSError):
                client.stop(axis, StopMode.IMMEDIATE)
            raise TimeoutError(
                f"Lost status while {axis.name} was moving; immediate stop was requested."
            ) from exc
        position = status.position(axis)
        position_stable = (
            previous_position is not None
            and abs(position - previous_position) <= position_tolerance
        )
        moved = abs(position - initial_position_mm) > position_tolerance
        if abs(status.speed(axis)) <= speed_tolerance and position_stable and moved:
            stable_reads += 1
            if stable_reads >= 3:
                return status
        else:
            stable_reads = 0
        previous_position = position
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {axis.name} motion to finish.")


def expected_motion_seconds(
    distance_mm: float,
    speed_mm_s: float,
    acceleration_mm_s2: float,
    deceleration_mm_s2: float,
) -> float:
    """Return the worst-case duration of one bounded trapezoidal move."""
    _validate_motion_values(
        distance_mm,
        speed_mm_s,
        acceleration_mm_s2,
        deceleration_mm_s2,
    )
    if distance_mm <= 0 or speed_mm_s <= 0:
        raise ValueError("Distance and speed must be greater than zero.")
    if acceleration_mm_s2 <= 0 or deceleration_mm_s2 <= 0:
        raise ValueError("Acceleration and deceleration must be greater than zero.")

    acceleration_distance = speed_mm_s**2 / (2.0 * acceleration_mm_s2)
    deceleration_distance = speed_mm_s**2 / (2.0 * deceleration_mm_s2)
    if distance_mm >= acceleration_distance + deceleration_distance:
        return (
            distance_mm / speed_mm_s
            + speed_mm_s / (2.0 * acceleration_mm_s2)
            + speed_mm_s / (2.0 * deceleration_mm_s2)
        )

    peak_speed = math.sqrt(
        2.0
        * distance_mm
        / (1.0 / acceleration_mm_s2 + 1.0 / deceleration_mm_s2)
    )
    return peak_speed / acceleration_mm_s2 + peak_speed / deceleration_mm_s2


def _timed_relative_move(
    client: FMC4030Client,
    axis: Axis,
    distance_mm: float,
    speed_mm_s: float,
    acceleration_mm_s2: float,
    deceleration_mm_s2: float,
    settle_seconds: float,
    phase: str,
    progress: Callable[[str, float], None] | None,
) -> None:
    wait_seconds = expected_motion_seconds(
        abs(distance_mm),
        speed_mm_s,
        acceleration_mm_s2,
        deceleration_mm_s2,
    ) + settle_seconds
    if progress is not None:
        progress(phase, wait_seconds)
    client.move(
        axis,
        distance_mm,
        speed_mm_s,
        acceleration_mm_s2,
        deceleration_mm_s2,
        Mode.RELATIVE,
    )
    # Match the proven manual workflow: finish the move transaction, release
    # that TCP session, wait for the bounded command to end, then open a fresh
    # session for the slow status request.
    client.close()
    time.sleep(wait_seconds)


def _read_stopped_limit(
    client: FMC4030Client,
    axis: Axis,
    limit: str,
    status_read_timeout: float,
    speed_tolerance: float,
    progress: Callable[[str, float], None] | None,
) -> CNCStatus:
    if progress is not None:
        progress(f"read_{limit}", 0.0)
    try:
        status = read_status_until_deadline(client, status_read_timeout)
    finally:
        client.close()

    if abs(status.speed(axis)) > speed_tolerance:
        raise RuntimeError(
            f"{axis.name} was still moving at {status.speed(axis):.3f} mm/s "
            f"after the timed {limit} move."
        )
    limit_active = status.negative_limit(axis) if limit == "min" else status.positive_limit(axis)
    if not limit_active:
        raise RuntimeError(
            f"{axis.name} stopped at {status.position(axis):.3f} mm, but its "
            f"{limit} limit input was not triggered. Increase --max-travel only "
            "after checking the physical workspace."
        )
    return status


def calibrate_axis_limits_timed(
    client: FMC4030Client,
    axis: Axis,
    max_travel_mm: float,
    speed_mm_s: float,
    acceleration_mm_s2: float,
    deceleration_mm_s2: float,
    backoff_mm: float,
    settle_seconds: float = DEFAULT_MOTION_SETTLE_SECONDS,
    status_read_timeout: float = DEFAULT_CALIBRATION_STATUS_READ_TIMEOUT_SECONDS,
    speed_tolerance: float = 0.01,
    progress: Callable[[str, float], None] | None = None,
) -> AxisLimitCalibration:
    """Capture both limits using bounded timed moves and two status reads."""
    _validate_motion_values(
        max_travel_mm,
        backoff_mm,
        settle_seconds,
        status_read_timeout,
        speed_tolerance,
    )
    if max_travel_mm <= 0 or backoff_mm <= 0:
        raise ValueError("Maximum travel and backoff must be greater than zero.")
    if settle_seconds < 0 or speed_tolerance < 0:
        raise ValueError("Settle time and speed tolerance cannot be negative.")
    if status_read_timeout <= 0:
        raise ValueError("Status-read timeout must be greater than zero.")
    if backoff_mm * 2.0 >= max_travel_mm:
        raise ValueError("Backoff must be less than half of maximum travel.")

    motion_may_be_active = False
    try:
        motion_may_be_active = True
        _timed_relative_move(
            client,
            axis,
            -max_travel_mm,
            speed_mm_s,
            acceleration_mm_s2,
            deceleration_mm_s2,
            settle_seconds,
            "move_min",
            progress,
        )
        minimum_status = _read_stopped_limit(
            client,
            axis,
            "min",
            status_read_timeout,
            speed_tolerance,
            progress,
        )
        motion_may_be_active = False
        minimum_switch = minimum_status.position(axis)

        motion_may_be_active = True
        _timed_relative_move(
            client,
            axis,
            backoff_mm,
            speed_mm_s,
            acceleration_mm_s2,
            deceleration_mm_s2,
            settle_seconds,
            "backoff_min",
            progress,
        )
        motion_may_be_active = False

        motion_may_be_active = True
        _timed_relative_move(
            client,
            axis,
            max_travel_mm,
            speed_mm_s,
            acceleration_mm_s2,
            deceleration_mm_s2,
            settle_seconds,
            "move_max",
            progress,
        )
        maximum_status = _read_stopped_limit(
            client,
            axis,
            "max",
            status_read_timeout,
            speed_tolerance,
            progress,
        )
        motion_may_be_active = False
        maximum_switch = maximum_status.position(axis)

        minimum_safe = minimum_switch + backoff_mm
        maximum_safe = maximum_switch - backoff_mm
        if minimum_safe >= maximum_safe:
            raise RuntimeError(
                f"{axis.name} measured an invalid interval: "
                f"min={minimum_switch:.3f}, max={maximum_switch:.3f} mm."
            )

        motion_may_be_active = True
        _timed_relative_move(
            client,
            axis,
            -backoff_mm,
            speed_mm_s,
            acceleration_mm_s2,
            deceleration_mm_s2,
            settle_seconds,
            "backoff_max",
            progress,
        )
        motion_may_be_active = False
    finally:
        if motion_may_be_active:
            client.close()
            with suppress(OSError):
                client.stop(axis, StopMode.IMMEDIATE)
            client.close()

    return AxisLimitCalibration(
        axis=axis,
        minimum=LimitSeekResult(
            axis=axis,
            limit="min",
            switch_position_mm=minimum_switch,
            safe_position_mm=minimum_safe,
        ),
        maximum=LimitSeekResult(
            axis=axis,
            limit="max",
            switch_position_mm=maximum_switch,
            safe_position_mm=maximum_safe,
        ),
    )


def move_to_absolute_targets_timed(
    client: FMC4030Client,
    targets_mm: dict[Axis, float],
    order: list[Axis],
    speed_mm_s: float,
    acceleration_mm_s2: float,
    deceleration_mm_s2: float,
    settle_seconds: float = DEFAULT_MOTION_SETTLE_SECONDS,
    status_read_timeout: float = DEFAULT_CALIBRATION_STATUS_READ_TIMEOUT_SECONDS,
    read_initial_status: bool = True,
    read_final_status: bool = True,
    travel_distance_bounds_mm: dict[Axis, float] | None = None,
    position_tolerance_mm: float = 1.0,
    speed_tolerance_mm_s: float = 0.01,
    progress: Callable[[str, Axis | None, float], None] | None = None,
) -> CNCStatus | None:
    """Command absolute X/Y/Z targets in parallel with optional status reads."""
    if set(targets_mm) != set(Axis):
        raise ValueError("Absolute targets must contain exactly X, Y, and Z.")
    if len(order) != len(Axis) or set(order) != set(Axis):
        raise ValueError("Movement order must contain X, Y, and Z exactly once.")
    if not read_initial_status:
        if travel_distance_bounds_mm is None or set(travel_distance_bounds_mm) != set(Axis):
            raise ValueError(
                "Skipping initial status requires X/Y/Z travel-distance bounds."
            )
        _validate_motion_values(*travel_distance_bounds_mm.values())
        if any(distance <= 0 for distance in travel_distance_bounds_mm.values()):
            raise ValueError("Travel-distance bounds must be greater than zero.")
    _validate_motion_values(
        *targets_mm.values(),
        speed_mm_s,
        acceleration_mm_s2,
        deceleration_mm_s2,
        settle_seconds,
        status_read_timeout,
        position_tolerance_mm,
        speed_tolerance_mm_s,
    )
    if speed_mm_s <= 0 or acceleration_mm_s2 <= 0 or deceleration_mm_s2 <= 0:
        raise ValueError("Speed, acceleration, and deceleration must be greater than zero.")
    if settle_seconds < 0 or speed_tolerance_mm_s < 0:
        raise ValueError("Settle time and speed tolerance cannot be negative.")
    if status_read_timeout <= 0 or position_tolerance_mm <= 0:
        raise ValueError("Status timeout and position tolerance must be greater than zero.")

    initial: CNCStatus | None = None
    if read_initial_status:
        if progress is not None:
            progress("read_initial", None, status_read_timeout)
        initial = read_status_until_deadline(client, status_read_timeout)
        client.close()
        moving_axes = [
            axis for axis in Axis if abs(initial.speed(axis)) > speed_tolerance_mm_s
        ]
        if moving_axes:
            names = ", ".join(axis.name for axis in moving_axes)
            raise RuntimeError(f"Cannot start because these axes are moving: {names}.")

    movement_started = False
    try:
        moving_axes: list[Axis] = []
        wait_seconds = 0.0
        for axis in order:
            if initial is not None:
                distance = abs(targets_mm[axis] - initial.position(axis))
            else:
                if travel_distance_bounds_mm is None:  # pragma: no cover - validated above
                    raise RuntimeError("Travel-distance bounds are unavailable.")
                distance = travel_distance_bounds_mm[axis]
            if initial is not None and distance <= position_tolerance_mm:
                if progress is not None:
                    progress("skip", axis, 0.0)
                continue
            moving_axes.append(axis)
            wait_seconds = max(
                wait_seconds,
                expected_motion_seconds(
                    distance,
                    speed_mm_s,
                    acceleration_mm_s2,
                    deceleration_mm_s2,
                )
                + settle_seconds,
            )
            if progress is not None:
                progress("command", axis, 0.0)
            movement_started = True
            client.move(
                axis,
                targets_mm[axis],
                speed_mm_s,
                acceleration_mm_s2,
                deceleration_mm_s2,
                Mode.ABSOLUTE,
            )

        if moving_axes:
            client.close()

        if not read_final_status:
            movement_started = False
            return None

        if moving_axes:
            if progress is not None:
                progress("wait_parallel", None, wait_seconds)
            time.sleep(wait_seconds)

        if progress is not None:
            progress("read_final", None, status_read_timeout)
        final = read_status_until_deadline(client, status_read_timeout)
        client.close()

        moving_axes = [
            axis for axis in Axis if abs(final.speed(axis)) > speed_tolerance_mm_s
        ]
        if moving_axes:
            names = ", ".join(axis.name for axis in moving_axes)
            raise RuntimeError(f"Final status shows these axes still moving: {names}.")
        errors = {
            axis: final.position(axis) - targets_mm[axis]
            for axis in Axis
            if abs(final.position(axis) - targets_mm[axis]) > position_tolerance_mm
        }
        if errors:
            details = ", ".join(
                f"{axis.name} error={error:+.3f} mm" for axis, error in errors.items()
            )
            raise RuntimeError(f"CNC did not reach the requested position: {details}.")
        movement_started = False
        return final
    finally:
        if movement_started:
            client.close()
            for axis in order:
                with suppress(OSError):
                    client.stop(axis, StopMode.IMMEDIATE)
                client.close()


def save_limit_calibration(
    path: Path,
    ip: str,
    port: int,
    result: LimitSeekResult,
) -> dict[str, object]:
    return save_limit_calibrations(path, ip, port, [result])


def save_limit_calibrations(
    path: Path,
    ip: str,
    port: int,
    results: list[LimitSeekResult],
) -> dict[str, object]:
    """Atomically update endpoints while preserving all other captures."""
    if not results:
        raise ValueError("At least one limit result is required.")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        controller = payload.get("controller", {})
        if (controller.get("ip"), controller.get("port")) != (ip, port):
            raise ValueError(
                f"Calibration belongs to {controller.get('ip')}:{controller.get('port')}, "
                f"not {ip}:{port}."
            )
        if payload.get("schema_version") != 1 or payload.get("units") != "mm":
            raise ValueError("Unsupported CNC limit-calibration file.")
    else:
        payload = {
            "schema_version": 1,
            "controller": {"ip": ip, "port": port},
            "units": "mm",
            "axes": {},
        }

    axes = payload.setdefault("axes", {})
    if not isinstance(axes, dict):
        raise ValueError("Calibration axes must be an object.")
    captured_at = datetime.now(UTC).isoformat()
    for result in results:
        axis_name = result.axis.name.lower()
        axis_payload = axes.setdefault(axis_name, {})
        if not isinstance(axis_payload, dict):
            raise ValueError(f"Calibration for {axis_name} must be an object.")
        axis_payload[result.limit] = {
            "position_mm": result.safe_position_mm,
            "switch_position_mm": result.switch_position_mm,
            "captured_at": captured_at,
        }

    for axis_name, axis_payload in axes.items():
        if not isinstance(axis_payload, dict):
            raise ValueError(f"Calibration for {axis_name} must be an object.")
        minimum = axis_payload.get("min")
        maximum = axis_payload.get("max")
        if isinstance(minimum, dict) and isinstance(maximum, dict):
            minimum_position = float(minimum["position_mm"])
            maximum_position = float(maximum["position_mm"])
            if minimum_position >= maximum_position:
                raise ValueError(
                    f"{axis_name.upper()} calibrated min {minimum_position:g} is not below "
                    f"max {maximum_position:g}; nothing was saved."
                )
            axis_payload["mid_position_mm"] = (minimum_position + maximum_position) / 2.0
        else:
            axis_payload.pop("mid_position_mm", None)

    payload["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def home_axes(
    client: FMC4030Client,
    axes: list[Axis],
    direction: HomeDirection,
    speed_mm_s: float,
    acceleration_mm_s2: float,
    backoff_mm: float,
    motion_timeout: float,
) -> dict[Axis, CNCStatus]:
    results: dict[Axis, CNCStatus] = {}
    for axis in axes:
        initial_position = read_status_with_retries(client).position(axis)
        client.home(
            axis,
            direction,
            speed_mm_s,
            acceleration_mm_s2,
            backoff_mm,
        )
        results[axis] = wait_for_home(
            client,
            axis,
            motion_timeout,
            initial_position_mm=initial_position,
        )
    return results


def new_position_config(
    ip: str,
    port: int,
    axes: dict[str, AxisPositions],
) -> CNCPositionConfig:
    config = CNCPositionConfig(
        ip=ip,
        port=port,
        axes=axes,
        created_at=datetime.now(UTC).isoformat(),
    )
    config.validate()
    return config


def save_position_config(path: Path, config: CNCPositionConfig) -> None:
    config.validate()
    payload = {
        "schema_version": config.schema_version,
        "controller": {"ip": config.ip, "port": config.port},
        "units": "mm",
        "created_at": config.created_at,
        "axes": {
            name: {
                "min": positions.min,
                "mid": positions.mid,
                "max": positions.max,
                "positive_limit_mm": positions.positive_limit_mm,
            }
            for name, positions in config.axes.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_position_config(path: Path) -> CNCPositionConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("units") != "mm":
        raise ValueError("Only millimetre CNC position configs are supported.")
    controller = payload["controller"]
    config = CNCPositionConfig(
        ip=str(controller["ip"]),
        port=int(controller["port"]),
        axes={
            name: AxisPositions(
                min=float(values["min"]),
                mid=float(values["mid"]),
                max=float(values["max"]),
                positive_limit_mm=float(values["positive_limit_mm"]),
            )
            for name, values in payload["axes"].items()
        },
        created_at=str(payload["created_at"]),
        schema_version=int(payload["schema_version"]),
    )
    config.validate()
    return config
