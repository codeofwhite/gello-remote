"""Thread-safe safety state for direct simulated Franka joint control."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import threading
import time

from .backends import FRANKA_ARM_DOF, validate_joint_targets
from .protocol import GelloCommand


FRANKA_HOME = (0.0, -0.65, 0.0, -2.25, 0.0, 1.75, 0.78)


class GelloSessionGate:
    """Reject stale sequences and competing live browser control sessions."""

    def __init__(
        self,
        watchdog_timeout: float = 0.3,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(watchdog_timeout) or watchdog_timeout <= 0:
            raise ValueError("watchdog_timeout must be positive and finite")
        self._watchdog_timeout = float(watchdog_timeout)
        self._clock = clock
        self._active_session_id: str | None = None
        self._last_sequence: int | None = None
        self._last_received_at: float | None = None

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    def accept(self, command: GelloCommand, *, received_at: float | None = None) -> bool:
        now = self._clock() if received_at is None else float(received_at)
        if not math.isfinite(now):
            raise ValueError("received_at must be finite")
        active_is_fresh = (
            self._active_session_id is not None
            and self._last_received_at is not None
            and now - self._last_received_at <= self._watchdog_timeout
        )
        if command.session_id != self._active_session_id:
            if active_is_fresh:
                return False
            self._active_session_id = command.session_id
            self._last_sequence = None
        if self._last_sequence is not None and command.sequence <= self._last_sequence:
            return False
        self._last_sequence = command.sequence
        self._last_received_at = now
        return True

    def release(self, session_id: str) -> None:
        if session_id == self._active_session_id:
            self._active_session_id = None
            self._last_sequence = None
            self._last_received_at = None


@dataclass(frozen=True)
class SafeFrankaCommand:
    """One safety-filtered simulated Franka command."""

    joint_targets: tuple[float, ...]
    gripper_closed: bool
    sequence: int | None
    source_timestamp_ns: int | None
    watchdog_holding: bool


class GelloCommandController:
    """Accept GELLO messages and emit bounded direct-joint commands."""

    def __init__(
        self,
        initial_joint_targets: Sequence[float],
        joint_limits: Sequence[Sequence[float]],
        *,
        control_hz: float = 20.0,
        max_joint_speed: float = 1.0,
        watchdog_timeout: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if control_hz <= 0 or max_joint_speed <= 0 or watchdog_timeout <= 0:
            raise ValueError("control_hz, max_joint_speed, and watchdog_timeout must be positive")
        if len(joint_limits) != FRANKA_ARM_DOF:
            raise ValueError("joint_limits must contain exactly 7 lower/upper pairs")
        parsed_limits: list[tuple[float, float]] = []
        for pair in joint_limits:
            if len(pair) != 2:
                raise ValueError("each joint limit must contain lower and upper values")
            lower, upper = float(pair[0]), float(pair[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("joint limits must be finite and strictly increasing")
            parsed_limits.append((lower, upper))

        initial = validate_joint_targets(initial_joint_targets)
        self._target = self._clip(initial, parsed_limits)
        self._joint_limits = tuple(parsed_limits)
        self._max_step = max_joint_speed / control_hz
        self._watchdog_timeout = watchdog_timeout
        self._clock = clock
        self._session_gate = GelloSessionGate(watchdog_timeout, clock=clock)
        self._lock = threading.Lock()
        self._latest: GelloCommand | None = None
        self._latest_received_at: float | None = None
        self._last_sequence: int | None = None
        self._gripper_closed = False
        self._watchdog_holding = False
        self._accepted_commands = 0
        self._rejected_stale_commands = 0
        self._watchdog_holds = 0

    @staticmethod
    def _clip(
        targets: Sequence[float], limits: Sequence[tuple[float, float]]
    ) -> tuple[float, ...]:
        return tuple(min(max(float(value), lower), upper) for value, (lower, upper) in zip(targets, limits))

    def submit(self, command: GelloCommand, *, received_at: float | None = None) -> bool:
        """Store a newer validated GELLO command, rejecting stale sequences."""
        now = self._clock() if received_at is None else float(received_at)
        if not math.isfinite(now):
            raise ValueError("received_at must be finite")
        with self._lock:
            if not self._session_gate.accept(command, received_at=now):
                self._rejected_stale_commands += 1
                return False
            self._latest = command
            self._latest_received_at = now
            self._last_sequence = command.sequence
            self._accepted_commands += 1
            self._watchdog_holding = False
            return True

    def release_session(self, session_id: str) -> None:
        """Release a disconnected browser session without changing the held target."""
        with self._lock:
            self._session_gate.release(session_id)
            if self._latest is not None and self._latest.session_id == session_id:
                self._latest = None
                self._latest_received_at = None

    def reset_targets(self, joint_targets: Sequence[float]) -> None:
        """Return the held command to a validated pose without weakening sequence checks."""
        target = self._clip(validate_joint_targets(joint_targets), self._joint_limits)
        with self._lock:
            self._target = target
            self._latest = None
            self._latest_received_at = None
            self._gripper_closed = False
            self._watchdog_holding = False

    def tick(self, *, now: float | None = None) -> SafeFrankaCommand:
        """Advance one control cycle and return the safety-filtered command."""
        current_time = self._clock() if now is None else float(now)
        if not math.isfinite(current_time):
            raise ValueError("now must be finite")
        with self._lock:
            fresh = (
                self._latest is not None
                and self._latest_received_at is not None
                and current_time - self._latest_received_at <= self._watchdog_timeout
            )
            if fresh:
                assert self._latest is not None
                desired = self._clip(validate_joint_targets(self._latest.joint_targets), self._joint_limits)
                self._target = tuple(
                    current + min(max(wanted - current, -self._max_step), self._max_step)
                    for current, wanted in zip(self._target, desired)
                )
                self._gripper_closed = self._latest.gripper_closed
                self._watchdog_holding = False
            elif self._latest_received_at is not None and not self._watchdog_holding:
                self._watchdog_holding = True
                self._watchdog_holds += 1

            return SafeFrankaCommand(
                joint_targets=self._target,
                gripper_closed=self._gripper_closed,
                sequence=self._last_sequence,
                source_timestamp_ns=(self._latest.timestamp_ns if self._latest is not None else None),
                watchdog_holding=self._watchdog_holding,
            )

    def status(self, *, now: float | None = None) -> dict[str, object]:
        """Return transport and watchdog counters for the control UI."""
        current_time = self._clock() if now is None else float(now)
        with self._lock:
            age_ms = (
                max(0.0, current_time - self._latest_received_at) * 1000.0
                if self._latest_received_at is not None
                else None
            )
            source_age_ms = (
                max(0.0, time.time_ns() - self._latest.timestamp_ns) / 1e6 if self._latest is not None else None
            )
            return {
                "sequence": self._last_sequence,
                "session_id": self._session_gate.active_session_id,
                "age_ms": round(age_ms, 1) if age_ms is not None else None,
                "source_age_ms": round(source_age_ms, 1) if source_age_ms is not None else None,
                "watchdog_holding": self._watchdog_holding,
                "watchdog_holds": self._watchdog_holds,
                "accepted_commands": self._accepted_commands,
                "rejected_stale_commands": self._rejected_stale_commands,
                "joint_targets": list(self._target),
                "gripper_closed": self._gripper_closed,
            }
