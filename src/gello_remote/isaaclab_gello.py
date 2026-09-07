"""Direct seven-joint GELLO teleoperation for one simulated Isaac Lab Franka."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from http import HTTPStatus
import json
import math
import mimetypes
import os
from pathlib import Path
import struct
import sys
import threading
import time
import traceback
from typing import Any

from robot_data_contracts import CameraProfileV1, load_camera_profile

from .backends import FRANKA_GRIPPER_CLOSED_M, IsaacLabFrankaBackend
from .control import FRANKA_HOME, GelloCommandController, GelloSessionGate
from .protocol import GelloCommand


WEB_ROOT = Path(__file__).with_name("web")
WEB_CONSOLE_ROOT = WEB_ROOT / "console"
STREAM_CAMERA_EYE = (2.2, 1.6, 1.7)
STREAM_CAMERA_TARGET = (0.45, 0.0, 0.65)
STREAM_WIDTH = 960
STREAM_HEIGHT = 720
STREAM_SCENE_PRIMS = ("/World/Franka", "/World/defaultGroundPlane", "/World/Light")
ISAAC_SCENE_IDS = ("franka_empty", "rigid_pick_place", "rigid_cube_lift")
DEFAULT_WORK_SURFACE_HEIGHT_M = 0.25
MIN_WORK_SURFACE_HEIGHT_M = 0.18
MAX_WORK_SURFACE_HEIGHT_M = 0.32
CUBE_LIFT_THRESHOLD_M = 0.08
CUBE_GRASP_THRESHOLD_M = 0.02
CUBE_HOLD_SECONDS = 1.0
DEFAULT_CAMERA_PROFILE: dict[str, Any] = {
    "format": "robot_data_camera_profile",
    "format_version": 1,
    "id": "builtin_d435i_nominal_v1",
    "label": "Built-in nominal Intel RealSense D435i geometry",
    "match_level": "nominal_pending_intrinsics",
    "roles": {
        name: {
            "enabled": True,
            "model": "Intel RealSense D435i",
            "calibration_source_serial": None,
            "nominal_vertical_fov_deg": 42.0,
            "stream": {"width": 640, "height": 480, "source_fps": 30, "format": "rgb8"},
            "intrinsics": None,
            "distortion": None,
        }
        for name in ("agentview", "wrist")
    },
    "engine_overrides": {
        "isaac_sim": {
            "base": {"eye_world_m": list(STREAM_CAMERA_EYE), "target_world_m": list(STREAM_CAMERA_TARGET)},
            "top": {
                "eye_world_m": [0.45, 0.0, 2.4],
                "target_world_m": [0.45, 0.0, 0.45],
                "nominal_vertical_fov_deg": 52.0,
                "virtual_auxiliary": True,
            },
            "side": {
                "eye_world_m": [0.45, 1.8, 1.0],
                "target_world_m": [0.45, 0.0, 0.55],
                "nominal_vertical_fov_deg": 55.0,
                "virtual_auxiliary": True,
            },
            "wrist": {
                "parent_prim": "/World/Franka/panda_hand",
                "prim_path": "/World/Franka/panda_hand/wrist_cam",
                "position_parent_m": [0.13, 0.0, -0.19],
                "rotation_parent_xyzw": [0.03701, 0.03701, -0.70614, -0.70614],
                "convention": "ros",
                "reference": "IsaacLab stack_ik_rel_visuomotor_env_cfg.py",
                "reference_revision": "28a37cecdd433c22d9eabd6a5954add9f13a8951",
                "tuning_note": "Lowered 0.04 m along panda_hand z to keep the gripper visible",
            },
        }
    },
}
GPU_KIT_SETTING_KEYS = (
    "--/renderer/activeGpu",
    "--/physics/cudaDevice",
    "--/renderer/multiGpu/enabled",
    "--/renderer/multiGpu/autoEnable",
    "--/renderer/multiGpu/activeGpus",
    "--/renderer/multiGpu/maxGpuCount",
)
CAMERA_NAMES = ("base", "top", "side", "wrist")
VIEW_NAMES = ("free", *CAMERA_NAMES)
PREVIEW_CAMERA_IDS = {name: index for index, name in enumerate(CAMERA_NAMES)}
PREVIEW_MAGIC = b"RDSC"
PREVIEW_PROTOCOL_VERSION = 1
PREVIEW_HEADER = struct.Struct("!4sBBQQ")
DEFAULT_PREVIEW_FPS = 10.0
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240
PREVIEW_JPEG_QUALITY = 70
PREVIEW_MAX_JPEG_BYTES = 256 * 1024
CONTROL_SOURCES = ("keyboard", "gello", "openpi", "trajectory")
KEYBOARD_KEYS = (
    "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE",
    "KeyZ", "KeyX", "KeyT", "KeyG", "KeyC", "KeyV",
)
KEYBOARD_AXIS_COMMANDS = {
    "KeyW": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "KeyS": (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "KeyA": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "KeyD": (0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    "KeyQ": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "KeyE": (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
    "KeyZ": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "KeyX": (0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
    "KeyT": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "KeyG": (0.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "KeyC": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "KeyV": (0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
}


def _euler_xyz_to_rotation_vector(euler_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """Match ``scipy Rotation.from_euler('XYZ', ...).as_rotvec()`` without a new dependency."""
    roll, pitch, yaw = (value / 2.0 for value in euler_xyz)
    sin_x, cos_x = math.sin(roll), math.cos(roll)
    sin_y, cos_y = math.sin(pitch), math.cos(pitch)
    sin_z, cos_z = math.sin(yaw), math.cos(yaw)
    quaternion_xyz = (
        sin_x * cos_y * cos_z + cos_x * sin_y * sin_z,
        cos_x * sin_y * cos_z - sin_x * cos_y * sin_z,
        cos_x * cos_y * sin_z + sin_x * sin_y * cos_z,
    )
    quaternion_w = cos_x * cos_y * cos_z - sin_x * sin_y * sin_z
    vector_norm = math.sqrt(sum(value * value for value in quaternion_xyz))
    if vector_norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, quaternion_w)
    return tuple(value * angle / vector_norm for value in quaternion_xyz)


@dataclass(frozen=True)
class BrowserKeyboardCommand:
    """Validated browser state matching Isaac Lab's ``Se3Keyboard`` contract."""

    session_id: str
    sequence: int
    timestamp_ns: int
    pressed: tuple[str, ...]
    gripper_closed: bool


@dataclass(frozen=True)
class BrowserKeyboardSample:
    """One task-space keyboard sample with source provenance."""

    delta_pose: tuple[float, ...]
    gripper_closed: bool
    session_id: str | None
    sequence: int | None
    source_timestamp_ns: int | None
    receive_timestamp_ns: int | None
    watchdog_holding: bool


class BrowserSe3KeyboardState:
    """Thread-safe browser transport for the official Isaac Lab SE(3) key map."""

    def __init__(
        self,
        *,
        position_step_m: float,
        rotation_step_rad: float,
        watchdog_timeout: float,
        clock=time.monotonic,
    ) -> None:
        parameters = (position_step_m, rotation_step_rad, watchdog_timeout)
        if not all(math.isfinite(value) and value > 0 for value in parameters):
            raise ValueError("keyboard steps and watchdog timeout must be positive")
        self._position_step_m = float(position_step_m)
        self._rotation_step_rad = float(rotation_step_rad)
        self._watchdog_timeout = float(watchdog_timeout)
        self._clock = clock
        self._gate = GelloSessionGate(watchdog_timeout, clock=clock)
        self._lock = threading.Lock()
        self._latest: BrowserKeyboardCommand | None = None
        self._received_at: float | None = None
        self._watchdog_holding = False
        self._watchdog_holds = 0
        self._accepted_commands = 0
        self._rejected_commands = 0

    @staticmethod
    def _validate(data: dict[str, Any]) -> BrowserKeyboardCommand:
        if data.get("type") != "keyboard" or data.get("version") != 1:
            raise ValueError("keyboard command requires type='keyboard' and version=1")
        session_id = data.get("session_id")
        sequence = data.get("sequence")
        timestamp_ns = data.get("timestamp_ns")
        pressed = data.get("pressed")
        gripper_closed = data.get("gripper_closed")
        if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
            raise ValueError("keyboard session_id must be a non-empty string")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("keyboard sequence must be a non-negative integer")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
            raise ValueError("keyboard timestamp_ns must be a positive integer")
        if not isinstance(pressed, list) or any(not isinstance(key, str) for key in pressed):
            raise ValueError("keyboard pressed must be a list of key codes")
        unknown = sorted(set(pressed) - set(KEYBOARD_KEYS))
        if unknown:
            raise ValueError(f"unsupported keyboard key: {', '.join(unknown)}")
        if len(pressed) != len(set(pressed)):
            raise ValueError("keyboard pressed keys must be unique")
        if not isinstance(gripper_closed, bool):
            raise ValueError("keyboard gripper_closed must be boolean")
        return BrowserKeyboardCommand(
            session_id=session_id,
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            pressed=tuple(pressed),
            gripper_closed=gripper_closed,
        )

    def submit(self, data: dict[str, Any], *, received_at: float | None = None) -> bool:
        command = self._validate(data)
        now = self._clock() if received_at is None else float(received_at)
        if not math.isfinite(now):
            raise ValueError("received_at must be finite")
        with self._lock:
            if not self._gate.accept(command, received_at=now):
                self._rejected_commands += 1
                return False
            self._latest = command
            self._received_at = now
            self._watchdog_holding = False
            self._accepted_commands += 1
            return True

    def release_session(self, session_id: str) -> None:
        with self._lock:
            self._gate.release(session_id)
            if self._latest is not None and self._latest.session_id == session_id:
                self._latest = None
                self._received_at = None

    def reset(self) -> None:
        with self._lock:
            self._latest = None
            self._received_at = None
            self._watchdog_holding = False

    def advance(self, *, now: float | None = None) -> BrowserKeyboardSample:
        current_time = self._clock() if now is None else float(now)
        with self._lock:
            fresh = (
                self._latest is not None
                and self._received_at is not None
                and current_time - self._received_at <= self._watchdog_timeout
            )
            if not fresh and self._received_at is not None and not self._watchdog_holding:
                self._watchdog_holding = True
                self._watchdog_holds += 1
            command = self._latest
            axes = [0.0] * 6
            if fresh and command is not None:
                for key in command.pressed:
                    for index, value in enumerate(KEYBOARD_AXIS_COMMANDS[key]):
                        axes[index] += value
                axes[:3] = [value * self._position_step_m for value in axes[:3]]
                euler_xyz = tuple(value * self._rotation_step_rad for value in axes[3:])
                axes[3:] = _euler_xyz_to_rotation_vector(euler_xyz)
            return BrowserKeyboardSample(
                delta_pose=tuple(axes),
                gripper_closed=command.gripper_closed if command is not None else False,
                session_id=command.session_id if command is not None else None,
                sequence=command.sequence if command is not None else None,
                source_timestamp_ns=command.timestamp_ns if command is not None else None,
                receive_timestamp_ns=(
                    int(self._received_at * 1e9) if self._received_at is not None else None
                ),
                watchdog_holding=self._watchdog_holding,
            )

    def status(self, *, now: float | None = None) -> dict[str, object]:
        sample = self.advance(now=now)
        current_time = self._clock() if now is None else float(now)
        with self._lock:
            age_ms = (
                max(0.0, current_time - self._received_at) * 1000.0
                if self._received_at is not None
                else None
            )
            return {
                "sequence": sample.sequence,
                "session_id": self._gate.active_session_id,
                "age_ms": round(age_ms, 1) if age_ms is not None else None,
                "watchdog_holding": sample.watchdog_holding,
                "watchdog_holds": self._watchdog_holds,
                "accepted_commands": self._accepted_commands,
                "rejected_commands": self._rejected_commands,
                "pressed": list(self._latest.pressed) if self._latest is not None else [],
                "gripper_closed": sample.gripper_closed,
                "delta_pose": list(sample.delta_pose),
            }


def _validate_work_surface_height(value: float) -> float:
    """Validate the task surface height relative to the Franka base."""
    height = float(value)
    if not math.isfinite(height) or not MIN_WORK_SURFACE_HEIGHT_M <= height <= MAX_WORK_SURFACE_HEIGHT_M:
        raise ValueError(
            "work-surface-height-m must be within "
            f"[{MIN_WORK_SURFACE_HEIGHT_M}, {MAX_WORK_SURFACE_HEIGHT_M}]"
        )
    return height


def _rigid_pick_place_geometry(work_surface_height_m: float) -> dict[str, object]:
    """Return the engine-independent P0 rigid-task geometry in Franka-base coordinates."""
    surface_z = _validate_work_surface_height(work_surface_height_m)
    object_size = 0.05
    return {
        "work_surface_center_m": (0.50, 0.00, surface_z / 2.0),
        "work_surface_size_m": (0.70, 0.60, surface_z),
        "work_surface_top_m": surface_z,
        "object_size_m": object_size,
        "object_start_m": (0.55, -0.10, surface_z + object_size / 2.0 + 0.002),
        "target_center_m": (0.48, 0.12, surface_z + 0.002),
        "target_size_m": (0.14, 0.14, 0.004),
        "task_instruction": "Pick up the green cube and place it on the blue target",
    }


def _rigid_cube_lift_geometry(work_surface_height_m: float) -> dict[str, object]:
    """Return the fixed P0 cube-lift scene without a placement target."""
    geometry = _rigid_pick_place_geometry(work_surface_height_m)
    geometry.pop("target_center_m")
    geometry.pop("target_size_m")
    geometry["task_instruction"] = (
        "Grasp the green cube, lift it at least 8 cm, hold for 1 second, then release"
    )
    geometry["evaluator"] = {
        "id": "isaac_cube_lift_v1",
        "lift_threshold_m": CUBE_LIFT_THRESHOLD_M,
        "hold_seconds": CUBE_HOLD_SECONDS,
    }
    return geometry


@dataclass
class CubeLiftEvaluator:
    """Dataset-rate evaluator for the fixed grasp-lift-hold-release pilot."""

    fps: int
    initial_object_z_m: float
    lift_threshold_m: float = CUBE_LIFT_THRESHOLD_M
    required_hold_seconds: float = CUBE_HOLD_SECONDS
    grasp_threshold_m: float = CUBE_GRASP_THRESHOLD_M
    success_mode: str = "lift_hold_release"
    close_seen: bool = False
    grasp_seen: bool = False
    release_after_hold_seen: bool = False
    max_lift_m: float = 0.0
    max_closed_lift_m: float = 0.0
    current_hold_frames: int = 0
    max_hold_frames: int = 0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("cube evaluator fps must be positive")
        if self.success_mode not in {"lift", "lift_hold_release"}:
            raise ValueError("cube evaluator success_mode must be lift or lift_hold_release")
        if (
            self.grasp_threshold_m <= 0
            or self.grasp_threshold_m >= self.lift_threshold_m
            or self.lift_threshold_m <= 0
            or self.required_hold_seconds <= 0
        ):
            raise ValueError("cube evaluator thresholds must be positive")

    @property
    def required_hold_frames(self) -> int:
        return int(math.ceil(self.required_hold_seconds * self.fps))

    def reset(self) -> None:
        self.close_seen = False
        self.grasp_seen = False
        self.release_after_hold_seen = False
        self.max_lift_m = 0.0
        self.max_closed_lift_m = 0.0
        self.current_hold_frames = 0
        self.max_hold_frames = 0

    def update(self, *, object_z_m: float, gripper_closed: bool) -> None:
        lift_m = max(0.0, float(object_z_m) - self.initial_object_z_m)
        self.max_lift_m = max(self.max_lift_m, lift_m)
        if gripper_closed:
            self.close_seen = True
            self.max_closed_lift_m = max(self.max_closed_lift_m, lift_m)
            self.grasp_seen = self.grasp_seen or lift_m >= self.grasp_threshold_m
            self.current_hold_frames = (
                self.current_hold_frames + 1
                if lift_m >= self.lift_threshold_m
                else 0
            )
            self.max_hold_frames = max(self.max_hold_frames, self.current_hold_frames)
        else:
            if self.close_seen and self.max_hold_frames >= self.required_hold_frames:
                self.release_after_hold_seen = True
            self.current_hold_frames = 0

    def result(self) -> dict[str, object]:
        lift_success = self.max_closed_lift_m >= self.lift_threshold_m
        success = self.grasp_seen and lift_success
        if self.success_mode == "lift_hold_release":
            success = (
                success
                and self.max_hold_frames >= self.required_hold_frames
                and self.release_after_hold_seen
            )
        return {
            "success": success,
            "success_mode": self.success_mode,
            "close_seen": self.close_seen,
            "grasp": self.grasp_seen,
            "lift": lift_success,
            "hold": self.max_hold_frames >= self.required_hold_frames,
            "release": self.release_after_hold_seen,
            "release_after_hold_seen": self.release_after_hold_seen,
            "grasp_threshold_m": self.grasp_threshold_m,
            "lift_threshold_m": self.lift_threshold_m,
            "max_lift_m": self.max_lift_m,
            "max_closed_lift_m": self.max_closed_lift_m,
            "required_hold_frames": self.required_hold_frames,
            "max_hold_frames": self.max_hold_frames,
            "hold_duration_s": self.max_hold_frames / self.fps,
        }


def _pose_tracking_error(
    actual_pose_wxyz: tuple[float, ...] | list[float] | Any,
    target_pose_wxyz: tuple[float, ...] | list[float] | Any,
) -> tuple[float, float]:
    """Return translation norm and shortest quaternion angle for two xyz+wxyz poses."""

    import numpy as np

    actual = np.asarray(actual_pose_wxyz, dtype=np.float64)
    target = np.asarray(target_pose_wxyz, dtype=np.float64)
    if actual.shape != (7,) or target.shape != (7,) or not np.isfinite((actual, target)).all():
        raise ValueError("pose tracking inputs must be finite xyz+wxyz vectors")
    actual_norm = float(np.linalg.norm(actual[3:]))
    target_norm = float(np.linalg.norm(target[3:]))
    if actual_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("pose tracking quaternions must have non-zero norm")
    actual_q = actual[3:] / actual_norm
    target_q = target[3:] / target_norm
    position_error = float(np.linalg.norm(actual[:3] - target[:3]))
    quaternion_dot = float(np.clip(abs(np.dot(actual_q, target_q)), 0.0, 1.0))
    rotation_error = 2.0 * math.acos(quaternion_dot)
    return position_error, rotation_error


@dataclass
class PoseConvergenceGate:
    position_tolerance_m: float
    rotation_tolerance_rad: float
    required_stable_ticks: int
    timeout_ticks: int
    elapsed_ticks: int = 0
    stable_ticks: int = 0

    def __post_init__(self) -> None:
        if (
            self.position_tolerance_m <= 0.0
            or self.rotation_tolerance_rad <= 0.0
            or self.required_stable_ticks <= 0
            or self.timeout_ticks < self.required_stable_ticks
        ):
            raise ValueError("invalid pose convergence gate parameters")

    def update(self, actual_pose: Any, target_pose: Any) -> tuple[str, float, float]:
        position_error, rotation_error = _pose_tracking_error(actual_pose, target_pose)
        self.elapsed_ticks += 1
        if (
            position_error <= self.position_tolerance_m
            and rotation_error <= self.rotation_tolerance_rad
        ):
            self.stable_ticks += 1
        else:
            self.stable_ticks = 0
        if self.stable_ticks >= self.required_stable_ticks:
            status = "converged"
        elif self.elapsed_ticks >= self.timeout_ticks:
            status = "timed_out"
        else:
            status = "waiting"
        return status, position_error, rotation_error


def _rollout_stage_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ordered cube-lift stage outcomes without hiding failed trials."""
    stages = ("grasp", "lift", "hold", "release")
    total = len(reports)
    return {
        "rollouts": total,
        "overall_successes": sum(bool(report["success"]) for report in reports),
        "overall_success_rate": (
            sum(bool(report["success"]) for report in reports) / total if total else None
        ),
        "stages": {
            stage: {
                "successes": sum(bool(report["metrics"][stage]) for report in reports),
                "rate": (
                    sum(bool(report["metrics"][stage]) for report in reports) / total
                    if total
                    else None
                ),
            }
            for stage in stages
        },
    }


def _evaluation_position_split(evaluation_plan: Any | None) -> str:
    """Name the position distribution used by a policy rollout batch."""
    return evaluation_plan.position_split if evaluation_plan is not None else "nominal"


def _scene_camera_overrides(
    profile: dict[str, Any], *, work_surface_top_m: float | None
) -> dict[str, dict[str, Any]]:
    """Resolve camera targets without mutating the reusable hardware profile."""
    overrides = json.loads(json.dumps(profile["engine_overrides"]["isaac_sim"]))
    if work_surface_top_m is None:
        return overrides
    overrides["base"]["target_world_m"] = [0.45, 0.0, work_surface_top_m + 0.15]
    overrides["top"]["target_world_m"] = [0.45, 0.0, work_surface_top_m]
    overrides["side"]["target_world_m"] = [0.45, 0.0, work_surface_top_m + 0.14]
    return overrides


def _load_camera_profile(path: Path | None, *, width: int, height: int) -> dict[str, Any]:
    """Load the import-safe camera profile before the Isaac application starts."""
    if path is None:
        payload = json.loads(json.dumps(DEFAULT_CAMERA_PROFILE))
        source = "builtin"
        for role in payload["roles"].values():
            role["stream"]["width"] = width
            role["stream"]["height"] = height
        profile = CameraProfileV1.model_validate(payload)
    else:
        resolved = path.expanduser().resolve()
        profile = load_camera_profile(resolved, width=width, height=height)
        source = str(resolved)
    if not {"agentview", "wrist"} <= profile.roles.keys():
        raise ValueError("Camera profile must define agentview and wrist roles")
    for role_name in ("agentview", "wrist"):
        role = profile.roles[role_name]
        if not role.enabled or (role.stream.width, role.stream.height) != (width, height):
            raise ValueError(
                f"Camera role {role_name!r} does not provide enabled {width}x{height} RGB"
            )
        if role.intrinsics is None and not role.nominal_vertical_fov_deg:
            raise ValueError(f"Camera role {role_name!r} requires a nominal vertical field of view")
    value = profile.model_dump(mode="json")
    overrides = value.get("engine_overrides", {}).get("isaac_sim")
    if not isinstance(overrides, dict) or not set(CAMERA_NAMES) <= overrides.keys():
        raise ValueError(
            "Camera profile must define Isaac Sim base, top, side, and wrist overrides"
        )
    value["profile_source"] = source
    return value


def _camera_intrinsic_matrix(
    role: dict[str, Any], *, width: int, height: int, nominal_vertical_fov_deg: float = 42.0
) -> tuple[list[float], str]:
    """Return row-major K and whether it came from calibration or the nominal FOV."""
    intrinsics = role.get("intrinsics")
    if intrinsics is not None:
        try:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Camera intrinsics must contain numeric fx, fy, cx, and cy") from exc
        values = (fx, fy, cx, cy)
        if min(fx, fy) <= 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("Camera intrinsics must be finite with positive focal lengths")
        return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], "calibrated_intrinsics"
    focal_px = (height / 2.0) / math.tan(math.radians(nominal_vertical_fov_deg) / 2.0)
    return [
        focal_px,
        0.0,
        width / 2.0,
        0.0,
        focal_px,
        height / 2.0,
        0.0,
        0.0,
        1.0,
    ], "nominal_vertical_fov"


def _camera_profile_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": profile["id"],
        "label": profile.get("label"),
        "match_level": profile.get("match_level"),
        "profile_source": profile["profile_source"],
        "roles": {
            name: {
                "model": profile["roles"][name].get("model"),
                "calibration_source_serial": profile["roles"][name].get(
                    "calibration_source_serial"
                ),
                "stream": profile["roles"][name].get("stream"),
                "nominal_vertical_fov_deg": profile["roles"][name].get(
                    "nominal_vertical_fov_deg"
                ),
                "intrinsics": profile["roles"][name].get("intrinsics"),
                "distortion": profile["roles"][name].get("distortion"),
            }
            for name in ("agentview", "wrist")
        },
        "virtual_auxiliary_views": {
            name: profile["engine_overrides"]["isaac_sim"][name]
            for name in ("top", "side")
        },
    }


def _frequency_decimation(source_hz: float, target_hz: float, *, label: str) -> int:
    """Return an exact control-step decimation for a requested camera frequency."""
    if target_hz <= 0 or target_hz > source_hz:
        raise ValueError(f"{label} must be positive and no greater than its source frequency")
    decimation = source_hz / target_hz
    if not math.isclose(decimation, round(decimation), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"source frequency must be an integer multiple of {label}")
    return int(round(decimation))


def _configure_gpu_selection(args: argparse.Namespace, argv: list[str]) -> None:
    """Expose one physical GPU while keeping CUDA and Vulkan ordinals aligned."""
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be non-negative")
    if any(token == "--device" or token.startswith("--device=") for token in argv):
        raise ValueError("use --gpu-id instead of --device for this entry point")

    kit_tokens = args.kit_args.split()
    conflicting_settings = sorted(
        {
            token.partition("=")[0]
            for token in kit_tokens
            if token.partition("=")[0] in GPU_KIT_SETTING_KEYS
        }
    )
    if conflicting_settings:
        raise ValueError(
            "GPU routing is managed by --gpu-id; remove these --kit_args settings: "
            f"{', '.join(conflicting_settings)}"
        )

    # CUDA renumbers a single CUDA_VISIBLE_DEVICES entry to logical device 0.
    # Keep the physical id for Vulkan/RTX, whose activeGpu setting follows the
    # host nvidia-smi numbering, while simulation and PhysX use logical cuda:0.
    args.renderer_gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = "cuda:0"
    gpu_kit_tokens = (
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        f"--/renderer/multiGpu/activeGpus={args.gpu_id}",
        "--/renderer/multiGpu/maxGpuCount=1",
    )
    args.kit_args = " ".join((*kit_tokens, *gpu_kit_tokens))


def _apply_split_gpu_launcher_config(
    launcher_args: dict[str, object], physical_gpu_id: int
) -> None:
    """Route RTX by host ordinal and PhysX by the masked CUDA ordinal."""
    launcher_args["active_gpu"] = physical_gpu_id
    launcher_args["physics_gpu"] = 0


def _missing_scene_prims(
    stage: object, required_prims: tuple[str, ...] = STREAM_SCENE_PRIMS
) -> list[str]:
    """Return required stream-scene prims that are missing or inactive."""
    missing: list[str] = []
    for prim_path in required_prims:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsActive():
            missing.append(prim_path)
    return missing


def _required_scene_prims(scene_geometry: dict[str, object] | None) -> tuple[str, ...]:
    """Return only the USD prims declared by the selected scene geometry."""
    required = STREAM_SCENE_PRIMS
    if scene_geometry is None:
        return required
    required += ("/World/WorkSurface", "/World/TaskObject")
    if "target_size_m" in scene_geometry:
        required += ("/World/TargetZone",)
    return required


def _configure_stream_view(
    *,
    camera_target: tuple[float, float, float] = STREAM_CAMERA_TARGET,
    required_prims: tuple[str, ...] = STREAM_SCENE_PRIMS,
) -> dict[str, object]:
    """Validate the scene and bind the live WebRTC viewport to the Franka view."""
    import omni.usd
    from isaacsim.core.rendering_manager import ViewportManager

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac stream scene has no active USD stage")
    missing_prims = _missing_scene_prims(stage, required_prims)
    if missing_prims:
        raise RuntimeError(f"Isaac stream scene is missing required prims: {missing_prims}")

    viewport = ViewportManager.get_viewport_api()
    if viewport is None:
        raise RuntimeError("Isaac stream has no active viewport")
    viewport_ready, waited_frames = ViewportManager.wait_for_viewport(viewport=viewport, max_frames=120)
    if not viewport_ready:
        raise RuntimeError("Isaac stream viewport did not become ready within 120 rendered frames")

    camera = ViewportManager.get_camera(viewport)
    ViewportManager.set_resolution(
        (STREAM_WIDTH, STREAM_HEIGHT), render_product_or_viewport=viewport
    )
    ViewportManager.set_camera_view(camera, eye=STREAM_CAMERA_EYE, target=camera_target)
    return {
        "scene_prims_ready": True,
        "viewport_ready": True,
        "viewport_camera": str(camera.GetPath()),
        "viewport_waited_frames": waited_frames,
    }


def _set_stream_view(
    view_name: str,
    camera_prim_paths: dict[str, str],
    *,
    camera_target: tuple[float, float, float] = STREAM_CAMERA_TARGET,
) -> str:
    """Switch the one WebRTC viewport to a free or recorded camera on the render thread."""
    from isaacsim.core.rendering_manager import ViewportManager

    viewport = ViewportManager.get_viewport_api()
    if viewport is None:
        raise RuntimeError("Isaac stream has no active viewport")
    if view_name == "free":
        camera_path = "/OmniverseKit_Persp"
        ViewportManager.set_camera(camera_path, render_product_or_viewport=viewport)
        camera = ViewportManager.get_camera(viewport)
        ViewportManager.set_camera_view(
            camera, eye=STREAM_CAMERA_EYE, target=camera_target
        )
        return camera_path
    try:
        camera_path = camera_prim_paths[view_name]
    except KeyError as exc:
        raise ValueError(f"unknown stream view: {view_name}") from exc
    ViewportManager.set_camera(camera_path, render_product_or_viewport=viewport)
    return camera_path


def _pump_kit_frame(simulation_app: object, settings: object | None = None) -> None:
    """Update the Kit/WebRTC frame without letting Kit advance physics itself."""
    if settings is None:
        import carb

        settings = carb.settings.get_settings()
    play_setting = "/app/player/playSimulations"
    previous_play_flag = settings.get(play_setting)
    settings.set_bool(play_setting, False)
    try:
        simulation_app.update()
    finally:
        settings.set_bool(play_setting, True if previous_play_flag is None else bool(previous_play_flag))


class TelemetryStore:
    """Share simulator observations with WebSocket handler threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, object] = {
            "sim_ready": False,
            "scene_prims_ready": False,
            "viewport_ready": False,
            "viewport_camera": None,
            "actual_joint_positions": None,
            "finger_positions": None,
            "tracking_error_max_rad": None,
        }

    def update(self, **values: object) -> None:
        with self._lock:
            self._values.update(values)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._values)


class RecordingCommandStore:
    """Queue browser recording commands for execution on the simulator thread."""

    _COMMANDS = {"start", "success", "failure", "discard", "reset"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[tuple[str, str]] = []
        self._status: dict[str, object] = {
            "recording": False,
            "episode": None,
            "frames": 0,
            "task": "",
            "last_saved_path": None,
        }

    def submit(self, command: str, task: str = "") -> None:
        if command not in self._COMMANDS:
            raise ValueError(f"unknown recording command: {command}")
        if command == "start" and not task.strip():
            raise ValueError("start requires a non-empty task")
        with self._lock:
            self._pending.append((command, task.strip()))

    def drain(self) -> list[tuple[str, str]]:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
            return pending

    def update(self, **values: object) -> None:
        with self._lock:
            self._status.update(values)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)


class ViewCommandStore:
    """Queue active WebRTC viewport changes for the simulator thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._status = {"active": "free", "requested": "free", "camera_path": None}

    def submit(self, view_name: str) -> None:
        if view_name not in VIEW_NAMES:
            raise ValueError(f"unknown stream view: {view_name}")
        with self._lock:
            self._pending.append(view_name)
            self._status["requested"] = view_name

    def drain(self) -> list[str]:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
            return pending

    def activate(self, view_name: str, camera_path: str) -> None:
        with self._lock:
            self._status.update(active=view_name, requested=view_name, camera_path=camera_path)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)


def _encode_preview_packet(
    camera_name: str, *, frame_index: int, captured_at_ns: int, jpeg: bytes
) -> bytes:
    """Encode one small camera preview for the control WebSocket."""
    try:
        camera_id = PREVIEW_CAMERA_IDS[camera_name]
    except KeyError as exc:
        raise ValueError(f"unknown preview camera: {camera_name}") from exc
    if frame_index < 0 or captured_at_ns < 0:
        raise ValueError("preview frame index and timestamp must be non-negative")
    if not jpeg or len(jpeg) > PREVIEW_MAX_JPEG_BYTES:
        raise ValueError(
            f"preview JPEG must contain 1..{PREVIEW_MAX_JPEG_BYTES} bytes"
        )
    return PREVIEW_HEADER.pack(
        PREVIEW_MAGIC,
        PREVIEW_PROTOCOL_VERSION,
        camera_id,
        frame_index,
        captured_at_ns,
    ) + jpeg


class CameraPreviewStore:
    """Encode previews off the control thread and retain only the newest sample."""

    def __init__(self, *, target_fps: float = DEFAULT_PREVIEW_FPS) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._versions = {name: 0 for name in PREVIEW_CAMERA_IDS}
        self._packets: dict[str, bytes] = {}
        self._target_fps = target_fps
        self._pending: tuple[dict[str, object], int, int] | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._failure: BaseException | None = None
        self._submitted = 0
        self._encoded = 0
        self._dropped = 0
        self._encode_ms_ewma: float | None = None
        self._encoded_times: deque[float] = deque()

    @staticmethod
    def _encode_jpeg(image: object) -> bytes:
        import cv2

        resized = cv2.resize(
            image, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_AREA
        )
        bgr = cv2.cvtColor(resized[..., :3], cv2.COLOR_RGB2BGR)
        encoded, buffer = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY]
        )
        if not encoded:
            raise RuntimeError("OpenCV failed to encode camera preview")
        return bytes(buffer)

    def update(
        self, images: dict[str, object], *, frame_index: int, captured_at_ns: int
    ) -> None:
        """Synchronously encode and atomically publish one four-camera sample."""
        packets: dict[str, bytes] = {}
        for camera_name in PREVIEW_CAMERA_IDS:
            if camera_name not in images:
                raise ValueError(f"preview images missing camera {camera_name!r}")
            packets[camera_name] = _encode_preview_packet(
                camera_name,
                frame_index=frame_index,
                captured_at_ns=captured_at_ns,
                jpeg=self._encode_jpeg(images[camera_name]),
            )
        with self._lock:
            for camera_name, packet in packets.items():
                self._versions[camera_name] += 1
                self._packets[camera_name] = packet

    def start(self) -> None:
        """Start the bounded latest-frame encoder worker."""
        with self._condition:
            if self._thread is not None:
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run, name="camera-preview-encoder", daemon=False
            )
            thread = self._thread
        thread.start()

    def submit(
        self, images: dict[str, object], *, frame_index: int, captured_at_ns: int
    ) -> None:
        """Queue the newest preview sample without blocking simulator control."""
        if set(images) != set(CAMERA_NAMES):
            raise ValueError(
                f"preview images have cameras {sorted(images)}, expected {list(CAMERA_NAMES)}"
            )
        with self._condition:
            if self._failure is not None:
                raise RuntimeError("camera preview encoder failed") from self._failure
            if self._thread is None or self._stopping:
                raise RuntimeError("camera preview encoder is not running")
            if self._pending is not None:
                self._dropped += 1
            self._pending = (dict(images), frame_index, captured_at_ns)
            self._submitted += 1
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopping
                )
                if self._pending is None and self._stopping:
                    return
                assert self._pending is not None
                images, frame_index, captured_at_ns = self._pending
                self._pending = None
            started = time.perf_counter()
            try:
                self.update(
                    images, frame_index=frame_index, captured_at_ns=captured_at_ns
                )
            except BaseException as exc:
                with self._condition:
                    self._failure = exc
                    self._stopping = True
                    self._condition.notify_all()
                return
            finished = time.perf_counter()
            encode_ms = (finished - started) * 1000.0
            with self._condition:
                self._encoded += 1
                self._encode_ms_ewma = (
                    encode_ms
                    if self._encode_ms_ewma is None
                    else 0.2 * encode_ms + 0.8 * self._encode_ms_ewma
                )
                self._encoded_times.append(finished)
                while self._encoded_times and finished - self._encoded_times[0] > 2.0:
                    self._encoded_times.popleft()

    def stop(self) -> None:
        """Flush the newest pending sample and stop the encoder worker."""
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            self._stopping = True
            self._condition.notify_all()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("camera preview encoder did not stop cleanly")
        with self._condition:
            self._thread = None
            if self._failure is not None:
                raise RuntimeError("camera preview encoder failed") from self._failure

    def stats(self) -> dict[str, object]:
        with self._lock:
            actual_fps = 0.0
            if len(self._encoded_times) >= 2:
                elapsed = self._encoded_times[-1] - self._encoded_times[0]
                if elapsed > 0:
                    actual_fps = (len(self._encoded_times) - 1) / elapsed
            return {
                "target_fps": self._target_fps,
                "actual_fps": round(actual_fps, 1),
                "encode_ms": (
                    round(self._encode_ms_ewma, 1)
                    if self._encode_ms_ewma is not None
                    else None
                ),
                "submitted": self._submitted,
                "encoded": self._encoded,
                "dropped": self._dropped,
                "pending": self._pending is not None,
            }

    def packets_since(self, versions: dict[str, int]) -> list[tuple[str, int, bytes]]:
        """Return only camera packets newer than a client's last sent versions."""
        with self._lock:
            return [
                (camera_name, self._versions[camera_name], self._packets[camera_name])
                for camera_name in PREVIEW_CAMERA_IDS
                if camera_name in self._packets
                and self._versions[camera_name] > versions.get(camera_name, -1)
            ]


class GelloWebServer:
    """Serve the Isaac Collector page and receive its selected control source."""

    def __init__(
        self,
        controller: GelloCommandController,
        telemetry: TelemetryStore,
        recording: RecordingCommandStore | None = None,
        previews: CameraPreviewStore | None = None,
        views: ViewCommandStore | None = None,
        keyboard: BrowserSe3KeyboardState | None = None,
        *,
        control_source: str = "gello",
        host: str,
        port: int,
        stream_host: str = "",
        signaling_port: int = 49100,
        media_port: int = 47998,
        stream_fps: int = 30,
        control_hz: float = 20.0,
    ) -> None:
        if control_source not in CONTROL_SOURCES:
            raise ValueError(f"unsupported control source: {control_source}")
        if control_source == "keyboard" and keyboard is None:
            raise ValueError("keyboard control requires BrowserSe3KeyboardState")
        self._controller = controller
        self._keyboard = keyboard
        self._control_source = control_source
        self._telemetry = telemetry
        self._recording = recording or RecordingCommandStore()
        self._previews = previews or CameraPreviewStore()
        self._views = views or ViewCommandStore()
        self._host = host
        self._port = port
        self._stream_host = stream_host
        self._signaling_port = signaling_port
        self._media_port = media_port
        self._stream_fps = stream_fps
        self._control_hz = control_hz
        self._thread: threading.Thread | None = None
        self._server = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None

    @staticmethod
    def _respond(connection, status: HTTPStatus, text: str):
        """Build an HTTP response across websockets 12 and 17 APIs."""
        if hasattr(connection, "respond"):
            return connection.respond(status, text)
        return connection.protocol.reject(status, text)

    @staticmethod
    def _set_header(response, name: str, value: str) -> None:
        """Replace a response header instead of appending a duplicate value."""
        try:
            del response.headers[name]
        except KeyError:
            pass
        response.headers[name] = value

    def _status_payload(self, *, log: str | None = None) -> str:
        selected_status = (
            self._keyboard.status()
            if self._control_source == "keyboard" and self._keyboard is not None
            else self._controller.status()
        )
        payload = {
            "type": "status",
            "backend": "isaaclab",
            "control_source": self._control_source,
            "control": selected_status,
            # Retained for existing GELLO clients while the UI migrates to ``control``.
            "gello": self._controller.status(),
            "recording": self._recording.snapshot(),
            "previews": self._previews.stats(),
            "view": self._views.snapshot(),
            **self._telemetry.snapshot(),
        }
        if log is not None:
            payload["log"] = log
        return json.dumps(payload)

    def _handler(self, websocket) -> None:
        from websockets.exceptions import ConnectionClosed

        sent_versions = {camera_name: -1 for camera_name in PREVIEW_CAMERA_IDS}
        bound_control_session_id: str | None = None

        def send_updates(*, log: str | None = None) -> None:
            websocket.send(self._status_payload(log=log))
            for camera_name, version, packet in self._previews.packets_since(sent_versions):
                websocket.send(packet)
                sent_versions[camera_name] = version

        try:
            send_updates(log="Client connected")
            while True:
                try:
                    message = websocket.recv(timeout=0.1)
                except TimeoutError:
                    send_updates()
                    continue

                try:
                    if not isinstance(message, str):
                        raise ValueError("GELLO commands must be JSON text messages")
                    data = json.loads(message)
                    message_type = data.get("type")
                    if message_type == "gello":
                        if self._control_source != "gello":
                            raise ValueError("GELLO commands are disabled in keyboard mode")
                        command = GelloCommand.model_validate(data)
                        accepted = self._controller.submit(command)
                        if accepted:
                            bound_control_session_id = command.session_id
                        log = (
                            f"Accepted GELLO sequence {command.sequence}"
                            if accepted
                            else f"Ignored stale or competing GELLO sequence {command.sequence}"
                        )
                    elif message_type == "keyboard":
                        if self._control_source != "keyboard" or self._keyboard is None:
                            raise ValueError("keyboard commands are disabled in GELLO mode")
                        accepted = self._keyboard.submit(data)
                        if accepted:
                            bound_control_session_id = str(data["session_id"])
                        log = (
                            f"Accepted keyboard sequence {data['sequence']}"
                            if accepted
                            else f"Ignored stale or competing keyboard sequence {data['sequence']}"
                        )
                    elif message_type == "command":
                        command_name = data.get("command")
                        task = data.get("task", "")
                        if not isinstance(command_name, str) or not isinstance(task, str):
                            raise ValueError("recording command and task must be strings")
                        self._recording.submit(command_name, task)
                        log = f"Queued recording command {command_name}"
                    elif message_type == "view":
                        view_name = data.get("view")
                        if not isinstance(view_name, str):
                            raise ValueError("view must be a string")
                        self._views.submit(view_name)
                        log = f"Queued stream view {view_name}"
                    else:
                        raise ValueError(
                            "expected message type 'gello', 'keyboard', 'command', or 'view'"
                        )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    log = f"ERROR: {exc}"
                send_updates(log=log)
        except ConnectionClosed:
            return
        finally:
            if bound_control_session_id is not None:
                if self._control_source == "keyboard" and self._keyboard is not None:
                    self._keyboard.release_session(bound_control_session_id)
                else:
                    self._controller.release_session(bound_control_session_id)

    def _process_request(self, connection, request):
        if "Upgrade" in request.headers:
            return None
        path = request.path.split("?", 1)[0]
        if path == "/":
            asset = WEB_CONSOLE_ROOT / "index.html"
            content_type = "text/html; charset=utf-8"
        elif path == "/teleop-config.json":
            response = self._respond(
                connection,
                HTTPStatus.OK,
                json.dumps(
                    {
                        "stream_host": self._stream_host,
                        "signaling_port": self._signaling_port,
                        "media_port": self._media_port,
                        "width": STREAM_WIDTH,
                        "height": STREAM_HEIGHT,
                        "fps": self._stream_fps,
                        "control_source": self._control_source,
                        "control_hz": self._control_hz,
                    }
                ),
            )
            self._set_header(response, "Content-Type", "application/json")
            response.headers["Cache-Control"] = "no-store"
            return response
        elif path.startswith("/assets/"):
            assets_root = (WEB_CONSOLE_ROOT / "assets").resolve()
            asset = (WEB_CONSOLE_ROOT / path.lstrip("/")).resolve()
            if not asset.is_relative_to(assets_root) or not asset.is_file():
                return self._respond(connection, HTTPStatus.NOT_FOUND, "Not found\n")
            content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        else:
            return self._respond(connection, HTTPStatus.NOT_FOUND, "Not found\n")
        if not asset.is_file():
            return self._respond(connection, HTTPStatus.NOT_FOUND, "Web console is not built\n")
        response = self._respond(connection, HTTPStatus.OK, asset.read_text(encoding="utf-8"))
        self._set_header(response, "Content-Type", content_type)
        response.headers["Cache-Control"] = "no-store"
        return response

    def start(self) -> None:
        """Start serving and wait until the listening socket is ready."""

        def run() -> None:
            try:
                from websockets.sync.server import serve

                with serve(
                    self._handler,
                    self._host,
                    self._port,
                    process_request=self._process_request,
                    max_size=2**20,
                ) as server:
                    self._server = server
                    self._ready.set()
                    server.serve_forever()
            except BaseException as exc:
                self._failure = exc
                self._ready.set()

        self._thread = threading.Thread(target=run, name="gello-web-server", daemon=False)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Timed out starting GELLO WebSocket server")
        if self._failure is not None:
            raise RuntimeError("Failed to start GELLO WebSocket server") from self._failure

    def stop(self) -> None:
        """Stop the listening socket and join the server thread."""
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("GELLO WebSocket server did not stop cleanly")


@contextmanager
def _running_control_services(
    web_server: GelloWebServer, previews: CameraPreviewStore
):
    """Start control services and unwind each successful acquisition in reverse order."""
    web_server.start()
    try:
        previews.start()
        try:
            yield
        finally:
            previews.stop()
    finally:
        web_server.stop()


def _parse_args() -> tuple[argparse.Namespace, object]:
    from isaaclab.app import AppLauncher

    class SingleVisibleGpuAppLauncher(AppLauncher):
        """Launch with distinct physical Vulkan and logical CUDA ordinals."""

        def _resolve_device_settings(self, launcher_args: dict) -> None:
            super()._resolve_device_settings(launcher_args)
            _apply_split_gpu_launcher_config(
                launcher_args, int(launcher_args["renderer_gpu_id"])
            )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-source",
        choices=CONTROL_SOURCES,
        default="gello",
        help="lock execution to browser keyboard, GELLO, OpenPI, or a resolved trajectory plan",
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-config", default="pi05_franka_joint8_lora")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--inference-steps", type=int, default=2)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--rollout-count", type=int, default=20)
    parser.add_argument("--max-episode-frames", type=int, default=400)
    parser.add_argument("--success-settle-frames", type=int, default=10)
    parser.add_argument(
        "--evaluation-plan",
        type=Path,
        help="validated held-out or M2-training XY conditions for an OpenPI evaluation",
    )
    parser.add_argument(
        "--evaluation-limit",
        type=int,
        help="run only the first N evaluation conditions (for bounded smoke tests)",
    )
    parser.add_argument(
        "--generation-plan",
        type=Path,
        help="resolved robot-data-generation plan for trajectory execution",
    )
    parser.add_argument(
        "--resume-generation",
        action="store_true",
        help="resume a matching generation batch report in --data-dir",
    )
    parser.add_argument(
        "--gripper-filter",
        choices=("single_threshold", "hysteresis"),
        default="single_threshold",
        help="convert OpenPI closed fractions with the legacy threshold or hysteresis",
    )
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-min-closed-seconds", type=float, default=0.0)
    parser.add_argument(
        "--full-horizon",
        action="store_true",
        help="record every requested frame even after task success for paired evaluations",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP/WebSocket bind host")
    parser.add_argument("--port", type=int, default=18081, help="HTTP/WebSocket bind port")
    parser.add_argument(
        "--stream-host",
        default=os.environ.get("PUBLIC_IP", ""),
        help="public WebRTC host exposed to the browser; defaults to PUBLIC_IP",
    )
    parser.add_argument("--signaling-port", type=int, default=49100, help="WebRTC signaling TCP port")
    parser.add_argument("--media-port", type=int, default=47998, help="WebRTC media UDP port")
    parser.add_argument(
        "--stream-fps", type=int, default=20, help="Kit/WebRTC viewport pump frequency"
    )
    parser.add_argument("--data-dir", default="data/isaac_episodes", help="raw episode output root")
    parser.add_argument("--seed", type=int, default=0, help="episode and simulator random seed")
    parser.add_argument("--scene", default="franka_empty", choices=ISAAC_SCENE_IDS, help="scene plugin id")
    parser.add_argument(
        "--work-surface-height-m",
        type=float,
        default=DEFAULT_WORK_SURFACE_HEIGHT_M,
        help="rigid-task work surface height above the Franka base [m]",
    )
    parser.add_argument("--width", type=int, default=640, help="recording camera width")
    parser.add_argument("--height", type=int, default=480, help="recording camera height")
    parser.add_argument(
        "--camera-profile-file",
        type=Path,
        help="Robot Data Studio camera-profile JSON; defaults to nominal D435i geometry",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="physical GPU index from nvidia-smi; configures simulation, PhysX, and RTX together",
    )
    parser.add_argument("--physics_hz", type=float, default=120.0, help="physics frequency [Hz]")
    parser.add_argument("--control_hz", type=float, default=20.0, help="teleoperation control frequency [Hz]")
    parser.add_argument(
        "--keyboard-position-step-m",
        type=float,
        default=0.01,
        help="official Se3Keyboard-style translation increment per control tick [m]",
    )
    parser.add_argument(
        "--keyboard-rotation-step-rad",
        type=float,
        default=0.05,
        help="official Se3Keyboard-style rotation increment per control tick [rad]",
    )
    parser.add_argument(
        "--dataset-fps",
        type=float,
        default=None,
        help="synchronized episode frequency; defaults to camera profile dataset_fps_default",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=DEFAULT_PREVIEW_FPS,
        help="base/top/side/wrist browser preview frequency",
    )
    parser.add_argument("--max_joint_speed", type=float, default=1.0, help="per-joint target slew limit [rad/s]")
    parser.add_argument("--gello_timeout", type=float, default=0.3, help="watchdog timeout [s]")
    parser.add_argument(
        "--franka-usd",
        default=None,
        help="optional Franka USD override; defaults to FRANKA_PANDA_HIGH_PD_CFG",
    )
    parser.add_argument(
        "--smoke_steps",
        type=int,
        default=0,
        help="run bounded internal-command physics steps without opening the WebSocket server",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        _configure_gpu_selection(args, sys.argv[1:])
    except ValueError as exc:
        parser.error(str(exc))
    if any(
        not 0 < port < 65536
        for port in (args.port, args.signaling_port, args.media_port, args.policy_port)
    ):
        parser.error("HTTP, signaling, and media ports must be in [1, 65535]")
    if min(
        args.physics_hz,
        args.control_hz,
        args.stream_fps,
        args.max_joint_speed,
        args.gello_timeout,
        args.keyboard_position_step_m,
        args.keyboard_rotation_step_rad,
    ) <= 0:
        parser.error("physics, control, speed, and timeout values must be positive")
    if args.smoke_steps < 0:
        parser.error("smoke_steps must be non-negative")
    if min(
        args.inference_steps,
        args.replan_steps,
        args.rollout_count,
        args.max_episode_frames,
        args.success_settle_frames,
    ) <= 0:
        parser.error("policy rollout values must be positive")
    if args.evaluation_limit is not None and args.evaluation_limit <= 0:
        parser.error("--evaluation-limit must be positive")
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (args.gripper_close_threshold, args.gripper_open_threshold)
    ):
        parser.error("gripper thresholds must be finite values in [0, 1]")
    if not math.isfinite(args.gripper_min_closed_seconds) or args.gripper_min_closed_seconds < 0:
        parser.error("gripper minimum closed seconds must be finite and non-negative")
    if (
        args.gripper_filter == "hysteresis"
        and args.gripper_close_threshold <= args.gripper_open_threshold
    ):
        parser.error("hysteresis close threshold must exceed open threshold")
    if args.control_source == "openpi":
        if args.scene != "rigid_cube_lift":
            parser.error("OpenPI rollout currently requires --scene rigid_cube_lift")
        if args.policy_checkpoint is None:
            parser.error("OpenPI rollout requires --policy-checkpoint")
        if args.smoke_steps:
            parser.error("OpenPI rollout does not use --smoke_steps")
        args.policy_checkpoint = args.policy_checkpoint.expanduser().resolve()
        if args.evaluation_plan is not None:
            args.evaluation_plan = args.evaluation_plan.expanduser().resolve()
    elif args.evaluation_plan is not None or args.evaluation_limit is not None:
        parser.error("evaluation plan options require --control-source openpi")
    if args.control_source == "trajectory":
        if args.scene != "rigid_cube_lift":
            parser.error("trajectory execution currently requires --scene rigid_cube_lift")
        if args.generation_plan is None:
            parser.error("trajectory execution requires --generation-plan")
        if args.smoke_steps:
            parser.error("trajectory execution does not use --smoke_steps")
        args.generation_plan = args.generation_plan.expanduser().resolve()
    elif args.generation_plan is not None or args.resume_generation:
        parser.error("generation plan options require --control-source trajectory")
    try:
        args.work_surface_height_m = _validate_work_surface_height(
            args.work_surface_height_m
        )
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.width, args.height) <= 0:
        parser.error("recording camera dimensions must be positive")
    try:
        args.camera_profile = _load_camera_profile(
            args.camera_profile_file, width=args.width, height=args.height
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.dataset_fps is None:
        args.dataset_fps = float(args.camera_profile.get("dataset_fps_default", 10.0))
    if not math.isclose(args.dataset_fps, round(args.dataset_fps), rel_tol=0.0, abs_tol=1e-9):
        parser.error("dataset-fps must be an integer because EpisodeWriter stores an integer fps")
    args.gripper_min_closed_frames = int(
        math.ceil(args.gripper_min_closed_seconds * args.dataset_fps)
    )
    decimation = args.physics_hz / args.control_hz
    if not math.isclose(decimation, round(decimation), rel_tol=0.0, abs_tol=1e-9):
        parser.error("physics_hz must be an integer multiple of control_hz")
    try:
        args.dataset_control_decimation = _frequency_decimation(
            args.control_hz, args.dataset_fps, label="dataset-fps"
        )
        args.preview_control_decimation = _frequency_decimation(
            args.control_hz, args.preview_fps, label="preview-fps"
        )
        args.stream_physics_decimation = _frequency_decimation(
            args.physics_hz, args.stream_fps, label="stream-fps"
        )
    except ValueError as exc:
        parser.error(str(exc))
    app_launcher = SingleVisibleGpuAppLauncher(args)
    return args, app_launcher.app


def _run_simulator(args: argparse.Namespace, simulation_app: object) -> bool:
    import importlib.metadata

    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

    sim_dt = 1.0 / args.physics_hz
    control_decimation = int(round(args.physics_hz / args.control_hz))
    stream_decimation = args.stream_physics_decimation
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args.device))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)

    if args.scene == "rigid_pick_place":
        scene_geometry = _rigid_pick_place_geometry(args.work_surface_height_m)
    elif args.scene == "rigid_cube_lift":
        scene_geometry = _rigid_cube_lift_geometry(args.work_surface_height_m)
    else:
        scene_geometry = None
    work_surface_top_m = (
        float(scene_geometry["work_surface_top_m"])
        if scene_geometry is not None
        else None
    )
    stream_camera_target = (
        (0.45, 0.0, work_surface_top_m + 0.15)
        if work_surface_top_m is not None
        else STREAM_CAMERA_TARGET
    )
    task_object: object | None = None
    if scene_geometry is not None:
        work_surface_cfg = sim_utils.CuboidCfg(
            size=scene_geometry["work_surface_size_m"],
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.67, 0.54, 0.38)),
        )
        work_surface_cfg.func(
            "/World/WorkSurface",
            work_surface_cfg,
            translation=scene_geometry["work_surface_center_m"],
        )
        if "target_size_m" in scene_geometry:
            target_cfg = sim_utils.CuboidCfg(
                size=scene_geometry["target_size_m"],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.48, 0.92)),
            )
            target_cfg.func(
                "/World/TargetZone",
                target_cfg,
                translation=scene_geometry["target_center_m"],
            )
        object_cfg = RigidObjectCfg(
            prim_path="/World/TaskObject",
            spawn=sim_utils.CuboidCfg(
                size=(scene_geometry["object_size_m"],) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.86, 0.38)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=scene_geometry["object_start_m"]
            ),
        )
        task_object = RigidObject(cfg=object_cfg)

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Franka")
    if args.franka_usd:
        robot_cfg.spawn.usd_path = args.franka_usd
    for index, value in enumerate(FRANKA_HOME, start=1):
        robot_cfg.init_state.joint_pos[f"panda_joint{index}"] = value
    robot_cfg.init_state.joint_pos["panda_finger_joint.*"] = 0.04
    robot = Articulation(cfg=robot_cfg)
    record_cameras: dict[str, object] = {}
    camera_prim_paths: dict[str, str] = {}
    camera_intrinsic_sources: dict[str, str] = {}
    camera_intrinsic_matrices: dict[str, list[float]] = {}
    if args.smoke_steps == 0:
        from isaaclab.sensors import Camera, CameraCfg

        profile = args.camera_profile
        isaac_overrides = _scene_camera_overrides(
            profile, work_surface_top_m=work_surface_top_m
        )
        camera_sensor_fps = max(args.dataset_fps, args.preview_fps)
        for camera_name, role_name, prim_path, offset in (
            ("base", "agentview", "/World/RecordCamera", None),
            ("top", "agentview", "/World/RecordCameraTop", None),
            ("side", "agentview", "/World/RecordCameraSide", None),
            (
                "wrist",
                "wrist",
                isaac_overrides["wrist"]["prim_path"],
                CameraCfg.OffsetCfg(
                    pos=tuple(isaac_overrides["wrist"]["position_parent_m"]),
                    rot=tuple(isaac_overrides["wrist"]["rotation_parent_xyzw"]),
                    convention=isaac_overrides["wrist"].get("convention", "ros"),
                ),
            ),
        ):
            matrix, intrinsic_source = _camera_intrinsic_matrix(
                profile["roles"][role_name],
                width=args.width,
                height=args.height,
                nominal_vertical_fov_deg=float(
                    isaac_overrides[camera_name].get(
                        "nominal_vertical_fov_deg",
                        profile["roles"][role_name].get("nominal_vertical_fov_deg", 42.0),
                    )
                ),
            )
            camera_intrinsic_sources[camera_name] = intrinsic_source
            camera_intrinsic_matrices[camera_name] = matrix
            cfg_kwargs: dict[str, object] = {
                "prim_path": prim_path,
                "update_period": 1.0 / camera_sensor_fps,
                "height": args.height,
                "width": args.width,
                "data_types": ["rgb"],
                "spawn": sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    matrix,
                    width=args.width,
                    height=args.height,
                    clipping_range=(0.01, 100.0),
                ),
            }
            if offset is not None:
                cfg_kwargs["offset"] = offset
            record_cameras[camera_name] = Camera(cfg=CameraCfg(**cfg_kwargs))
            camera_prim_paths[camera_name] = prim_path
    sim.reset()

    def reset_task_assets(
        object_start_position_m: tuple[float, float, float] | None = None,
    ) -> None:
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch)
        robot.reset()
        robot.update(sim_dt)
        if task_object is not None:
            root_state = task_object.data.default_root_state.torch.clone()
            if object_start_position_m is not None:
                root_state[:, :3] = torch.tensor(
                    [object_start_position_m], dtype=torch.float32, device=sim.device
                )
            task_object.write_root_pose_to_sim_index(root_pose=root_state[:, :7])
            task_object.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:])
            task_object.reset()
            task_object.update(sim_dt)

    reset_task_assets()
    if record_cameras:
        for camera_name in ("base", "top", "side"):
            camera_override = isaac_overrides[camera_name]
            record_cameras[camera_name].set_world_poses_from_view(
                torch.tensor(
                    [camera_override["eye_world_m"]], dtype=torch.float32, device=sim.device
                ),
                torch.tensor(
                    [camera_override["target_world_m"]], dtype=torch.float32, device=sim.device
                ),
            )
        for camera in record_cameras.values():
            camera.update(dt=sim_dt)

    scene_status: dict[str, object] = {}
    if args.smoke_steps == 0:
        scene_status = _configure_stream_view(
            camera_target=stream_camera_target,
            required_prims=_required_scene_prims(scene_geometry),
        )
        # Publish the newly-bound camera and populated stage before accepting a browser client.
        _pump_kit_frame(simulation_app)
        print(f"ISAAC_SCENE_READY {json.dumps(scene_status, sort_keys=True)}", file=sys.__stderr__, flush=True)

    backend = IsaacLabFrankaBackend(robot, device=sim.device)
    initial_joint_positions = backend.current_joint_positions()
    controller = GelloCommandController(
        initial_joint_positions,
        backend.joint_position_limits(),
        control_hz=args.control_hz,
        max_joint_speed=args.max_joint_speed,
        watchdog_timeout=args.gello_timeout,
    )
    keyboard = BrowserSe3KeyboardState(
        position_step_m=args.keyboard_position_step_m,
        rotation_step_rad=args.keyboard_rotation_step_rad,
        watchdog_timeout=args.gello_timeout,
    )
    diff_ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        num_envs=1,
        device=sim.device,
    )
    arm_joint_ids = robot.find_joints(
        [f"panda_joint{index}" for index in range(1, 8)], preserve_order=True
    )[0]
    hand_body_index = robot.find_bodies("panda_hand")[0][0]
    hand_jacobian_index = hand_body_index - 1 if robot.is_fixed_base else hand_body_index
    jacobian_joint_ids = [index + robot.num_base_dofs for index in arm_joint_ids]

    def keyboard_joint_targets(
        sample: BrowserKeyboardSample, hold_targets: tuple[float, ...]
    ) -> tuple[float, ...]:
        """Map an official-style relative SE(3) command through Isaac Lab DLS IK."""
        if not any(sample.delta_pose):
            return hold_targets
        ee_pose_w = robot.data.body_pose_w.torch[:, hand_body_index]
        root_pose_w = robot.data.root_pose_w.torch
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )
        command = torch.tensor(
            [sample.delta_pose], dtype=torch.float32, device=sim.device
        )
        diff_ik_controller.set_command(command, ee_pos_b, ee_quat_b)
        jacobian = robot.data.body_link_jacobian_w.torch[
            :, hand_jacobian_index, :, jacobian_joint_ids
        ]
        joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
        desired = diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )
        if not bool(torch.isfinite(desired).all().item()):
            raise RuntimeError("Differential IK produced a non-finite Franka target")
        return tuple(float(value) for value in desired[0].detach().cpu().tolist())

    trajectory_ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        num_envs=1,
        device=sim.device,
    )

    def trajectory_joint_targets(target_pose_w: np.ndarray) -> tuple[float, ...]:
        """Convert one world-frame target pose (xyz + quaternion wxyz) through DLS IK."""
        target = torch.as_tensor(
            target_pose_w, dtype=torch.float32, device=sim.device
        ).reshape(1, 7)
        root_pose_w = robot.data.root_pose_w.torch
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], target[:, :3], target[:, 3:7]
        )
        trajectory_ik_controller.set_command(
            torch.cat((target_pos_b, target_quat_b), dim=-1)
        )
        ee_pose_w = robot.data.body_pose_w.torch[:, hand_body_index]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        jacobian = robot.data.body_link_jacobian_w.torch[
            :, hand_jacobian_index, :, jacobian_joint_ids
        ]
        joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
        desired = trajectory_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )
        if not bool(torch.isfinite(desired).all().item()):
            raise RuntimeError("trajectory IK produced a non-finite Franka target")
        return tuple(float(value) for value in desired[0].detach().cpu().tolist())

    def recover_controller_target_poses(applied_action: np.ndarray) -> np.ndarray:
        """Run FK on the loaded Isaac articulation for each applied arm target."""

        actions = np.asarray(applied_action, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
            raise ValueError("FK recovery requires finite applied_action with shape [frames, 8]")
        targets = torch.as_tensor(actions[:, :7], dtype=torch.float32, device=sim.device)
        zero_velocity = torch.zeros((1, len(arm_joint_ids)), dtype=torch.float32, device=sim.device)
        poses: list[np.ndarray] = []
        try:
            for target in targets:
                robot.write_joint_position_to_sim_index(
                    position=target.reshape(1, -1), joint_ids=arm_joint_ids
                )
                robot.write_joint_velocity_to_sim_index(
                    velocity=zero_velocity, joint_ids=arm_joint_ids
                )
                sim.forward()
                # Advance the asset-data timestamp so lazy body-pose buffers are
                # invalidated after each kinematic write. Physics itself is not stepped.
                robot.update(sim_dt)
                pose = (
                    robot.data.body_pose_w.torch[0, hand_body_index]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                if pose.shape != (7,) or not np.isfinite(pose).all():
                    raise RuntimeError("Isaac FK produced a non-finite panda_hand pose")
                poses.append(pose)
        finally:
            reset_task_assets()
        return np.asarray(poses, dtype=np.float32)

    telemetry = TelemetryStore()
    scene_task = (
        str(scene_geometry["task_instruction"])
        if scene_geometry is not None
        else f"Move the Franka safely with {args.control_source}"
    )
    telemetry.update(
        sim_ready=True,
        control_source=args.control_source,
        scene_id=args.scene,
        scene_task=scene_task,
        work_surface_top_m=work_surface_top_m,
        scene_geometry=scene_geometry,
        **scene_status,
    )
    recording = RecordingCommandStore()
    views = ViewCommandStore()
    if args.smoke_steps == 0:
        views.activate("free", str(scene_status["viewport_camera"]))
    previews = CameraPreviewStore(target_fps=args.preview_fps)
    writer = None
    cube_lift_evaluator = (
        CubeLiftEvaluator(
            fps=int(round(args.dataset_fps)),
            initial_object_z_m=float(scene_geometry["object_start_m"][2]),
        )
        if args.scene == "rigid_cube_lift" and scene_geometry is not None
        else None
    )
    make_episode_writer = None
    if args.smoke_steps == 0:
        from robot_data_contracts import AugmentationLineageV1, FrankaEpisodeWriter

        try:
            engine_version = importlib.metadata.version("isaacsim")
        except importlib.metadata.PackageNotFoundError:
            engine_version = "6.0.1"
        state_names = (
            [f"joint_position/panda_joint{index}_rad" for index in range(1, 8)]
            + ["joint_position/panda_finger_joint1_m", "joint_position/panda_finger_joint2_m"]
            + [f"joint_velocity/panda_joint{index}_rad_s" for index in range(1, 8)]
            + ["joint_velocity/panda_finger_joint1_m_s", "joint_velocity/panda_finger_joint2_m_s"]
        )
        action_names = [f"target_joint_position/panda_joint{index}_rad" for index in range(1, 8)] + [
            "target_joint_position/panda_finger_joint1_m",
            "target_joint_position/panda_finger_joint2_m",
        ]
        base_run_metadata = {
            "isaaclab_device": args.device,
            "gpu_id": args.gpu_id,
            "physics_hz": args.physics_hz,
            "control_hz": args.control_hz,
            "control_source_options": [
                "browser_se3_keyboard",
                "gello_joint_targets",
                "openpi_policy",
                "retargeted_trajectory",
            ],
            "operator_command_layouts": {
                "browser_se3_keyboard": [
                    "dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad",
                    "gripper_closed",
                ],
                "gello_joint_targets": [
                    *[f"panda_joint{index}_rad" for index in range(1, 8)],
                    "gripper_closed",
                ],
                "retargeted_trajectory": [
                    "target_x_m", "target_y_m", "target_z_m", "target_qw", "target_qx",
                    "target_qy", "target_qz", "gripper_closed",
                ],
            },
            "keyboard_mapping": KEYBOARD_AXIS_COMMANDS,
            "keyboard_position_step_m": args.keyboard_position_step_m,
            "keyboard_rotation_step_rad": args.keyboard_rotation_step_rad,
            "keyboard_ik": {
                "command_type": "pose",
                "use_relative_mode": True,
                "method": "dls",
            },
            "dataset_fps": args.dataset_fps,
            "preview_fps": args.preview_fps,
            "stream_fps": args.stream_fps,
            "franka_usd": args.franka_usd,
            "camera_profile": _camera_profile_metadata(args.camera_profile),
            "camera_intrinsic_sources": camera_intrinsic_sources,
            "camera_intrinsic_matrices": camera_intrinsic_matrices,
            "camera_extrinsics": isaac_overrides,
            "camera_roles": {
                "base": "third-person agentview",
                "top": "virtual auxiliary top-down XY alignment",
                "side": "virtual auxiliary side-view height and contact",
                "wrist": "Franka panda_hand wrist view",
            },
            "automatic_outcome_available": cube_lift_evaluator is not None,
            "scene_geometry": scene_geometry,
            "git_roots": {"gello_remote": Path(__file__).resolve().parents[2]},
            "dependency_packages": ["isaacsim", "numpy"],
        }

        def make_episode_writer(
            *,
            control_source: str,
            operator_command_names: list[str],
            seed: int,
            generation_method: str | None = None,
            lineage: object | None = None,
            metadata: dict[str, object] | None = None,
        ):
            run_metadata = dict(base_run_metadata)
            run_metadata["control_source"] = control_source
            if metadata:
                run_metadata.update(metadata)
            return FrankaEpisodeWriter(
                args.data_dir,
                fps=int(round(args.dataset_fps)),
                engine_id="isaac_sim",
                engine_version=engine_version,
                scene_id=args.scene,
                seed=seed,
                worker_id="gello_remote_isaac_sim",
                control_source=control_source,
                state_names=state_names,
                action_names=action_names,
                operator_command_names=operator_command_names,
                camera_names=list(CAMERA_NAMES),
                run_metadata=run_metadata,
                generation_method=generation_method,
                lineage=lineage,
            )

        if args.control_source != "trajectory":
            control_source = (
                "browser_se3_keyboard"
                if args.control_source == "keyboard"
                else ("openpi_policy" if args.control_source == "openpi" else "gello_joint_targets")
            )
            operator_command_names = (
                ["dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper_closed"]
                if args.control_source == "keyboard"
                else (
                    [
                        *(f"model_target_joint/panda_joint{index}_rad" for index in range(1, 8)),
                        "model_target_gripper_closed_fraction",
                    ]
                    if args.control_source == "openpi"
                    else [f"requested_joint_position/panda_joint{index}_rad" for index in range(1, 8)]
                    + ["requested_gripper_closed"]
                )
            )
            policy_metadata = (
                {
                    "policy": {
                        "consumer": "openpi_franka_joint8",
                        "config": args.policy_config,
                        "checkpoint": str(args.policy_checkpoint),
                        "server": f"ws://{args.policy_host}:{args.policy_port}",
                        "replan_steps": args.replan_steps,
                        "inference_steps": args.inference_steps,
                        "gripper_filter": {
                            "mode": args.gripper_filter,
                            "close_threshold": args.gripper_close_threshold,
                            "open_threshold": args.gripper_open_threshold,
                            "min_closed_seconds": args.gripper_min_closed_seconds,
                            "min_closed_frames": args.gripper_min_closed_frames,
                        },
                        "full_horizon": args.full_horizon,
                        "evaluation_plan": (
                            str(args.evaluation_plan) if args.evaluation_plan else None
                        ),
                    }
                }
                if args.control_source == "openpi"
                else None
            )
            writer = make_episode_writer(
                control_source=control_source,
                operator_command_names=operator_command_names,
                seed=args.seed,
                generation_method=("policy_rollout" if args.control_source == "openpi" else None),
                metadata=policy_metadata,
            )

    if args.control_source == "trajectory":
        from robot_data_generation.generation_plan import (
            load_batch_report,
            load_generation_plan,
            new_batch_report,
            pending_attempts,
            record_attempt_result,
            write_json_atomic,
        )
        from robot_data_generation.models import AttemptResultV1, SourceEpisodeRefV1
        from robot_data_generation.rigid_retarget import retarget_rigid_cube_lift
        from robot_data_generation.source_episode import load_source_trajectory

        if make_episode_writer is None or cube_lift_evaluator is None or task_object is None:
            raise RuntimeError(
                "trajectory execution requires episode writer, cube evaluator, and task object"
            )
        plan = load_generation_plan(args.generation_plan)
        if plan.execution is None:
            raise RuntimeError(
                "trajectory execution requires a v2 plan with resolved execution settings"
            )
        if not math.isclose(
            args.control_hz,
            plan.execution.control_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"--control_hz={args.control_hz} does not match resolved plan "
                f"control_hz={plan.execution.control_hz}"
            )
        output_root = Path(args.data_dir).expanduser().resolve()
        report_path = output_root / "generation_batch_report.json"
        if report_path.exists():
            if not args.resume_generation:
                raise RuntimeError(
                    f"generation report already exists: {report_path}; use --resume-generation"
                )
            report = load_batch_report(report_path, plan=plan)
        else:
            if args.resume_generation:
                raise RuntimeError(
                    f"cannot resume because generation report does not exist: {report_path}"
                )
            report = new_batch_report(plan, args.generation_plan)
            write_json_atomic(report_path, report)

        source_cache: dict[str, tuple[object, np.ndarray]] = {}
        trajectory_controller = GelloCommandController(
            initial_joint_positions,
            backend.joint_position_limits(),
            control_hz=args.control_hz,
            max_joint_speed=args.max_joint_speed,
            watchdog_timeout=args.gello_timeout,
        )
        physics_steps_per_control = control_decimation
        dataset_control_decimation = args.dataset_control_decimation
        pregrasp_timeout_ticks = int(
            math.ceil(plan.execution.pregrasp_timeout_s * args.control_hz)
        )

        attempt_order = {
            attempt.attempt_id: index for index, attempt in enumerate(plan.attempts)
        }

        for attempt in pending_attempts(plan, report):
            attempt_writer = None
            episode_path: Path | None = None
            metrics: dict[str, object] = {}
            try:
                np.random.seed(attempt.randomization_seed)
                torch.manual_seed(attempt.randomization_seed)
                source_key = str(attempt.source_episode_path)
                cached_source = source_cache.get(source_key)
                if cached_source is None:
                    source = load_source_trajectory(
                        SourceEpisodeRefV1(
                            path=attempt.source_episode_path,
                            episode_uuid=attempt.source_episode_uuid,
                        ),
                        grasp_lift_threshold_m=(
                            plan.execution.grasp_lift_threshold_m
                        ),
                        task_lift_threshold_m=plan.execution.task_lift_threshold_m,
                    )
                    controller_target_pose = recover_controller_target_poses(
                        source.applied_action
                    )
                    source_cache[source_key] = (source, controller_target_pose)
                else:
                    source, controller_target_pose = cached_source
                trajectory = retarget_rigid_cube_lift(
                    source,
                    controller_target_pose,
                    attempt.object_xy_offset_m,
                    transition_frames=attempt.transition_frames,
                    target_hz=plan.execution.control_hz,
                )
                max_control_steps = int(
                    args.max_episode_frames * dataset_control_decimation
                )
                worst_case_control_steps = trajectory.length + pregrasp_timeout_ticks
                if worst_case_control_steps > max_control_steps:
                    raise ValueError(
                        f"trajectory may require {worst_case_control_steps} control ticks, "
                        f"exceeding the {max_control_steps}-tick dataset horizon"
                    )

                reset_task_assets(trajectory.object_start_position_m)
                trajectory_controller.reset_targets(initial_joint_positions)
                trajectory_ik_controller.reset()
                cube_lift_evaluator = CubeLiftEvaluator(
                    fps=int(round(args.control_hz)),
                    initial_object_z_m=trajectory.object_start_position_m[2],
                    grasp_threshold_m=plan.execution.grasp_lift_threshold_m,
                    lift_threshold_m=plan.execution.task_lift_threshold_m,
                    success_mode=plan.execution.success_mode,
                )
                lineage = AugmentationLineageV1(
                    parent_episode_uuid=attempt.source_episode_uuid,
                    generation_method=(
                        "physics_replay" if plan.nominal_only else "retargeted"
                    ),
                    randomization_spec_id=plan.spec_id,
                    randomization_seed=attempt.randomization_seed,
                    resolved_parameters={
                        "attempt_id": attempt.attempt_id,
                        "object_xy_offset_m": list(attempt.object_xy_offset_m),
                        "object_start_position_m": list(
                            trajectory.object_start_position_m
                        ),
                        "transition_frames": attempt.transition_frames,
                        "source_pose_reconstruction": "isaac_fk_from_applied_action",
                        "source_fps": source.fps,
                        "execution_control_hz": plan.execution.control_hz,
                        "pose_interpolation": "linear_position_shortest_path_slerp",
                        "pregrasp_position_tolerance_m": (
                            plan.execution.pregrasp_position_tolerance_m
                        ),
                        "pregrasp_rotation_tolerance_rad": (
                            plan.execution.pregrasp_rotation_tolerance_rad
                        ),
                        "pregrasp_stable_ticks": plan.execution.pregrasp_stable_ticks,
                        "pregrasp_timeout_s": plan.execution.pregrasp_timeout_s,
                        "success_mode": plan.execution.success_mode,
                        "nominal_only": plan.nominal_only,
                    },
                    trajectory_modified=not plan.nominal_only,
                )
                attempt_writer = make_episode_writer(
                    control_source="retargeted_trajectory",
                    operator_command_names=[
                        "target_x_m",
                        "target_y_m",
                        "target_z_m",
                        "target_qw",
                        "target_qx",
                        "target_qy",
                        "target_qz",
                        "gripper_closed",
                    ],
                    seed=attempt.randomization_seed,
                    lineage=lineage,
                    metadata={
                        "generation_plan": str(args.generation_plan),
                        "generation_attempt_id": attempt.attempt_id,
                        "source_episode_path": str(attempt.source_episode_path),
                    },
                )
                attempt_writer.start(source.task)

                pregrasp_gate = PoseConvergenceGate(
                    position_tolerance_m=(
                        plan.execution.pregrasp_position_tolerance_m
                    ),
                    rotation_tolerance_rad=(
                        plan.execution.pregrasp_rotation_tolerance_rad
                    ),
                    required_stable_ticks=plan.execution.pregrasp_stable_ticks,
                    timeout_ticks=pregrasp_timeout_ticks,
                )
                pregrasp_converged = False
                waypoint_index = 0
                control_step_index = 0
                max_tracking_position_error_m = 0.0
                max_tracking_rotation_error_rad = 0.0
                final_pregrasp_position_error_m: float | None = None
                final_pregrasp_rotation_error_rad: float | None = None

                while waypoint_index < trajectory.length:
                    if not simulation_app.is_running():
                        raise RuntimeError(
                            "Isaac application stopped during trajectory execution"
                        )
                    if control_step_index >= max_control_steps:
                        raise RuntimeError("trajectory exceeded the resolved control horizon")

                    target_pose = trajectory.target_ee_pose[waypoint_index]
                    waiting_for_pregrasp = (
                        waypoint_index == trajectory.close_waypoint_index
                        and not pregrasp_converged
                    )
                    desired_joints = trajectory_joint_targets(target_pose)
                    gripper_closed = (
                        False
                        if waiting_for_pregrasp
                        else bool(trajectory.gripper_closed[waypoint_index])
                    )
                    sequence = (
                        attempt_order[attempt.attempt_id] * max_control_steps
                        + control_step_index
                    )
                    command_timestamp_ns = time.time_ns()
                    command_receive_ns = time.monotonic_ns()
                    accepted = trajectory_controller.submit(
                        GelloCommand(
                            session_id="trajectory-batch",
                            sequence=sequence,
                            timestamp_ns=command_timestamp_ns,
                            joint_targets=desired_joints,
                            gripper_closed=gripper_closed,
                        )
                    )
                    if not accepted:
                        raise RuntimeError(
                            f"trajectory command sequence {sequence} was rejected"
                        )
                    safe_command = trajectory_controller.tick()
                    requested_finger = (
                        FRANKA_GRIPPER_CLOSED_M if gripper_closed else 0.04
                    )
                    applied_finger = (
                        FRANKA_GRIPPER_CLOSED_M
                        if safe_command.gripper_closed
                        else 0.04
                    )
                    requested_native = np.asarray(
                        (*desired_joints, requested_finger, requested_finger),
                        dtype=np.float32,
                    )
                    safe_native = np.asarray(
                        (*safe_command.joint_targets, applied_finger, applied_finger),
                        dtype=np.float32,
                    )
                    arm_positions = backend.current_joint_positions()
                    finger_positions = backend.current_gripper_positions()
                    arm_velocities = backend.current_joint_velocities()
                    finger_velocities = backend.current_gripper_velocities()
                    state = np.asarray(
                        (*arm_positions, *finger_positions, *arm_velocities, *finger_velocities),
                        dtype=np.float32,
                    )
                    ee_pose = (
                        robot.data.body_pose_w.torch[0, hand_body_index]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    object_pose = task_object.data.root_pose_w.torch[0].detach().cpu().numpy()
                    object_velocity = (
                        task_object.data.root_vel_w.torch[0].detach().cpu().numpy()
                    )
                    position_error, rotation_error = _pose_tracking_error(
                        ee_pose, target_pose
                    )
                    max_tracking_position_error_m = max(
                        max_tracking_position_error_m, position_error
                    )
                    max_tracking_rotation_error_rad = max(
                        max_tracking_rotation_error_rad, rotation_error
                    )

                    if control_step_index % dataset_control_decimation == 0:
                        for camera in record_cameras.values():
                            camera.update(dt=sim_dt)
                        images = {
                            name: camera.data.output["rgb"].torch[0].detach().cpu().numpy()
                            for name, camera in record_cameras.items()
                        }
                        attempt_writer.add_frame(
                            images=images,
                            state=state,
                            action=safe_native,
                            requested_action=requested_native,
                            safe_action=safe_native,
                            applied_action=safe_native,
                            operator_command=np.asarray(
                                (*target_pose, float(gripper_closed)), dtype=np.float32
                            ),
                            sim_time=control_step_index / args.control_hz,
                            control_sequence=sequence,
                            control_timestamp_ns=command_timestamp_ns,
                            control_receive_timestamp_ns=command_receive_ns,
                            capture_timestamp_ns=time.monotonic_ns(),
                            ee_pose=ee_pose,
                            target_ee_pose=target_pose,
                            object_pose=object_pose,
                            object_velocity=object_velocity,
                            extra_arrays={
                                "source_frame_index": np.asarray(
                                    [trajectory.source_frame_index[waypoint_index]],
                                    dtype=np.int64,
                                ),
                                "source_frame_coordinate": np.asarray(
                                    [trajectory.source_frame_coordinate[waypoint_index]],
                                    dtype=np.float32,
                                ),
                                "trajectory_phase_id": np.asarray(
                                    [trajectory.phase_id[waypoint_index]], dtype=np.int8
                                ),
                                "pregrasp_gate_active": np.asarray(
                                    [waiting_for_pregrasp], dtype=np.bool_
                                ),
                                "target_position_error_m": np.asarray(
                                    [position_error], dtype=np.float32
                                ),
                                "target_rotation_error_rad": np.asarray(
                                    [rotation_error], dtype=np.float32
                                ),
                            },
                        )
                    backend.set_command(
                        safe_command.joint_targets,
                        gripper_closed=safe_command.gripper_closed,
                    )
                    for _ in range(physics_steps_per_control):
                        sim.step()
                        robot.update(sim_dt)
                        task_object.update(sim_dt)
                        _pump_kit_frame(simulation_app)
                    cube_lift_evaluator.update(
                        object_z_m=float(task_object.data.root_pos_w.torch[0, 2].item()),
                        gripper_closed=safe_command.gripper_closed,
                    )

                    if waiting_for_pregrasp:
                        post_step_pose = (
                            robot.data.body_pose_w.torch[0, hand_body_index]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        gate_status, final_pregrasp_position_error_m, final_pregrasp_rotation_error_rad = (
                            pregrasp_gate.update(post_step_pose, target_pose)
                        )
                        if gate_status == "timed_out":
                            raise RuntimeError(
                                "pregrasp convergence timed out before gripper close"
                            )
                        if gate_status == "converged":
                            pregrasp_converged = True
                    else:
                        waypoint_index += 1
                    control_step_index += 1

                metrics = cube_lift_evaluator.result()
                metrics.update(
                    {
                        "control_steps": control_step_index,
                        "recorded_frames": attempt_writer.frame_count,
                        "pregrasp_converged": pregrasp_converged,
                        "pregrasp_wait_ticks": pregrasp_gate.elapsed_ticks,
                        "final_pregrasp_position_error_m": (
                            final_pregrasp_position_error_m
                        ),
                        "final_pregrasp_rotation_error_rad": (
                            final_pregrasp_rotation_error_rad
                        ),
                        "max_tracking_position_error_m": (
                            max_tracking_position_error_m
                        ),
                        "max_tracking_rotation_error_rad": (
                            max_tracking_rotation_error_rad
                        ),
                    }
                )
                succeeded = bool(metrics["success"])
                episode_path = attempt_writer.finish(
                    operator_success=None,
                    auto_success=succeeded,
                    episode_metrics={
                        key: value for key, value in metrics.items() if key != "success"
                    },
                    evaluator_id="isaac_cube_lift_retarget_v2",
                    evaluator_version="2",
                )
                result = AttemptResultV1(
                    attempt_id=attempt.attempt_id,
                    status="succeeded" if succeeded else "failed",
                    episode_path=episode_path,
                    error=None if succeeded else "automatic_criterion_not_met",
                    metrics=metrics,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if attempt_writer is not None and attempt_writer.is_recording:
                    if attempt_writer.frame_count:
                        metrics = cube_lift_evaluator.result()
                        episode_path = attempt_writer.finish(
                            operator_success=False,
                            auto_success=False,
                            episode_metrics={
                                **{key: value for key, value in metrics.items() if key != "success"},
                                "runtime_error": str(exc),
                            },
                            evaluator_id="isaac_cube_lift_retarget_v2",
                            evaluator_version="2",
                        )
                    else:
                        attempt_writer.discard()
                result = AttemptResultV1(
                    attempt_id=attempt.attempt_id,
                    status="failed",
                    episode_path=episode_path,
                    error=f"{type(exc).__name__}: {exc}",
                    metrics=metrics,
                )
            report = record_attempt_result(report, result)
            write_json_atomic(report_path, report)
            print(
                "ISAAC_TRAJECTORY_ATTEMPT " + result.model_dump_json(),
                file=sys.__stderr__,
                flush=True,
            )
            if not simulation_app.is_running():
                raise RuntimeError("Isaac application stopped; generation report is resumable")

        successes = sum(result.status == "succeeded" for result in report.results)
        gate_passed = (
            successes == report.requested_attempts
            if plan.nominal_only
            else successes * 2 >= report.requested_attempts
        )
        summary = {
            "status": report.status,
            "gate_passed": gate_passed,
            "gate": "all nominal attempts" if plan.nominal_only else "at least 50 percent",
            "requested_attempts": report.requested_attempts,
            "completed_attempts": len(report.results),
            "successful_attempts": successes,
            "failed_attempts": len(report.results) - successes,
            "report_path": str(report_path),
        }
        print(
            "ISAAC_TRAJECTORY_BATCH " + json.dumps(summary, sort_keys=True),
            file=sys.__stderr__,
            flush=True,
        )
        return gate_passed

    if args.control_source == "openpi":
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        from .openpi_policy import (
            GripperCommandFilter,
            decode_policy_action,
            encode_policy_state,
            encode_videos,
            validate_action_chunk,
        )
        from robot_data_generation.evaluation_plan import (
            evaluation_plan_sha256,
            load_evaluation_plan,
        )

        if writer is None or cube_lift_evaluator is None or task_object is None:
            raise RuntimeError("OpenPI rollout requires writer, cube evaluator, and task object")
        client = WebsocketClientPolicy(args.policy_host, args.policy_port)
        policy_metadata = client.get_server_metadata()
        if policy_metadata.get("action_contract") != "joint7_absolute_plus_gripper_closed_fraction":
            raise RuntimeError(f"unexpected policy action contract: {policy_metadata!r}")
        if int(policy_metadata.get("action_horizon", -1)) < args.replan_steps:
            raise RuntimeError(
                f"policy horizon {policy_metadata.get('action_horizon')} is shorter than "
                f"replan_steps {args.replan_steps}"
            )
        served_checkpoint = Path(str(policy_metadata.get("checkpoint", ""))).resolve()
        if served_checkpoint != args.policy_checkpoint:
            raise RuntimeError(
                f"policy server checkpoint {served_checkpoint} does not match "
                f"requested {args.policy_checkpoint}"
            )

        policy_controller = GelloCommandController(
            initial_joint_positions,
            backend.joint_position_limits(),
            control_hz=args.dataset_fps,
            max_joint_speed=args.max_joint_speed,
            watchdog_timeout=args.gello_timeout,
        )
        physics_steps_per_policy = int(round(args.physics_hz / args.dataset_fps))
        gripper_filter = GripperCommandFilter(
            mode=args.gripper_filter,
            close_threshold=args.gripper_close_threshold,
            open_threshold=args.gripper_open_threshold,
            min_closed_frames=args.gripper_min_closed_frames,
        )
        gripper_filter_config = {
            **gripper_filter.configuration(),
            "min_closed_seconds": args.gripper_min_closed_seconds,
        }
        output_root = Path(args.data_dir).expanduser().resolve()
        batch_report_path = output_root / "batch_report.json"
        batch_csv_path = output_root / "rollouts.csv"
        if batch_report_path.exists():
            raise RuntimeError(
                f"OpenPI batch report already exists; choose a new --data-dir: {batch_report_path}"
            )
        evaluation_plan = (
            load_evaluation_plan(args.evaluation_plan) if args.evaluation_plan else None
        )
        evaluation_digest = (
            evaluation_plan_sha256(evaluation_plan) if evaluation_plan else None
        )
        evaluation_split = _evaluation_position_split(evaluation_plan)
        conditions = (
            list(evaluation_plan.conditions) if evaluation_plan else [None] * args.rollout_count
        )
        if args.evaluation_limit is not None:
            conditions = conditions[: args.evaluation_limit]
        requested_rollouts = len(conditions)
        rollout_reports: list[dict[str, Any]] = []

        def write_batch_report(*, completed: bool) -> None:
            summary = _rollout_stage_summary(rollout_reports)
            payload = {
                "status": "completed" if completed else "running",
                "scene": args.scene,
                "fixed_training_distribution": True,
                "evaluation_split": evaluation_split,
                "heldout_evaluation": evaluation_split == "heldout",
                "generalization_claimed": False,
                "evaluation_plan": str(args.evaluation_plan) if args.evaluation_plan else None,
                "evaluation_plan_sha256": evaluation_digest,
                "requested_rollouts": requested_rollouts,
                "completed_rollouts": len(rollout_reports),
                "policy_config": args.policy_config,
                "policy_checkpoint": str(args.policy_checkpoint),
                "policy_server_metadata": policy_metadata,
                "control_fps": args.dataset_fps,
                "replan_steps": args.replan_steps,
                "inference_steps": args.inference_steps,
                "gripper_filter": gripper_filter_config,
                "full_horizon": args.full_horizon,
                "summary": summary,
                "rollouts": rollout_reports,
            }
            batch_report_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            fieldnames = [
                "rollout_index",
                "condition_id",
                "object_x_offset_m",
                "object_y_offset_m",
                "episode_path",
                "success",
                "grasp",
                "lift",
                "hold",
                "release",
                "frames",
                "max_lift_m",
                "hold_duration_s",
                "raw_gripper_switches",
                "applied_gripper_switches",
                "base_video",
                "top_video",
                "side_video",
                "wrist_video",
            ]
            with batch_csv_path.open("w", newline="", encoding="utf-8") as stream:
                table = csv.DictWriter(stream, fieldnames=fieldnames)
                table.writeheader()
                for report in rollout_reports:
                    metrics = report["metrics"]
                    videos = report["videos"]
                    table.writerow(
                        {
                            "rollout_index": report["rollout_index"],
                            "condition_id": report["condition_id"],
                            "object_x_offset_m": report["object_xy_offset_m"][0],
                            "object_y_offset_m": report["object_xy_offset_m"][1],
                            "episode_path": report["episode_path"],
                            "success": report["success"],
                            "grasp": metrics["grasp"],
                            "lift": metrics["lift"],
                            "hold": metrics["hold"],
                            "release": metrics["release"],
                            "frames": report["frames"],
                            "max_lift_m": metrics["max_lift_m"],
                            "hold_duration_s": metrics["hold_duration_s"],
                            "raw_gripper_switches": report["gripper_diagnostics"]["raw_switches"],
                            "applied_gripper_switches": report["gripper_diagnostics"]["applied_switches"],
                            **{f"{name}_video": videos[name] for name in CAMERA_NAMES},
                        }
                    )

        for rollout_index, condition in enumerate(conditions):
            object_offset = (
                tuple(condition.object_xy_offset_m) if condition is not None else (0.0, 0.0)
            )
            object_start = (
                float(scene_geometry["object_start_m"][0]) + object_offset[0],
                float(scene_geometry["object_start_m"][1]) + object_offset[1],
                float(scene_geometry["object_start_m"][2]),
            )
            condition_id = (
                condition.condition_id if condition is not None else f"nominal-{rollout_index:03d}"
            )
            condition_seed = condition.rollout_seed if condition is not None else args.seed + rollout_index
            np.random.seed(condition_seed % (2**32))
            torch.manual_seed(condition_seed)
            reset_task_assets(object_start)
            policy_controller.reset_targets(initial_joint_positions)
            cube_lift_evaluator.reset()
            gripper_filter.reset()
            action_plan: list[np.ndarray] = []
            inference_samples: list[np.ndarray] = []
            raw_gripper_commands: list[bool] = []
            applied_gripper_commands: list[bool] = []
            replan_id = -1
            stable_success_frames = 0
            episode_index = writer.start(scene_task)

            for frame_index in range(args.max_episode_frames):
                if not simulation_app.is_running():
                    raise RuntimeError("Isaac application stopped during policy rollout")
                for camera in record_cameras.values():
                    camera.update(dt=sim_dt)
                images = {
                    name: camera.data.output["rgb"].torch[0].detach().cpu().numpy()
                    for name, camera in record_cameras.items()
                }
                capture_timestamp_ns = time.monotonic_ns()
                arm_positions = backend.current_joint_positions()
                finger_positions = backend.current_gripper_positions()
                arm_velocities = backend.current_joint_velocities()
                finger_velocities = backend.current_gripper_velocities()
                state = np.asarray(
                    (*arm_positions, *finger_positions, *arm_velocities, *finger_velocities),
                    dtype=np.float32,
                )
                if not action_plan:
                    command_timestamp_ns = time.time_ns()
                    inference_started = time.perf_counter()
                    result = client.infer(
                        {
                            "observation/image": images["base"],
                            "observation/wrist_image": images["wrist"],
                            "observation/state": encode_policy_state(state),
                            "prompt": scene_task,
                        }
                    )
                    client_total_ms = (time.perf_counter() - inference_started) * 1000.0
                    command_receive_ns = time.monotonic_ns()
                    action_plan.extend(
                        validate_action_chunk(result["actions"], replan_steps=args.replan_steps)
                    )
                    replan_id += 1
                    server_ms = float(result.get("server_timing", {}).get("infer_ms", client_total_ms))
                    model_ms = float(result.get("policy_timing", {}).get("infer_ms", server_ms))
                    inference_samples.append(
                        np.asarray([client_total_ms, server_ms, model_ms], dtype=np.float32)
                    )

                model_action = action_plan.pop(0)
                requested_joints, _, requested_native = decode_policy_action(model_action)
                raw_gripper_closed = float(model_action[7]) >= 0.5
                gripper_closed = gripper_filter.update(float(model_action[7]))
                raw_gripper_commands.append(raw_gripper_closed)
                applied_gripper_commands.append(gripper_closed)
                global_sequence = rollout_index * args.max_episode_frames + frame_index
                accepted = policy_controller.submit(
                    GelloCommand(
                        session_id="openpi-policy",
                        sequence=global_sequence,
                        timestamp_ns=command_timestamp_ns,
                        joint_targets=tuple(float(value) for value in requested_joints),
                        gripper_closed=gripper_closed,
                    )
                )
                if not accepted:
                    raise RuntimeError(f"policy command sequence {global_sequence} was rejected")
                safe_command = policy_controller.tick()
                finger_target = FRANKA_GRIPPER_CLOSED_M if safe_command.gripper_closed else 0.04
                safe_native = np.asarray(
                    (*safe_command.joint_targets, finger_target, finger_target), dtype=np.float32
                )
                ee_pose = robot.data.body_pose_w.torch[0, hand_body_index].detach().cpu().numpy()
                object_pose = task_object.data.root_pose_w.torch[0].detach().cpu().numpy()
                object_velocity = task_object.data.root_vel_w.torch[0].detach().cpu().numpy()
                latest_timing = inference_samples[-1]
                writer.add_frame(
                    images=images,
                    state=state,
                    action=safe_native,
                    requested_action=requested_native,
                    safe_action=safe_native,
                    applied_action=safe_native,
                    operator_command=model_action,
                    sim_time=frame_index / args.dataset_fps,
                    control_sequence=global_sequence,
                    control_timestamp_ns=command_timestamp_ns,
                    control_receive_timestamp_ns=command_receive_ns,
                    capture_timestamp_ns=capture_timestamp_ns,
                    ee_pose=ee_pose,
                    object_pose=object_pose,
                    object_velocity=object_velocity,
                    extra_arrays={
                        "model_action_raw": model_action,
                        "policy_replan_id": np.asarray([replan_id], dtype=np.int64),
                        "policy_inference_timing_ms": latest_timing,
                        "policy_gripper_raw_closed": np.asarray(
                            [raw_gripper_closed], dtype=np.bool_
                        ),
                        "policy_gripper_filtered_closed": np.asarray(
                            [gripper_closed], dtype=np.bool_
                        ),
                        "evaluation_object_xy_offset_m": np.asarray(
                            object_offset, dtype=np.float32
                        ),
                        "evaluation_condition_seed": np.asarray(
                            [condition_seed], dtype=np.int64
                        ),
                    },
                )
                backend.set_command(
                    safe_command.joint_targets,
                    gripper_closed=safe_command.gripper_closed,
                )
                for _ in range(physics_steps_per_policy):
                    sim.step()
                    robot.update(sim_dt)
                    task_object.update(sim_dt)
                    _pump_kit_frame(simulation_app)

                post_object_z = float(task_object.data.root_pos_w.torch[0, 2].item())
                cube_lift_evaluator.update(
                    object_z_m=post_object_z,
                    gripper_closed=safe_command.gripper_closed,
                )
                evaluation = cube_lift_evaluator.result()
                stable_success_frames = (
                    stable_success_frames + 1 if evaluation["success"] else 0
                )
                if stable_success_frames >= args.success_settle_frames and not args.full_horizon:
                    break

            metrics = cube_lift_evaluator.result()
            metrics["stable_success_frames"] = stable_success_frames
            metrics["required_stable_success_frames"] = args.success_settle_frames
            metrics["rollout_index"] = rollout_index
            frame_count = writer.frame_count
            episode_path = writer.finish(
                operator_success=None,
                auto_success=bool(metrics["success"]),
                episode_metrics={key: value for key, value in metrics.items() if key != "success"},
                evaluator_id="isaac_cube_lift_openpi_v1",
                evaluator_version="1",
            )
            videos = encode_videos(episode_path, list(CAMERA_NAMES), int(round(args.dataset_fps)))
            timing = np.stack(inference_samples) if inference_samples else np.empty((0, 3))
            raw_switches = sum(
                current != previous
                for previous, current in zip(raw_gripper_commands, raw_gripper_commands[1:])
            )
            applied_switches = sum(
                current != previous
                for previous, current in zip(
                    applied_gripper_commands, applied_gripper_commands[1:]
                )
            )
            report = {
                "rollout_index": rollout_index,
                "condition_id": condition_id,
                "condition_seed": condition_seed,
                "object_xy_offset_m": list(object_offset),
                "object_start_position_m": list(object_start),
                "episode_index": episode_index,
                "episode_path": str(episode_path),
                "success": bool(metrics["success"]),
                "frames": frame_count,
                "videos": videos,
                "metrics": metrics,
                "gripper_filter": gripper_filter_config,
                "gripper_diagnostics": {
                    "raw_switches": raw_switches,
                    "applied_switches": applied_switches,
                    "raw_closed_fraction": float(np.mean(raw_gripper_commands)),
                    "applied_closed_fraction": float(np.mean(applied_gripper_commands)),
                    "first_raw_close_frame": next(
                        (index for index, value in enumerate(raw_gripper_commands) if value),
                        None,
                    ),
                    "first_applied_close_frame": next(
                        (index for index, value in enumerate(applied_gripper_commands) if value),
                        None,
                    ),
                },
                "inference_timing_ms": (
                    {
                        "calls": len(timing),
                        "client_total_mean": float(timing[:, 0].mean()),
                        "client_total_max": float(timing[:, 0].max()),
                        "server_mean": float(timing[:, 1].mean()),
                        "model_mean": float(timing[:, 2].mean()),
                    }
                    if len(timing)
                    else {"calls": 0}
                ),
            }
            (episode_path / "rollout_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rollout_reports.append(report)
            write_batch_report(completed=False)
            print(
                "ISAAC_OPENPI_ROLLOUT " + json.dumps(report, sort_keys=True),
                file=sys.__stderr__,
                flush=True,
            )

        write_batch_report(completed=True)
        final_summary = _rollout_stage_summary(rollout_reports)
        print(
            "ISAAC_OPENPI_BATCH " + json.dumps(final_summary, sort_keys=True),
            file=sys.__stderr__,
            flush=True,
        )
        return True

    web_server = GelloWebServer(
        controller,
        telemetry,
        recording,
        previews,
        views,
        keyboard,
        control_source=args.control_source,
        host=args.host,
        port=args.port,
        stream_host=args.stream_host,
        signaling_port=args.signaling_port,
        media_port=args.media_port,
        stream_fps=args.stream_fps,
        control_hz=args.control_hz,
    )
    physics_step = 0
    control_step = 0
    control_tick_ms: deque[float] = deque(maxlen=max(20, int(round(args.control_hz * 5))))
    control_tick_times: deque[float] = deque(maxlen=max(20, int(round(args.control_hz * 5))))
    safe_command = controller.tick(now=0.0) if args.smoke_steps else controller.tick()
    keyboard_sample = keyboard.advance()
    last_keyboard_command: tuple[str, int] | None = None
    target_at_watchdog: tuple[float, ...] | None = None
    initial_tcp_position = tuple(
        float(value)
        for value in robot.data.body_pos_w.torch[0, hand_body_index].detach().cpu().tolist()
    )
    services = (
        _running_control_services(web_server, previews)
        if args.smoke_steps == 0
        else nullcontext()
    )
    services_entered = False
    try:
        services.__enter__()
        services_entered = True
        if args.smoke_steps == 0:
            print(
                "ISAAC_CONTROL_READY "
                + json.dumps(
                    {
                        "control_source": args.control_source,
                        "host": args.host,
                        "port": args.port,
                    },
                    sort_keys=True,
                ),
                file=sys.__stderr__,
                flush=True,
            )
        while simulation_app.is_running():
            simulated_time = physics_step * sim_dt
            if physics_step % control_decimation == 0:
                control_started = time.perf_counter()
                if writer is not None:
                    for recording_command, task in recording.drain():
                        try:
                            if recording_command == "start":
                                if writer.is_recording:
                                    raise RuntimeError("an episode is already recording")
                                reset_task_assets()
                                controller.reset_targets(initial_joint_positions)
                                keyboard.reset()
                                diff_ik_controller.reset()
                                if cube_lift_evaluator is not None:
                                    cube_lift_evaluator.reset()
                                episode_index = writer.start(task)
                                recording.update(
                                    recording=True,
                                    episode=episode_index,
                                    frames=0,
                                    task=task,
                                    error=None,
                                    last_event=f"Episode {episode_index} started",
                                )
                            elif recording_command in {"success", "failure"}:
                                evaluation = (
                                    cube_lift_evaluator.result()
                                    if cube_lift_evaluator is not None
                                    else None
                                )
                                path = writer.finish(
                                    operator_success=recording_command == "success",
                                    episode_metrics=(
                                        {key: value for key, value in evaluation.items() if key != "success"}
                                        if evaluation is not None
                                        else {"automatic_outcome": "unavailable"}
                                    ),
                                    auto_success=(
                                        bool(evaluation["success"])
                                        if evaluation is not None
                                        else None
                                    ),
                                    evaluator_id=(
                                        "isaac_cube_lift_v1"
                                        if evaluation is not None
                                        else "task_evaluator"
                                    ),
                                )
                                recording.update(
                                    recording=False,
                                    episode=None,
                                    frames=0,
                                    task="",
                                    last_saved_path=str(path),
                                    error=None,
                                    last_event=f"Episode saved: {path}",
                                )
                            elif recording_command == "discard":
                                writer.discard()
                                recording.update(
                                    recording=False,
                                    episode=None,
                                    frames=0,
                                    task="",
                                    error=None,
                                    last_event="Episode discarded",
                                )
                            elif recording_command == "reset":
                                if writer.is_recording:
                                    raise RuntimeError("finish or discard the active episode before reset")
                                reset_task_assets()
                                controller.reset_targets(initial_joint_positions)
                                keyboard.reset()
                                diff_ik_controller.reset()
                                recording.update(error=None, last_event="Environment reset")
                        except (OSError, RuntimeError, ValueError) as exc:
                            recording.update(error=str(exc), last_event=f"ERROR: {exc}")
                pending_views = views.drain()
                if pending_views:
                    view_name = pending_views[-1]
                    try:
                        camera_path = _set_stream_view(
                            view_name,
                            camera_prim_paths,
                            camera_target=stream_camera_target,
                        )
                    except (RuntimeError, ValueError) as exc:
                        telemetry.update(view_error=str(exc))
                    else:
                        views.activate(view_name, camera_path)
                        telemetry.update(viewport_camera=camera_path, view_error=None)
                if args.smoke_steps and control_step == 0:
                    if args.control_source == "keyboard":
                        keyboard.submit(
                            {
                                "type": "keyboard",
                                "version": 1,
                                "session_id": "keyboard-smoke",
                                "sequence": 0,
                                "timestamp_ns": 1,
                                "pressed": ["KeyW"],
                                "gripper_closed": True,
                            },
                            received_at=simulated_time,
                        )
                    else:
                        smoke_targets = list(initial_joint_positions)
                        smoke_targets[0] += 0.03
                        controller.submit(
                            GelloCommand(
                                sequence=0,
                                timestamp_ns=1,
                                joint_targets=smoke_targets,
                                gripper_closed=True,
                            ),
                            received_at=simulated_time,
                        )
                if args.control_source == "keyboard":
                    keyboard_sample = keyboard.advance(
                        now=simulated_time if args.smoke_steps else None
                    )
                    keyboard_command = (
                        (keyboard_sample.session_id, keyboard_sample.sequence)
                        if keyboard_sample.session_id is not None
                        and keyboard_sample.sequence is not None
                        else None
                    )
                    if (
                        not keyboard_sample.watchdog_holding
                        and keyboard_command is not None
                        and keyboard_command != last_keyboard_command
                    ):
                        requested_joint_targets = keyboard_joint_targets(
                            keyboard_sample, safe_command.joint_targets
                        )
                        controller.submit(
                            GelloCommand(
                                session_id="browser-keyboard-ik",
                                # The browser sequence restarts after a reload. Use the
                                # monotonic simulator tick for this internal controller hop.
                                sequence=control_step,
                                timestamp_ns=keyboard_sample.source_timestamp_ns,
                                joint_targets=requested_joint_targets,
                                gripper_closed=keyboard_sample.gripper_closed,
                            ),
                            received_at=simulated_time if args.smoke_steps else None,
                        )
                        last_keyboard_command = keyboard_command
                safe_command = controller.tick(now=simulated_time) if args.smoke_steps else controller.tick()
                recording_active = writer is not None and writer.is_recording
                dataset_due = (
                    recording_active
                    and control_step % args.dataset_control_decimation == 0
                )
                preview_due = (
                    bool(record_cameras)
                    and control_step % args.preview_control_decimation == 0
                )
                images = None
                capture_timestamp_ns = None
                if dataset_due or preview_due:
                    assert record_cameras
                    for camera in record_cameras.values():
                        camera.update(dt=sim_dt)
                    images = {
                        name: camera.data.output["rgb"].torch[0].detach().cpu().numpy()
                        for name, camera in record_cameras.items()
                    }
                    capture_timestamp_ns = time.monotonic_ns()
                if preview_due and images is not None:
                    previews.submit(
                        images,
                        frame_index=control_step,
                        captured_at_ns=time.monotonic_ns(),
                    )
                if dataset_due:
                    assert writer is not None and images is not None
                    arm_positions = backend.current_joint_positions()
                    finger_positions_before_action = backend.current_gripper_positions()
                    arm_velocities = backend.current_joint_velocities()
                    finger_velocities = backend.current_gripper_velocities()
                    finger_target = FRANKA_GRIPPER_CLOSED_M if safe_command.gripper_closed else 0.04
                    state = np.asarray(
                        (*arm_positions, *finger_positions_before_action, *arm_velocities, *finger_velocities),
                        dtype=np.float32,
                    )
                    action = np.asarray((*safe_command.joint_targets, finger_target, finger_target), dtype=np.float32)
                    requested_finger_target = FRANKA_GRIPPER_CLOSED_M if safe_command.gripper_closed else 0.04
                    requested_action = np.asarray(
                        (*safe_command.requested_joint_targets, requested_finger_target, requested_finger_target),
                        dtype=np.float32,
                    )
                    operator_command = np.asarray(
                        (
                            (*keyboard_sample.delta_pose, float(keyboard_sample.gripper_closed))
                            if args.control_source == "keyboard"
                            else (*safe_command.requested_joint_targets, float(safe_command.gripper_closed))
                        ),
                        dtype=np.float32,
                    )
                    ee_pose = None
                    object_pose = None
                    object_velocity = None
                    if cube_lift_evaluator is not None and task_object is not None:
                        ee_pose = (
                            robot.data.body_pose_w.torch[0, hand_body_index]
                            .detach().cpu().numpy()
                        )
                        object_pose = task_object.data.root_pose_w.torch[0].detach().cpu().numpy()
                        object_velocity = task_object.data.root_vel_w.torch[0].detach().cpu().numpy()
                        cube_lift_evaluator.update(
                            object_z_m=float(object_pose[2]),
                            gripper_closed=safe_command.gripper_closed,
                        )
                        telemetry.update(task_evaluator=cube_lift_evaluator.result())
                    backend.set_command(
                        safe_command.joint_targets,
                        gripper_closed=safe_command.gripper_closed,
                    )
                    writer.add_frame(
                        images=images,
                        state=state,
                        action=action,
                        requested_action=requested_action,
                        safe_action=action,
                        applied_action=action,
                        operator_command=operator_command,
                        sim_time=simulated_time,
                        control_sequence=safe_command.sequence if safe_command.sequence is not None else -1,
                        control_timestamp_ns=(
                            safe_command.source_timestamp_ns
                            if safe_command.source_timestamp_ns is not None
                            else -1
                        ),
                        control_receive_timestamp_ns=(
                            safe_command.receive_timestamp_ns
                            if safe_command.receive_timestamp_ns is not None
                            else -1
                        ),
                        capture_timestamp_ns=capture_timestamp_ns,
                        ee_pose=ee_pose,
                        object_pose=object_pose,
                        object_velocity=object_velocity,
                    )
                    recording.update(frames=writer.frame_count)
                if not dataset_due:
                    backend.set_command(
                        safe_command.joint_targets,
                        gripper_closed=safe_command.gripper_closed,
                    )
                if safe_command.watchdog_holding and target_at_watchdog is None:
                    target_at_watchdog = safe_command.joint_targets
                control_finished = time.perf_counter()
                elapsed_control_ms = (control_finished - control_started) * 1000.0
                control_tick_ms.append(elapsed_control_ms)
                control_tick_times.append(control_finished)
                ordered_tick_ms = sorted(control_tick_ms)
                p95_index = min(
                    len(ordered_tick_ms) - 1,
                    int(math.ceil(0.95 * len(ordered_tick_ms))) - 1,
                )
                telemetry.update(
                    control_tick_ms=round(elapsed_control_ms, 2),
                    control_tick_p95_ms=round(ordered_tick_ms[p95_index], 2),
                    control_actual_hz=(
                        round(
                            (len(control_tick_times) - 1)
                            / (control_tick_times[-1] - control_tick_times[0]),
                            1,
                        )
                        if len(control_tick_times) >= 2
                        and control_tick_times[-1] > control_tick_times[0]
                        else 0.0
                    ),
                    dataset_fps=args.dataset_fps,
                )
                control_step += 1

            sim.step()
            robot.update(sim_dt)
            if task_object is not None:
                task_object.update(sim_dt)
            if args.smoke_steps == 0 and physics_step % stream_decimation == 0:
                _pump_kit_frame(simulation_app)
            actual_joint_positions = backend.current_joint_positions()
            finger_positions = backend.current_gripper_positions()
            tracking_error = max(
                abs(actual - target) for actual, target in zip(actual_joint_positions, safe_command.joint_targets)
            )
            tcp_z_m = float(robot.data.body_pos_w.torch[0, hand_body_index, 2].item())
            object_position_m = (
                task_object.data.root_pos_w.torch[0].detach().cpu().tolist()
                if task_object is not None
                else None
            )
            telemetry.update(
                actual_joint_positions=list(actual_joint_positions),
                finger_positions=list(finger_positions),
                tracking_error_max_rad=round(tracking_error, 6),
                tcp_z_m=round(tcp_z_m, 4),
                tcp_surface_clearance_m=(
                    round(tcp_z_m - work_surface_top_m, 4)
                    if work_surface_top_m is not None
                    else None
                ),
                object_position_m=object_position_m,
                physics_step=physics_step,
            )
            physics_step += 1
            if args.smoke_steps and physics_step >= args.smoke_steps:
                break
    finally:
        if writer is not None and writer.is_recording:
            writer.discard()
        if services_entered:
            services.__exit__(*sys.exc_info())

    if args.smoke_steps:
        actual_joint_positions = backend.current_joint_positions()
        finger_positions = backend.current_gripper_positions()
        final_tracking_error = max(
            abs(actual - target) for actual, target in zip(actual_joint_positions, safe_command.joint_targets)
        )
        finger_error = max(abs(value - FRANKA_GRIPPER_CLOSED_M) for value in finger_positions)
        controller_status = controller.status(now=physics_step * sim_dt)
        target_held = target_at_watchdog == safe_command.joint_targets
        final_tcp_position = tuple(
            float(value)
            for value in robot.data.body_pos_w.torch[0, hand_body_index].detach().cpu().tolist()
        )
        tcp_x_gain = final_tcp_position[0] - initial_tcp_position[0]
        motion_passed = (
            tcp_x_gain >= 0.005
            if args.control_source == "keyboard"
            else actual_joint_positions[0] - initial_joint_positions[0] >= 0.015
        )
        report = {
            "control_source": args.control_source,
            "passed": (
                motion_passed
                and final_tracking_error <= 0.02
                and finger_error <= 0.005
                and controller_status["watchdog_holds"] == 1
                and target_held
            ),
            "physics_steps": physics_step,
            "initial_j1_rad": initial_joint_positions[0],
            "actual_j1_rad": actual_joint_positions[0],
            "j1_gain_rad": actual_joint_positions[0] - initial_joint_positions[0],
            "tcp_x_gain_m": tcp_x_gain,
            "max_tracking_error_rad": final_tracking_error,
            "finger_positions_m": list(finger_positions),
            "finger_error_m": finger_error,
            "watchdog_holds": controller_status["watchdog_holds"],
            "target_held_after_watchdog": target_held,
        }
        print(
            f"GELLO_ISAAC_SMOKE {json.dumps(report, sort_keys=True)}",
            file=sys.__stderr__,
            flush=True,
        )
        return bool(report["passed"])
    return True


def main() -> None:
    args, simulation_app = _parse_args()
    try:
        passed = _run_simulator(args, simulation_app)
    except BaseException:
        traceback.print_exc(file=sys.__stderr__)
        simulation_app.close(exit_code=1)
        raise
    simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
