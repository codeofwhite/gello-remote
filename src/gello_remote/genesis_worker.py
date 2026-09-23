"""Shared browser teleoperation and synchronized Franka data collection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import cv2
import franka_only
import gello_remote
import numpy as np
import websockets
from gello_remote.control import (
    DualGelloCommandController,
    GelloCommandController,
    GelloSessionGate,
    parse_arm_session_id,
)
from gello_remote.protocol import GelloCommandV1

from franka_only.control import BrowserKeyboardControlSource
from franka_only.camera_profiles import load_camera_profile
from robot_data_contracts import DualFrankaEpisodeWriter, FrankaEpisodeWriter
from franka_only.scene_registry import (
    SCENE_IDS,
    apply_scene_overrides,
    is_cloth_scene,
    scene_catalog,
    scene_parameter_schema,
)
from franka_only.scenes import build_scene, validate_control_timing


PROJECT_ROOT = Path(franka_only.__file__).resolve().parents[1]
CONTROL_SOURCES = ("keyboard", "gello")
UI_DIR = Path(__file__).parent / "web" / "genesis"
GELLO_WEB_DIR = Path(gello_remote.__file__).resolve().parent / "web"
HTML_PAGES = {
    "keyboard": (UI_DIR / "keyboard_ui.html").read_bytes(),
    "gello": (UI_DIR / "gello_ui.html").read_bytes(),
}
STATIC_ASSETS = {
    "/collector_ui.css": ("text/css; charset=utf-8", UI_DIR / "collector_ui.css"),
    "/collector_common.mjs": ("text/javascript; charset=utf-8", UI_DIR / "collector_common.mjs"),
    "/keyboard_ui.mjs": ("text/javascript; charset=utf-8", UI_DIR / "keyboard_ui.mjs"),
    "/gello_ui.mjs": ("text/javascript; charset=utf-8", UI_DIR / "gello_ui.mjs"),
    "/gello_web_serial.mjs": (
        "text/javascript; charset=utf-8",
        GELLO_WEB_DIR / "gello_web_serial.mjs",
    ),
}


def parse_gello_message(data: dict) -> GelloCommandV1:
    """Validate one versioned command using the canonical gello-remote contract."""
    return GelloCommandV1.model_validate(data)


def parse_keyboard_event(data: dict, *, last_sequence: int | None) -> tuple[int, int]:
    """Validate browser keyboard ordering and its wall-clock source timestamp."""
    sequence = data.get("sequence")
    timestamp_ns = data.get("timestamp_ns")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("Keyboard sequence must be a non-negative integer")
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
        raise ValueError("Keyboard timestamp_ns must be a positive integer")
    if last_sequence is not None and sequence <= last_sequence:
        raise ValueError(
            f"Keyboard sequence must increase within one connection: {sequence} <= {last_sequence}"
        )
    return sequence, timestamp_ns


def control_message_allowed(control_source: str, message_type: str) -> bool:
    """Return whether a client message is valid for the startup-locked mode."""
    if message_type in {"command", "reconfigure"}:
        return True
    if control_source == "keyboard":
        return message_type in {"keydown", "keyup"}
    if control_source == "gello":
        return message_type == "gello"
    raise ValueError(f"Unknown control source: {control_source}")


def parse_scene_params(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Scene parameter must use key=value syntax: {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key or key in overrides:
            raise ValueError(f"Scene parameter key is empty or repeated: {key!r}")
        try:
            overrides[key] = float(raw)
        except ValueError as error:
            raise ValueError(f"Scene parameter {key!r} must be numeric") from error
    return overrides


def build_restart_argv(
    args: argparse.Namespace,
    scene_id: str,
    overrides: dict[str, float],
) -> list[str]:
    """Build a shell-free, whitelist-only argv for apply-and-restart."""
    argv = [sys.executable, "-m", args.launch_module]
    if args.launch_module == "gello_remote.genesis_worker":
        argv += ["--control-source", args.control_source]
    argv += [
        "--scene",
        scene_id,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--control-fps",
        str(args.control_fps),
        "--physics-steps-per-control",
        str(args.physics_steps_per_control),
        "--sim-dt",
        str(args.sim_dt),
        "--substeps",
        str(args.substeps),
        "--seed",
        str(args.seed),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--stream-jpeg-quality",
        str(args.stream_jpeg_quality),
        "--position-step",
        str(args.position_step),
        "--rotation-step",
        str(args.rotation_step),
        "--gello-timeout",
        str(args.gello_timeout),
        "--logging-level",
        args.logging_level,
    ]
    if args.data_dir_explicit:
        argv += ["--data-dir", str(args.data_dir)]
    if getattr(args, "camera_profile_file", None) is not None:
        argv += ["--camera-profile-file", str(args.camera_profile_file)]
    for key in sorted(overrides):
        argv += ["--scene-param", f"{key}={overrides[key]:.12g}"]
    return argv


class TeleopServer:
    def __init__(self, args: argparse.Namespace):
        import genesis as gs

        self.args = args
        self.clients: set = set()
        self.control = BrowserKeyboardControlSource(args.position_step, args.rotation_step)
        self.sim_time = 0.0
        self.current_episode: int | None = None
        self.current_task = ""
        self.stream_fps = 0.0
        self._fps_window_started = time.perf_counter()
        self._fps_window_ticks = 0
        self.gello_sequence: int | None = None
        self.gello_session_id: str | None = None
        self.gello_last_sequences: dict[str, int] = {}
        self.gello_timestamp_ns: int | None = None
        self.gello_received_at: float | None = None
        self.gello_joint_targets: np.ndarray | None = None
        self.gello_gripper_closed = False
        self.gello_watchdog_holds = 0
        self.gello_session_gate = GelloSessionGate(args.gello_timeout)
        self._watchdog_holding = False
        self.keyboard_sequence: int | None = None
        self.keyboard_timestamp_ns: int | None = None
        self.keyboard_receive_timestamp_ns: int | None = None
        gs.init(backend=gs.gpu, precision="32", logging_level=args.logging_level, seed=args.seed)
        self.task_scene = build_scene(args.scene, args, args.scene_overrides)
        # Arm topology comes from the scene. Single-arm scenes expose no ``arm_ids``;
        # a dual scene must expose exactly ("left", "right"). Nothing is inferred.
        scene_arm_ids = tuple(getattr(self.task_scene, "arm_ids", ()))
        self.dual_arm = scene_arm_ids == ("left", "right")
        self.arm_ids = scene_arm_ids if scene_arm_ids else ("panda",)
        self.dual_gello: DualGelloCommandController | None = None
        # Time-shared keyboard: one command source per arm, driven by Tab.
        # Single-arm scenes keep using ``self.control`` unchanged.
        self.arm_controls: dict[str, BrowserKeyboardControlSource] = {}
        self.active_arm = "left"
        if self.dual_arm:
            self.arm_controls = {
                "left": BrowserKeyboardControlSource(args.position_step, args.rotation_step),
                "right": BrowserKeyboardControlSource(args.position_step, args.rotation_step),
            }
            if args.control_source == "gello":
                self.dual_gello = self._build_dual_gello_controller()
        geometry_metadata = self.task_scene.geometry_metadata()
        control_source = (
            "browser_keyboard" if args.control_source == "keyboard" else "gello_joint_targets"
        )
        if args.control_source == "keyboard":
            operator_command_names = [
                "dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper_m"
            ]
        else:
            operator_command_names = [
                f"requested_joint_position/panda_joint{i}_rad" for i in range(1, 8)
            ]
        if self.dual_arm:
            # Left block then right block; names are the single-arm names with an arm prefix.
            operator_command_names = [
                f"{arm}/{name}" for arm in ("left", "right") for name in operator_command_names
            ]
        writer_class = DualFrankaEpisodeWriter if self.dual_arm else FrankaEpisodeWriter
        self.recorder = writer_class(
            args.data_dir,
            fps=args.control_fps,
            engine_id="genesis",
            engine_version=getattr(gs, "__version__", "unknown"),
            scene_id=self.task_scene.scene_id,
            seed=args.seed,
            worker_id="gello_remote_genesis",
            control_source=control_source,
            state_names=self.task_scene.state_names,
            action_names=self.task_scene.action_names,
            operator_command_names=operator_command_names,
            camera_names=self.task_scene.camera_names,
            run_metadata={
                "robot": (
                    "Two Genesis built-in Franka Panda with original parallel grippers"
                    if self.dual_arm
                    else "Genesis built-in Franka Panda with original parallel gripper"
                ),
                "arm_ids": list(self.arm_ids),
                "object_assets": (
                    "Genesis built-in cloth.obj with PBD material"
                    if is_cloth_scene(args.scene)
                    else "Genesis primitives defined by SceneSpec"
                ),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                "sim_dt_s": args.sim_dt,
                "substeps": args.substeps,
                "physics_steps_per_control": args.physics_steps_per_control,
                "control_fps": args.control_fps,
                "camera_resolution": [args.width, args.height],
                "camera_roles": self.task_scene.camera_roles,
                "camera_profile": self.task_scene.camera_profile,
                "scene_geometry": geometry_metadata,
                "action_bounds": geometry_metadata["action_bounds"],
                "resolved_scene_spec": self.task_scene.resolved_scene_spec(),
                "control_source": control_source,
                "control_source_options": ["browser_keyboard", "gello_joint_targets"],
                "operator_command_layouts": self._operator_command_layouts(),
                "git_roots": {"gello_remote": Path(gello_remote.__file__).resolve().parents[2],
                              "clothtwin": PROJECT_ROOT},
                "dependency_packages": ["genesis-world", "numpy", "opencv-python"],
            },
        )

    def _operator_command_layouts(self) -> dict[str, list[str]]:
        """Name the operator command channels actually used by this arm topology."""
        keyboard = ["dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper_m"]
        gello = [f"panda_joint{i}_rad" for i in range(1, 8)]
        if not self.dual_arm:
            return {"browser_keyboard": keyboard, "gello_joint_targets": gello}
        return {
            "browser_keyboard": [f"{arm}/{n}" for arm in ("left", "right") for n in keyboard],
            "gello_joint_targets": [f"{arm}/{n}" for arm in ("left", "right") for n in gello],
        }

    def _dual_joint_limits(self) -> list[list[tuple[float, float]]]:
        """Per-arm 7-DoF limits taken from the scene so both safety layers agree."""
        low = np.asarray(self.task_scene.arm_joint_low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.task_scene.arm_joint_high, dtype=np.float32).reshape(-1)
        if low.shape != (14,) or high.shape != (14,):
            raise RuntimeError(
                "a dual-arm scene must expose 14 per-arm joint limits "
                f"(left 7 then right 7); got low={low.shape}, high={high.shape}"
            )
        return [
            [(float(low[i]), float(high[i])) for i in range(offset, offset + 7)]
            for offset in (0, 7)
        ]

    def _build_dual_gello_controller(self) -> DualGelloCommandController:
        limits = self._dual_joint_limits()
        initial = np.asarray(self.task_scene.arm_target, dtype=np.float32).reshape(-1)
        if initial.shape != (14,):
            raise RuntimeError(
                f"a dual-arm scene must expose a 14-DoF arm_target; got {initial.shape}"
            )
        # Match the scene's per-tick slew so the two safety layers do not fight.
        max_joint_speed = float(self.task_scene.max_joint_target_step) * float(self.args.control_fps)
        arms = {
            "left": GelloCommandController(
                initial[:7], limits[0],
                control_hz=self.args.control_fps,
                max_joint_speed=max_joint_speed,
                watchdog_timeout=self.args.gello_timeout,
            ),
            "right": GelloCommandController(
                initial[7:], limits[1],
                control_hz=self.args.control_fps,
                max_joint_speed=max_joint_speed,
                watchdog_timeout=self.args.gello_timeout,
            ),
        }
        return DualGelloCommandController(arms["left"], arms["right"])

    def _require_both_arms_live(self) -> None:
        """Hard gate for dual GELLO episodes. No single-arm fallback, ever."""
        assert self.dual_gello is not None
        now = time.monotonic()
        status = self.dual_gello.status(now=now)
        stale = []
        for arm in ("left", "right"):
            arm_status = status[arm]
            age_ms = arm_status.get("age_ms")
            fresh = age_ms is not None and float(age_ms) <= self.args.gello_timeout * 1000.0
            if arm_status.get("session_id") is None or not fresh:
                stale.append(f"{arm}(session={arm_status.get('session_id')}, age_ms={age_ms})")
        if stale:
            raise RuntimeError(
                "Refusing to start a dual-arm episode without both GELLO arms streaming: "
                + ", ".join(stale)
                + " (dual teleoperation is synchronous; there is no single-arm fallback)"
            )

    def _keyboard_source(self) -> BrowserKeyboardControlSource:
        """The arm the operator is currently driving.

        Single-arm scenes always return ``self.control`` so that path stays
        bit-identical. A dual scene returns the active arm's own source.
        """
        if not self.dual_arm:
            return self.control
        return self.arm_controls[self.active_arm]

    def _keyboard_sources(self) -> tuple[BrowserKeyboardControlSource, ...]:
        """Every live keyboard source. Key-up must clear the arm that holds the key."""
        if not self.dual_arm:
            return (self.control,)
        return (self.arm_controls["left"], self.arm_controls["right"])

    def reset(self) -> str:
        if self.recorder.is_recording:
            raise RuntimeError("Finish or discard the active episode before resetting")
        self.task_scene.reset()
        self.control.reset()
        for source in self._keyboard_sources():
            source.reset()
        self.sim_time = 0.0
        return "Environment reset"

    def start(self, task: str) -> str:
        if self.recorder.is_recording:
            raise RuntimeError("An episode is already recording")
        # Gate before reset(): reset drops nothing here, but the gate reads live
        # command ages and must judge the operator's actual streaming state.
        if self.dual_arm and self.args.control_source == "gello":
            self._require_both_arms_live()
        self.reset()
        if self.args.control_source == "keyboard" and self.keyboard_sequence is None:
            self.keyboard_sequence = 0
            self.keyboard_timestamp_ns = time.time_ns()
            self.keyboard_receive_timestamp_ns = time.monotonic_ns()
        self.current_episode = self.recorder.start(task)
        self.current_task = task.strip()
        return f"Episode {self.current_episode} started"

    def finish(self, operator_success: bool) -> str:
        auto_success, metrics = self.task_scene.evaluate()
        path = self.recorder.finish(
            operator_success=operator_success,
            auto_success=auto_success,
            episode_metrics=metrics,
        )
        self.current_episode = None
        self.current_task = ""
        return f"Episode saved: {path} (operator={operator_success}, auto={auto_success})"

    def discard(self) -> str:
        episode = self.current_episode
        self.recorder.discard()
        self.current_episode = None
        self.current_task = ""
        return f"Episode {episode} discarded"

    def handle_keydown(self, code: str, task: str) -> str | None:
        if self.args.control_source != "keyboard":
            raise ValueError("Keyboard motion is disabled in GELLO mode")
        if code == "KeyR":
            return self.reset()
        if code == "KeyC":
            return self.finish(True) if self.recorder.is_recording else self.start(task)
        if code == "KeyX":
            return self.finish(False)
        if code == "Backspace" and self.recorder.is_recording:
            return self.discard()
        if code == "Tab":
            return self._switch_active_arm()
        return self._keyboard_source().keydown(code)

    def _switch_active_arm(self) -> str | None:
        """Time-share the keyboard: Tab moves control to the other arm.

        The arm being left has its held motion keys cleared, so a key that is
        still physically down cannot keep driving the arm the operator just
        switched away from. Each arm keeps its own gripper state.
        """
        if not self.dual_arm:
            return None
        outgoing = self.active_arm
        self.active_arm = "right" if outgoing == "left" else "left"
        self.arm_controls[outgoing].pressed_keys.clear()
        return f"ACTIVE: {self.active_arm.upper()}"

    def record_keyboard_event(
        self, sequence: int, timestamp_ns: int, *, received_at_ns: int | None = None
    ) -> None:
        """Store the latest browser event provenance used by held keyboard commands."""
        self.keyboard_sequence = sequence
        self.keyboard_timestamp_ns = timestamp_ns
        self.keyboard_receive_timestamp_ns = (
            time.monotonic_ns() if received_at_ns is None else received_at_ns
        )

    def handle_command(self, command: str, task: str) -> str:
        if command == "start":
            return self.start(task)
        if command == "success":
            return self.finish(True)
        if command == "failure":
            return self.finish(False)
        if command == "discard":
            return self.discard()
        if command == "reset":
            return self.reset()
        raise ValueError(f"Unknown command: {command}")

    def handle_gello(self, data: dict) -> str | None:
        if self.args.control_source != "gello":
            raise ValueError("GELLO commands are disabled in keyboard mode")
        command = parse_gello_message(data)
        received_at = time.monotonic()
        routed = parse_arm_session_id(command.session_id)
        if self.dual_arm:
            if routed is None:
                raise ValueError(
                    "Dual-arm scenes route GELLO by session_id suffix; expected "
                    f"'<base>:left' or '<base>:right', got {command.session_id!r}"
                )
            return self._handle_dual_arm_gello(command, received_at, routed[1])
        if routed is not None:
            raise ValueError(
                f"Arm-suffixed session {command.session_id!r} requires a dual-arm scene"
            )
        return self._handle_single_arm_gello(command, received_at)

    def _handle_single_arm_gello(self, command: GelloCommandV1, received_at: float) -> str | None:
        if not self.gello_session_gate.accept(command, received_at=received_at):
            return f"Ignored stale or competing GELLO sequence {command.sequence} for session {command.session_id}"
        self.gello_last_sequences[command.session_id] = command.sequence
        self.gello_session_id = command.session_id
        self.gello_sequence = command.sequence
        self.gello_timestamp_ns = command.timestamp_ns
        self.gello_received_at = received_at
        self.gello_joint_targets = np.asarray(command.joint_targets, dtype=np.float32)
        self.gello_gripper_closed = command.gripper_closed
        self._watchdog_holding = False
        return None

    def _handle_dual_arm_gello(
        self, command: GelloCommandV1, received_at: float, arm: str
    ) -> str | None:
        assert self.dual_gello is not None
        if not self.dual_gello.submit(command, arm=arm, received_at=received_at):
            return f"Ignored stale or competing GELLO sequence {command.sequence} for session {command.session_id}"
        self.gello_last_sequences[command.session_id] = command.sequence
        return None

    def release_gello_session(self, session_id: str) -> None:
        """Release one exact session id from the gate that holds it."""
        if self.dual_arm:
            assert self.dual_gello is not None
            self.dual_gello.release_session(session_id)
            self.gello_last_sequences.pop(session_id, None)
        else:
            self.gello_session_gate.release(session_id)
            self.gello_last_sequences.pop(session_id, None)

    def handle_reconfigure(self, data: dict) -> str:
        if self.recorder.is_recording:
            raise RuntimeError("Finish or discard the active episode before applying parameters")
        scene_id = data.get("scene_id")
        parameters = data.get("parameters", {})
        if not isinstance(scene_id, str) or scene_id not in SCENE_IDS:
            raise ValueError(f"Unknown scene: {scene_id!r}")
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        validated: dict[str, float] = {}
        for key, value in parameters.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Scene parameter values must be numeric")
            validated[key] = float(value)
        apply_scene_overrides(scene_id, validated)
        self.pending_restart_argv = build_restart_argv(self.args, scene_id, validated)
        return f"Applying {scene_id}; collector will restart and the page will reconnect"

    def execute_pending_restart(self) -> None:
        argv = getattr(self, "pending_restart_argv", None)
        if argv:
            os.execv(sys.executable, argv)

    def _dual_gello_stages(self) -> tuple:
        """One coupled dual-arm GELLO cycle.

        Both arms are advanced only when both streams are live; otherwise the
        ``DualGelloCommandController`` freezes both. Native action layout is left
        7 joints + 2 gripper widths then right 7 + 2 (18 total). ``operator_command``
        is the raw per-arm ask (14), matching the single-arm rule that records what
        the operator requested before safety filtering.
        """
        assert self.dual_gello is not None
        dual = self.dual_gello.tick(now=time.monotonic())
        left, right = dual.left, dual.right
        left_closed = bool(left.gripper_closed)
        right_closed = bool(right.gripper_closed)
        action, _ = self.task_scene.prepare_joint_action(
            np.concatenate(
                [
                    np.asarray(left.joint_targets, dtype=np.float32),
                    np.asarray(right.joint_targets, dtype=np.float32),
                ]
            ),
            (left_closed, right_closed),
        )
        left_ask = np.asarray(left.requested_joint_targets, dtype=np.float32)
        right_ask = np.asarray(right.requested_joint_targets, dtype=np.float32)
        left_width = 0.0 if left_closed else 0.04
        right_width = 0.0 if right_closed else 0.04
        requested_action = np.concatenate(
            [left_ask, [left_width, left_width], right_ask, [right_width, right_width]]
        ).astype(np.float32)
        operator_command = np.concatenate([left_ask, right_ask]).astype(np.float32)
        (
            control_sequence,
            control_timestamp_ns,
            control_receive_timestamp_ns,
            extra,
        ) = self._dual_control_provenance(left, right, dual.coupled_hold)
        return (
            requested_action,
            action,
            operator_command,
            control_sequence,
            control_timestamp_ns,
            control_receive_timestamp_ns,
            extra,
        )

    @staticmethod
    def _dual_control_provenance(left, right, coupled_hold: bool) -> tuple:
        """Scalar provenance from the fresher arm; the full per-arm split is kept.

        A dual frame has two independent 7-DoF commands, so the scalar
        ``control_*`` fields cannot hold both. When the pair is advancing they
        carry the more recently received arm; when coupled hold freezes the pair
        they are -1 (matching the single-arm hold rule). ``extra_arrays`` always
        keeps both arms' sequence and timestamps plus the hold flag.
        """

        def block(prefix: str, side) -> dict[str, np.ndarray]:
            return {
                f"{prefix}_control_sequence": np.asarray(
                    [side.sequence if side.sequence is not None else -1], dtype=np.int64
                ),
                f"{prefix}_control_timestamp_ns": np.asarray(
                    [side.source_timestamp_ns if side.source_timestamp_ns is not None else -1],
                    dtype=np.int64,
                ),
                f"{prefix}_control_receive_timestamp_ns": np.asarray(
                    [side.receive_timestamp_ns if side.receive_timestamp_ns is not None else -1],
                    dtype=np.int64,
                ),
            }

        extra = {
            **block("left", left),
            **block("right", right),
            "dual_coupled_hold": np.asarray([coupled_hold], dtype=np.bool_),
        }
        both = (
            left.receive_timestamp_ns is not None and right.receive_timestamp_ns is not None
        )
        if coupled_hold or not both:
            return -1, -1, -1, extra
        newer = left if left.receive_timestamp_ns >= right.receive_timestamp_ns else right
        return (
            int(newer.sequence) if newer.sequence is not None else -1,
            int(newer.source_timestamp_ns) if newer.source_timestamp_ns is not None else -1,
            int(newer.receive_timestamp_ns),
            extra,
        )

    def _dual_keyboard_provenance(self) -> dict[str, np.ndarray]:
        """Per-arm split for a time-shared dual keyboard frame.

        Only the active arm has a live browser event this tick; the idle arm's
        three scalars are -1. ``dual_active_arm`` is 0 for left, 1 for right, and
        ``dual_coupled_hold`` is always False here: a hold means no command is
        advancing either arm, which is the GELLO watchdog's job, not Tab's.
        """
        active = self.active_arm

        def block(prefix: str, live: bool) -> dict[str, np.ndarray]:
            has_event = live and self.keyboard_sequence is not None
            return {
                f"{prefix}_control_sequence": np.asarray(
                    [int(self.keyboard_sequence) if has_event else -1], dtype=np.int64
                ),
                f"{prefix}_control_timestamp_ns": np.asarray(
                    [int(self.keyboard_timestamp_ns) if has_event else -1], dtype=np.int64
                ),
                f"{prefix}_control_receive_timestamp_ns": np.asarray(
                    [
                        int(self.keyboard_receive_timestamp_ns)
                        if has_event and self.keyboard_receive_timestamp_ns is not None
                        else -1
                    ],
                    dtype=np.int64,
                ),
            }

        return {
            **block("left", active == "left"),
            **block("right", active == "right"),
            "dual_active_arm": np.asarray([0 if active == "left" else 1], dtype=np.int8),
            "dual_coupled_hold": np.asarray([False], dtype=np.bool_),
        }

    def control_tick(self) -> bytes:
        gello_fresh = (
            self.gello_received_at is not None
            and time.monotonic() - self.gello_received_at <= self.args.gello_timeout
        )
        dual_provenance: dict[str, np.ndarray] = {}
        if self.dual_arm and self.args.control_source == "gello":
            (
                requested_action,
                action,
                operator_command,
                control_sequence,
                control_timestamp_ns,
                control_receive_timestamp_ns,
                dual_provenance,
            ) = self._dual_gello_stages()
        elif self.args.control_source == "keyboard":
            sample = self._keyboard_source().sample()
            if self.dual_arm:
                requested_action, action, operator_command = self.task_scene.prepare_keyboard_action(
                    sample, self.active_arm
                )
                dual_provenance = self._dual_keyboard_provenance()
            else:
                requested_action, action, operator_command = self.task_scene.prepare_action_stages(sample)
            control_sequence = int(self.keyboard_sequence) if self.keyboard_sequence is not None else -1
            control_timestamp_ns = (
                int(self.keyboard_timestamp_ns) if self.keyboard_timestamp_ns is not None else -1
            )
            control_receive_timestamp_ns = (
                int(self.keyboard_receive_timestamp_ns)
                if self.keyboard_receive_timestamp_ns is not None
                else -1
            )
        elif gello_fresh:
            action, operator_command = self.task_scene.prepare_joint_action(
                self.gello_joint_targets, self.gello_gripper_closed
            )
            operator_command = self.gello_joint_targets.copy()
            control_sequence = int(self.gello_sequence)
            control_timestamp_ns = int(self.gello_timestamp_ns)
            control_receive_timestamp_ns = int(self.gello_received_at * 1e9)
            gripper_target = 0.0 if self.gello_gripper_closed else 0.04
            requested_action = np.concatenate(
                [self.gello_joint_targets, [gripper_target, gripper_target]]
            ).astype(np.float32)
        else:
            if self.gello_received_at is not None and not self._watchdog_holding:
                self.gello_watchdog_holds += 1
                self._watchdog_holding = True
            action = np.concatenate(
                [
                    self.task_scene.arm_target,
                    [self.task_scene.gripper_target, self.task_scene.gripper_target],
                ]
            ).astype(np.float32)
            operator_command = np.zeros(7, dtype=np.float32)
            control_sequence = -1
            control_timestamp_ns = -1
            control_receive_timestamp_ns = -1
            requested_action = action
        observation = self.task_scene.observe()
        images = self.task_scene.render()
        capture_timestamp_ns = time.monotonic_ns()
        self.task_scene.apply_action(action, self.args.physics_steps_per_control)
        if self.recorder.is_recording:
            self.recorder.add_frame(
                images=images,
                state=observation.state,
                action=action,
                requested_action=requested_action,
                safe_action=action,
                applied_action=action,
                operator_command=operator_command,
                ee_pose=observation.ee_pose,
                target_ee_pose=self.task_scene.target_pose(),
                object_pose=observation.object_pose,
                object_velocity=observation.object_velocity,
                sim_time=self.sim_time,
                control_sequence=control_sequence,
                control_timestamp_ns=control_timestamp_ns,
                control_receive_timestamp_ns=control_receive_timestamp_ns,
                capture_timestamp_ns=capture_timestamp_ns,
                cloth_particles=observation.cloth_particles,
                cloth_particle_velocities=observation.cloth_particle_velocities,
                extra_arrays=dual_provenance,
            )
        self.sim_time += self.args.sim_dt * self.args.physics_steps_per_control
        mosaic = self._compose_mosaic(images)
        ok, jpeg = cv2.imencode(
            ".jpg",
            cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.args.stream_jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode browser frame")
        self._fps_window_ticks += 1
        now = time.perf_counter()
        elapsed = now - self._fps_window_started
        if elapsed >= 1.0:
            self.stream_fps = self._fps_window_ticks / elapsed
            self._fps_window_ticks = 0
            self._fps_window_started = now
        return jpeg.tobytes()

    def _compose_mosaic(self, images: dict[str, np.ndarray]) -> np.ndarray:
        # The live mosaic stays 2x2 so the browser zoom grid in collector_common.mjs
        # keeps working unchanged. Dual-arm streams put both wrists in the bottom
        # row; the `side` view is still rendered and recorded in the episode, it is
        # only left out of the live mosaic.
        tiles = (
            (
                ("base", "1  THIRD-PERSON"),
                ("top", "2  TOP"),
                ("left_wrist", "3  LEFT WRIST"),
                ("right_wrist", "4  RIGHT WRIST"),
            )
            if self.dual_arm
            else (
                ("base", "1  THIRD-PERSON"),
                ("top", "2  TOP"),
                ("side", "3  SIDE"),
                ("wrist", "4  WRIST"),
            )
        )
        views = []
        for name, label in tiles:
            view = images[name].copy()
            cv2.rectangle(view, (0, 0), (230, 32), (12, 12, 12), thickness=-1)
            cv2.putText(
                view,
                label,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            views.append(view)
        return np.concatenate(
            [np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)],
            axis=0,
        )

    def _gello_status(self) -> dict:
        """Top-level keys stay single-arm shaped so the browser latency chip keeps working."""
        # Keyboard mode never builds a DualGelloCommandController, so a dual scene
        # has no per-arm GELLO state to report. Report the no-session status rather
        # than asserting: status() runs on every websocket connect and every frame.
        if not self.dual_arm or self.dual_gello is None:
            return {
                "connected": bool(
                    self.gello_received_at is not None
                    and time.monotonic() - self.gello_received_at <= self.args.gello_timeout
                ),
                "sequence": self.gello_sequence,
                "session_id": self.gello_session_id,
                "age_ms": (
                    round((time.monotonic() - self.gello_received_at) * 1000.0, 1)
                    if self.gello_received_at is not None
                    else None
                ),
                "source_age_ms": (
                    round((time.time_ns() - self.gello_timestamp_ns) / 1e6, 1)
                    if self.gello_timestamp_ns is not None
                    else None
                ),
                "watchdog_holds": self.gello_watchdog_holds,
            }
        assert self.dual_gello is not None
        now = time.monotonic()
        combined = self.dual_gello.status(now=now)
        arms = {"left": combined["left"], "right": combined["right"]}
        ages = [arms[arm]["age_ms"] for arm in ("left", "right")]
        source_ages = [arms[arm]["source_age_ms"] for arm in ("left", "right")]
        live = [
            arms[arm]["session_id"] is not None
            and arms[arm]["age_ms"] is not None
            and float(arms[arm]["age_ms"]) <= self.args.gello_timeout * 1000.0
            for arm in ("left", "right")
        ]
        return {
            "connected": all(live),
            "arms": arms,
            "coupled_hold": combined["coupled_hold"],
            "coupled_holds": combined["coupled_holds"],
            "sequence": max(
                (arms[arm]["sequence"] for arm in ("left", "right") if arms[arm]["sequence"] is not None),
                default=None,
            ),
            "session_id": {
                arm: arms[arm]["session_id"] for arm in ("left", "right")
            },
            # Staler arm wins so the chip reflects the pair, not the leader.
            "age_ms": max((value for value in ages if value is not None), default=None),
            "source_age_ms": max(
                (value for value in source_ages if value is not None), default=None
            ),
            "watchdog_holds": sum(arms[arm]["watchdog_holds"] for arm in ("left", "right")),
        }

    def status(self, log: str | None = None) -> dict:
        status = {
            "scene_id": self.task_scene.scene_id,
            "scene_description": self.task_scene.spec.description,
            "default_task": self.task_scene.task_instruction,
            "recording": self.recorder.is_recording,
            "episode": self.current_episode,
            "frames": self.recorder.frame_count,
            "task": self.current_task,
            "metrics": self.task_scene.status_metrics(),
            "stream_fps": self.stream_fps,
            "control_mode": self.args.control_source,
            "arm_ids": list(self.arm_ids),
            "dual_arm": self.dual_arm,
            "active_arm": self.active_arm if self.dual_arm else None,
            "scene_catalog": scene_catalog(),
            "scene_parameters": scene_parameter_schema(self.task_scene.scene_id),
            "resolved_scene_spec": self.task_scene.resolved_scene_spec(),
            "runtime": {
                "sim_dt_s": self.args.sim_dt,
                "substeps": self.args.substeps,
                "physics_steps_per_control": self.args.physics_steps_per_control,
                "control_fps": self.args.control_fps,
                "camera_resolution": [self.args.width, self.args.height],
                "camera_profile": self.task_scene.camera_profile,
            },
            "gello": self._gello_status(),
        }
        if log:
            status["log"] = log
        return status


async def ws_handler(websocket, server: TeleopServer) -> None:
    server.clients.add(websocket)
    legacy_gello_session_id = f"legacy-{time.time_ns()}-{id(websocket)}"
    # A dual GELLO page owns two arm streams; each binds its own session id.
    bound_gello_session_ids: set[str] = set()
    last_keyboard_sequence: int | None = None
    await websocket.send(json.dumps(server.status("Client connected")))
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                log = None
                message_type = data.get("type")
                if not isinstance(message_type, str) or not control_message_allowed(
                    server.args.control_source, message_type
                ):
                    raise ValueError(
                        f"Message type {message_type!r} is disabled in {server.args.control_source} mode"
                    )
                if message_type == "keydown":
                    sequence, timestamp_ns = parse_keyboard_event(
                        data, last_sequence=last_keyboard_sequence
                    )
                    last_keyboard_sequence = sequence
                    server.record_keyboard_event(sequence, timestamp_ns)
                    log = server.handle_keydown(data["code"], data.get("task", server.task_scene.task_instruction))
                elif message_type == "keyup":
                    sequence, timestamp_ns = parse_keyboard_event(
                        data, last_sequence=last_keyboard_sequence
                    )
                    last_keyboard_sequence = sequence
                    server.record_keyboard_event(sequence, timestamp_ns)
                    # Clear from every source: the key may still be held on the arm
                    # the operator switched away from, whose source is no longer active.
                    for source in server._keyboard_sources():
                        source.keyup(data["code"])
                elif message_type == "command":
                    log = server.handle_command(data["command"], data.get("task", server.task_scene.task_instruction))
                elif message_type == "gello":
                    data.setdefault("session_id", legacy_gello_session_id)
                    log = server.handle_gello(data)
                    if log is None:
                        bound_gello_session_ids.add(str(data["session_id"]))
                elif message_type == "reconfigure":
                    log = server.handle_reconfigure(data)
                if log:
                    await websocket.send(json.dumps(server.status(log)))
                if message_type == "reconfigure" and getattr(server, "pending_restart_argv", None):
                    asyncio.get_running_loop().call_later(0.25, server.execute_pending_restart)
            except (KeyError, RuntimeError, ValueError, OSError) as error:
                await websocket.send(json.dumps(server.status(f"ERROR: {error}")))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        for session_id in bound_gello_session_ids:
            server.release_gello_session(session_id)
        server.clients.discard(websocket)
        server.control.pressed_keys.clear()


async def collector_loop(server: TeleopServer) -> None:
    target_period = 1.0 / server.args.control_fps
    while True:
        started = time.perf_counter()
        jpeg = server.control_tick()
        await asyncio.sleep(0)
        if server.clients:
            status = json.dumps(server.status())
            for client in list(server.clients):
                try:
                    await client.send(jpeg)
                    await client.send(status)
                except websockets.exceptions.ConnectionClosed:
                    server.clients.discard(client)
        remaining = target_period - (time.perf_counter() - started)
        if remaining > 0:
            await asyncio.sleep(remaining)


async def serve(server: TeleopServer) -> None:
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def process_request(connection, request):
        if request.path == "/" and "Upgrade" not in request.headers:
            return Response(
                200,
                "OK",
                Headers([("Content-Type", "text/html; charset=utf-8")]),
                HTML_PAGES[server.args.control_source],
            )
        if request.path in STATIC_ASSETS and "Upgrade" not in request.headers:
            content_type, path = STATIC_ASSETS[request.path]
            return Response(200, "OK", Headers([("Content-Type", content_type)]), path.read_bytes())
        return None

    ws_server = await websockets.serve(
        lambda ws: ws_handler(ws, server),
        server.args.host,
        server.args.port,
        process_request=process_request,
        max_size=2**20,
    )
    print(f"Collector UI ({server.args.control_source}): http://{server.args.host}:{server.args.port}")
    print(f"Raw dataset: {Path(server.args.data_dir).resolve()}")
    loop_task = asyncio.create_task(collector_loop(server))
    try:
        await ws_server.wait_closed()
    finally:
        loop_task.cancel()


def parse_args(control_source: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    if control_source is None:
        parser.add_argument("--control-source", choices=CONTROL_SOURCES, default="keyboard")
    else:
        parser.set_defaults(control_source=control_source)
    parser.add_argument("--scene", choices=SCENE_IDS, default="rigid_pick_place")
    parser.add_argument(
        "--scene-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a whitelisted build-time scene parameter (repeatable)",
    )
    parser.add_argument("--list-scenes", action="store_true", help="Print registered scenes and exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--control-fps", type=int, default=10)
    parser.add_argument("--physics-steps-per-control", type=int, default=10)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--camera-profile-file",
        type=Path,
        help="Robot Data Studio camera-profile JSON; defaults to the built-in nominal D435i profile",
    )
    parser.add_argument("--stream-jpeg-quality", type=int, default=80)
    parser.add_argument("--position-step", type=float, default=0.006)
    parser.add_argument("--rotation-step", type=float, default=0.025)
    parser.add_argument("--gello-timeout", type=float, default=0.3)
    parser.add_argument("--logging-level", choices=("debug", "info", "warning", "error"), default="warning")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Record N ticks and exit without starting a server")
    args = parser.parse_args()
    args.data_dir_explicit = args.data_dir is not None
    if args.data_dir is None:
        args.data_dir = PROJECT_ROOT / "data/franka/teleop_raw" / args.scene
    try:
        args.scene_overrides = parse_scene_params(args.scene_param)
        apply_scene_overrides(args.scene, args.scene_overrides)
    except ValueError as error:
        parser.error(str(error))
    positive = (
        args.control_fps,
        args.physics_steps_per_control,
        args.sim_dt,
        args.substeps,
        args.width,
        args.height,
        args.position_step,
        args.rotation_step,
        args.gello_timeout,
    )
    if min(positive) <= 0 or args.smoke_steps < 0:
        parser.error("simulation, image, control values must be positive; smoke-steps cannot be negative")
    try:
        validate_control_timing(
            args.control_fps, args.sim_dt, args.physics_steps_per_control
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        args.camera_profile = load_camera_profile(
            args.camera_profile_file, width=args.width, height=args.height
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    return args


def main(control_source: str | None = None) -> None:
    args = parse_args(control_source)
    args.launch_module = "gello_remote.genesis_worker"
    if args.list_scenes:
        for item in scene_catalog():
            print(f"{item['scene_id']:<20} {item['description']}")
        return
    server = TeleopServer(args)
    if args.smoke_steps:
        server.start(server.task_scene.task_instruction)
        for _ in range(args.smoke_steps):
            server.control_tick()
        print(server.finish(False))
        return
    try:
        asyncio.run(serve(server))
    except KeyboardInterrupt:
        print("\nCollector stopped")


if __name__ == "__main__":
    main()
