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
from gello_remote.control import GelloSessionGate
from gello_remote.protocol import GelloCommandV1

from franka_only.control import BrowserKeyboardControlSource
from franka_only.camera_profiles import load_camera_profile
from franka_only.dataset_schema import CONTROL_SOURCE_IDS
from franka_only.recording import RawEpisodeRecorder
from franka_only.scene_registry import (
    SCENE_IDS,
    apply_scene_overrides,
    scene_catalog,
    scene_parameter_schema,
)
from franka_only.scenes import build_scene


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
        gs.init(backend=gs.gpu, precision="32", logging_level=args.logging_level)
        self.task_scene = build_scene(args.scene, args, args.scene_overrides)
        geometry_metadata = self.task_scene.geometry_metadata()
        self.recorder = RawEpisodeRecorder(
            args.data_dir,
            fps=args.control_fps,
            state_names=self.task_scene.state_names,
            action_names=self.task_scene.action_names,
            operator_command_names=self.task_scene.operator_command_names,
            camera_names=self.task_scene.camera_names,
            run_metadata={
                "scene_id": self.task_scene.scene_id,
                "robot": "Genesis built-in Franka Panda with original parallel gripper",
                "object_assets": (
                    "Genesis built-in cloth.obj with PBD material"
                    if args.scene == "cloth_corner_lift"
                    else "Genesis primitives defined by SceneSpec"
                ),
                "genesis_version": getattr(gs, "__version__", "unknown"),
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
                "control_source": (
                    "browser_keyboard" if args.control_source == "keyboard" else "gello_joint_targets"
                ),
                "control_source_options": ["browser_keyboard", "gello_joint_targets"],
                "operator_command_layouts": {
                    "browser_keyboard": [
                        "dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper_m"
                    ],
                    "gello_joint_targets": [f"panda_joint{i}_rad" for i in range(1, 8)],
                },
            },
        )

    def reset(self) -> str:
        if self.recorder.is_recording:
            raise RuntimeError("Finish or discard the active episode before resetting")
        self.task_scene.reset()
        self.control.reset()
        self.sim_time = 0.0
        return "Environment reset"

    def start(self, task: str) -> str:
        if self.recorder.is_recording:
            raise RuntimeError("An episode is already recording")
        self.reset()
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
        return self.control.keydown(code)

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

    def control_tick(self) -> bytes:
        gello_fresh = (
            self.gello_received_at is not None
            and time.monotonic() - self.gello_received_at <= self.args.gello_timeout
        )
        if self.args.control_source == "keyboard":
            sample = self.control.sample()
            action, operator_command = self.task_scene.prepare_action(sample)
            control_sequence = -1
            control_timestamp_ns = -1
        elif gello_fresh:
            action, operator_command = self.task_scene.prepare_joint_action(
                self.gello_joint_targets, self.gello_gripper_closed
            )
            control_sequence = int(self.gello_sequence)
            control_timestamp_ns = int(self.gello_timestamp_ns)
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
        observation = self.task_scene.observe()
        images = self.task_scene.render()
        if self.recorder.is_recording:
            self.recorder.add_frame(
                images=images,
                state=observation.state,
                action=action,
                operator_command=operator_command,
                ee_pose=observation.ee_pose,
                target_ee_pose=self.task_scene.target_pose(),
                object_pose=observation.object_pose,
                object_velocity=observation.object_velocity,
                sim_time=self.sim_time,
                control_source_id=CONTROL_SOURCE_IDS[self.args.control_source],
                control_sequence=control_sequence,
                control_timestamp_ns=control_timestamp_ns,
                cloth_particles=observation.cloth_particles,
                cloth_particle_velocities=observation.cloth_particle_velocities,
            )
        self.task_scene.apply_action(action, self.args.physics_steps_per_control)
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

    @staticmethod
    def _compose_mosaic(images: dict[str, np.ndarray]) -> np.ndarray:
        labels = {
            "base": "1  THIRD-PERSON",
            "top": "2  TOP",
            "side": "3  SIDE",
            "wrist": "4  WRIST",
        }
        views = []
        for name in ("base", "top", "side", "wrist"):
            view = images[name].copy()
            cv2.rectangle(view, (0, 0), (230, 32), (12, 12, 12), thickness=-1)
            cv2.putText(
                view,
                labels[name],
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
            "gello": {
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
            },
        }
        if log:
            status["log"] = log
        return status


async def ws_handler(websocket, server: TeleopServer) -> None:
    server.clients.add(websocket)
    legacy_gello_session_id = f"legacy-{time.time_ns()}-{id(websocket)}"
    bound_gello_session_id: str | None = None
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
                    log = server.handle_keydown(data["code"], data.get("task", server.task_scene.task_instruction))
                elif message_type == "keyup":
                    server.control.keyup(data["code"])
                elif message_type == "command":
                    log = server.handle_command(data["command"], data.get("task", server.task_scene.task_instruction))
                elif message_type == "gello":
                    data.setdefault("session_id", legacy_gello_session_id)
                    log = server.handle_gello(data)
                    if log is None:
                        bound_gello_session_id = str(data["session_id"])
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
        if bound_gello_session_id is not None:
            server.gello_session_gate.release(bound_gello_session_id)
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
    parser.add_argument("--physics-steps-per-control", type=int, default=2)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=4)
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
