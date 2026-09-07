from contextlib import contextmanager
from http import HTTPStatus
import math
import os
from pathlib import Path
from types import SimpleNamespace

from websockets.datastructures import Headers

from gello_remote.isaaclab_gello import (
    CAMERA_NAMES,
    CONTROL_SOURCES,
    CameraPreviewStore,
    BrowserSe3KeyboardState,
    CubeLiftEvaluator,
    DEFAULT_CAMERA_PROFILE,
    DEFAULT_WORK_SURFACE_HEIGHT_M,
    GelloWebServer,
    PREVIEW_CAMERA_IDS,
    PREVIEW_HEADER,
    PREVIEW_MAGIC,
    PREVIEW_PROTOCOL_VERSION,
    PoseConvergenceGate,
    RecordingCommandStore,
    STREAM_HEIGHT,
    STREAM_SCENE_PRIMS,
    STREAM_WIDTH,
    TelemetryStore,
    ViewCommandStore,
    WEB_CONSOLE_ROOT,
    _apply_split_gpu_launcher_config,
    _configure_gpu_selection,
    _encode_preview_packet,
    _euler_xyz_to_rotation_vector,
    _evaluation_position_split,
    _frequency_decimation,
    _camera_intrinsic_matrix,
    _camera_profile_metadata,
    _rigid_pick_place_geometry,
    _rigid_cube_lift_geometry,
    _rollout_stage_summary,
    _scene_camera_overrides,
    _load_camera_profile,
    _missing_scene_prims,
    _pump_kit_frame,
    _required_scene_prims,
    _running_control_services,
)


def test_trajectory_is_an_explicit_control_source() -> None:
    assert "trajectory" in CONTROL_SOURCES


def test_evaluation_position_split_distinguishes_nominal_and_planned_runs() -> None:
    assert _evaluation_position_split(None) == "nominal"
    assert _evaluation_position_split(SimpleNamespace(position_split="m2_train")) == "m2_train"
    assert _evaluation_position_split(SimpleNamespace(position_split="heldout")) == "heldout"


def keyboard_message(
    *,
    session_id: str = "browser-keyboard-test",
    sequence: int = 0,
    pressed: list[str] | None = None,
    gripper_closed: bool = False,
) -> dict:
    return {
        "type": "keyboard",
        "version": 1,
        "session_id": session_id,
        "sequence": sequence,
        "timestamp_ns": sequence + 1,
        "pressed": pressed or [],
        "gripper_closed": gripper_closed,
    }


def test_browser_keyboard_matches_official_se3_key_mapping() -> None:
    keyboard = BrowserSe3KeyboardState(
        position_step_m=0.01,
        rotation_step_rad=0.05,
        watchdog_timeout=0.3,
        clock=lambda: 1.0,
    )

    assert keyboard.submit(
        keyboard_message(sequence=0, pressed=["KeyW", "KeyA", "KeyQ", "KeyZ", "KeyT", "KeyC"]),
        received_at=1.0,
    )
    sample = keyboard.advance(now=1.1)

    expected_rotation = (0.05122917172495723, 0.04872967969036861, 0.05122917172495724)
    assert sample.delta_pose[:3] == (0.01, 0.01, 0.01)
    assert sample.session_id == "browser-keyboard-test"
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(sample.delta_pose[3:], expected_rotation)
    )
    assert sample.watchdog_holding is False


def test_keyboard_euler_conversion_matches_official_scipy_semantics() -> None:
    rotation = _euler_xyz_to_rotation_vector((0.05, 0.05, 0.05))

    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            rotation,
            (0.05122917172495723, 0.04872967969036861, 0.05122917172495724),
        )
    )


def test_browser_keyboard_full_state_stops_on_release_and_watchdog() -> None:
    keyboard = BrowserSe3KeyboardState(
        position_step_m=0.01,
        rotation_step_rad=0.05,
        watchdog_timeout=0.3,
        clock=lambda: 1.0,
    )
    keyboard.submit(keyboard_message(sequence=0, pressed=["KeyS"]), received_at=1.0)
    keyboard.submit(
        keyboard_message(sequence=1, pressed=[], gripper_closed=True), received_at=1.1
    )

    released = keyboard.advance(now=1.2)
    stale = keyboard.advance(now=1.5)

    assert released.delta_pose == (0.0,) * 6
    assert released.gripper_closed is True
    assert stale.delta_pose == (0.0,) * 6
    assert stale.watchdog_holding is True
    assert keyboard.status(now=1.5)["watchdog_holds"] == 1


def test_browser_keyboard_rejects_unknown_and_stale_inputs() -> None:
    keyboard = BrowserSe3KeyboardState(
        position_step_m=0.01,
        rotation_step_rad=0.05,
        watchdog_timeout=0.3,
        clock=lambda: 1.0,
    )
    keyboard.submit(keyboard_message(sequence=2), received_at=1.0)

    assert keyboard.submit(keyboard_message(sequence=2), received_at=1.1) is False
    try:
        keyboard.submit(keyboard_message(sequence=3, pressed=["KeyR"]), received_at=1.2)
    except ValueError as exc:
        assert "unsupported keyboard key" in str(exc)
    else:
        raise AssertionError("expected unknown keyboard motion key to fail")


def test_browser_keyboard_accepts_restarted_sequence_from_new_session() -> None:
    keyboard = BrowserSe3KeyboardState(
        position_step_m=0.01,
        rotation_step_rad=0.05,
        watchdog_timeout=0.3,
        clock=lambda: 1.0,
    )
    assert keyboard.submit(
        keyboard_message(session_id="old-browser", sequence=1200), received_at=1.0
    )
    keyboard.release_session("old-browser")
    assert keyboard.submit(
        keyboard_message(session_id="new-browser", sequence=0, pressed=["KeyW"]),
        received_at=1.1,
    )

    sample = keyboard.advance(now=1.2)
    assert sample.session_id == "new-browser"
    assert sample.sequence == 0
    assert sample.delta_pose[:3] == (0.01, 0.0, 0.0)


def test_rigid_pick_place_geometry_raises_objects_above_robot_base() -> None:
    geometry = _rigid_pick_place_geometry(DEFAULT_WORK_SURFACE_HEIGHT_M)

    assert geometry["work_surface_top_m"] == 0.25
    assert geometry["work_surface_center_m"] == (0.50, 0.00, 0.125)
    assert geometry["object_start_m"] == (0.55, -0.10, 0.277)
    assert geometry["target_center_m"] == (0.48, 0.12, 0.252)

    for invalid in (0.17, 0.33, math.nan):
        try:
            _rigid_pick_place_geometry(invalid)
        except ValueError as exc:
            assert "work-surface-height-m" in str(exc)
        else:
            raise AssertionError("expected invalid work surface height")


def test_rigid_cube_lift_geometry_has_no_placement_target() -> None:
    geometry = _rigid_cube_lift_geometry(DEFAULT_WORK_SURFACE_HEIGHT_M)

    assert geometry["object_start_m"] == (0.55, -0.10, 0.277)
    assert "target_center_m" not in geometry
    assert geometry["evaluator"]["lift_threshold_m"] == 0.08
    assert "hold for 1 second" in geometry["task_instruction"]


def test_pick_place_stream_validation_requires_target_zone() -> None:
    geometry = _rigid_pick_place_geometry(DEFAULT_WORK_SURFACE_HEIGHT_M)

    assert "/World/TargetZone" in _required_scene_prims(geometry)


def test_cube_lift_stream_validation_does_not_require_target_zone() -> None:
    geometry = _rigid_cube_lift_geometry(DEFAULT_WORK_SURFACE_HEIGHT_M)

    required = _required_scene_prims(geometry)
    assert "/World/WorkSurface" in required
    assert "/World/TaskObject" in required
    assert "/World/TargetZone" not in required


def test_cube_lift_evaluator_requires_close_lift_hold_and_release() -> None:
    evaluator = CubeLiftEvaluator(fps=10, initial_object_z_m=0.277)

    evaluator.update(object_z_m=0.277, gripper_closed=False)
    evaluator.update(object_z_m=0.367, gripper_closed=True)
    assert evaluator.result()["success"] is False
    assert evaluator.result()["grasp"] is True
    assert evaluator.result()["lift"] is True

    for _ in range(9):
        evaluator.update(object_z_m=0.367, gripper_closed=True)
    assert evaluator.result()["max_hold_frames"] == 10
    assert evaluator.result()["success"] is False

    evaluator.update(object_z_m=0.277, gripper_closed=False)
    result = evaluator.result()
    assert result["success"] is True
    assert result["hold"] is True
    assert result["release"] is True
    assert result["release_after_hold_seen"] is True
    assert math.isclose(result["hold_duration_s"], 1.0)


def test_cube_lift_evaluator_rejects_motion_without_a_valid_hold() -> None:
    evaluator = CubeLiftEvaluator(fps=10, initial_object_z_m=0.277)

    for _ in range(20):
        evaluator.update(object_z_m=0.357, gripper_closed=False)
    assert evaluator.result()["success"] is False

    evaluator.reset()
    for _ in range(9):
        evaluator.update(object_z_m=0.367, gripper_closed=True)
    evaluator.update(object_z_m=0.277, gripper_closed=False)
    assert evaluator.result()["success"] is False


def test_cube_lift_evaluator_does_not_call_close_alone_a_grasp() -> None:
    evaluator = CubeLiftEvaluator(fps=10, initial_object_z_m=0.277)

    for _ in range(20):
        evaluator.update(object_z_m=0.277, gripper_closed=True)

    result = evaluator.result()
    assert result["close_seen"] is True
    assert result["grasp"] is False
    assert result["lift"] is False


def test_cube_lift_evaluator_lift_mode_accepts_without_hold_or_release() -> None:
    evaluator = CubeLiftEvaluator(
        fps=20, initial_object_z_m=0.277, success_mode="lift"
    )

    evaluator.update(object_z_m=0.367, gripper_closed=True)

    result = evaluator.result()
    assert result["success"] is True
    assert result["lift"] is True
    assert result["hold"] is False
    assert result["release"] is False


def test_cube_lift_evaluator_requires_task_height_while_closed() -> None:
    evaluator = CubeLiftEvaluator(
        fps=20, initial_object_z_m=0.277, success_mode="lift"
    )
    evaluator.update(object_z_m=0.307, gripper_closed=True)
    evaluator.update(object_z_m=0.377, gripper_closed=False)

    assert evaluator.result()["success"] is False
    assert math.isclose(evaluator.result()["max_lift_m"], 0.1)
    assert evaluator.result()["max_closed_lift_m"] < 0.08


def test_pose_convergence_gate_requires_consecutive_stable_ticks() -> None:
    gate = PoseConvergenceGate(
        position_tolerance_m=0.003,
        rotation_tolerance_rad=math.radians(5.0),
        required_stable_ticks=2,
        timeout_ticks=4,
    )
    target = (0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0)

    assert gate.update((0.41, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0), target)[0] == "waiting"
    assert gate.update((0.402, 0.0, 0.3, -1.0, 0.0, 0.0, 0.0), target)[0] == "waiting"
    status, position_error, rotation_error = gate.update(
        (0.401, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0), target
    )
    assert status == "converged"
    assert position_error <= 0.003
    assert rotation_error == 0.0


def test_rollout_stage_summary_keeps_each_denominator_equal() -> None:
    reports = [
        {
            "success": False,
            "metrics": {"grasp": True, "lift": True, "hold": False, "release": False},
        },
        {
            "success": True,
            "metrics": {"grasp": True, "lift": True, "hold": True, "release": True},
        },
    ]

    summary = _rollout_stage_summary(reports)
    assert summary["overall_success_rate"] == 0.5
    assert summary["stages"]["grasp"]["rate"] == 1.0
    assert summary["stages"]["hold"]["rate"] == 0.5


def test_scene_camera_targets_follow_surface_without_mutating_profile() -> None:
    original_target = list(
        DEFAULT_CAMERA_PROFILE["engine_overrides"]["isaac_sim"]["base"]["target_world_m"]
    )
    overrides = _scene_camera_overrides(
        DEFAULT_CAMERA_PROFILE, work_surface_top_m=0.25
    )

    assert overrides["base"]["target_world_m"] == [0.45, 0.0, 0.40]
    assert overrides["top"]["target_world_m"] == [0.45, 0.0, 0.25]
    assert overrides["side"]["target_world_m"] == [0.45, 0.0, 0.39]
    assert (
        DEFAULT_CAMERA_PROFILE["engine_overrides"]["isaac_sim"]["base"]["target_world_m"]
        == original_target
    )


def test_stream_and_recording_defaults_share_four_by_three_framing() -> None:
    assert (STREAM_WIDTH, STREAM_HEIGHT) == (960, 720)
    assert STREAM_WIDTH / STREAM_HEIGHT == 640 / 480


def test_camera_preview_store_publishes_versioned_binary_packets(monkeypatch) -> None:
    monkeypatch.setattr(
        CameraPreviewStore,
        "_encode_jpeg",
        staticmethod(lambda image: b"jpeg-" + bytes([image])),
    )
    previews = CameraPreviewStore()

    images = {name: index for index, name in enumerate(CAMERA_NAMES, start=1)}
    previews.update(images, frame_index=42, captured_at_ns=123456)
    packets = previews.packets_since({name: -1 for name in CAMERA_NAMES})

    assert [name for name, _, _ in packets] == list(CAMERA_NAMES)
    for name, version, packet in packets:
        magic, protocol, camera_id, frame_index, captured_at_ns = PREVIEW_HEADER.unpack(
            packet[: PREVIEW_HEADER.size]
        )
        assert magic == PREVIEW_MAGIC
        assert protocol == PREVIEW_PROTOCOL_VERSION
        assert camera_id == PREVIEW_CAMERA_IDS[name]
        assert frame_index == 42
        assert captured_at_ns == 123456
        assert version == 1
        assert packet[PREVIEW_HEADER.size :].startswith(b"jpeg-")
    assert previews.packets_since({name: 1 for name in CAMERA_NAMES}) == []


def test_preview_packet_rejects_unknown_camera_and_empty_payload() -> None:
    for camera_name, jpeg, message in (
        ("unknown", b"jpeg", "unknown preview camera"),
        ("base", b"", "preview JPEG must contain"),
    ):
        try:
            _encode_preview_packet(
                camera_name, frame_index=0, captured_at_ns=0, jpeg=jpeg
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("expected invalid preview packet to fail")


def test_preview_worker_encodes_four_views_and_stops(monkeypatch) -> None:
    monkeypatch.setattr(
        CameraPreviewStore,
        "_encode_jpeg",
        staticmethod(lambda image: b"jpeg-" + bytes([image])),
    )
    previews = CameraPreviewStore(target_fps=10.0)
    previews.start()
    previews.submit(
        {name: index for index, name in enumerate(CAMERA_NAMES, start=1)},
        frame_index=2,
        captured_at_ns=3,
    )
    previews.stop()

    assert len(previews.packets_since({name: -1 for name in CAMERA_NAMES})) == 4
    assert previews.stats()["encoded"] == 1


def test_control_services_unwind_server_when_preview_start_fails() -> None:
    events: list[str] = []

    class FakeServer:
        def start(self) -> None:
            events.append("server.start")

        def stop(self) -> None:
            events.append("server.stop")

    class FailingPreviews:
        def start(self) -> None:
            events.append("previews.start")
            raise RuntimeError("preview startup failed")

        def stop(self) -> None:
            events.append("previews.stop")

    try:
        with _running_control_services(FakeServer(), FailingPreviews()):
            raise AssertionError("control services unexpectedly started")
    except RuntimeError as exc:
        assert str(exc) == "preview startup failed"
    else:
        raise AssertionError("expected preview startup to fail")

    assert events == ["server.start", "previews.start", "server.stop"]


def test_view_commands_validate_and_keep_active_view_separate() -> None:
    views = ViewCommandStore()
    views.submit("top")

    assert views.snapshot()["active"] == "free"
    assert views.snapshot()["requested"] == "top"
    assert views.drain() == ["top"]
    views.activate("top", "/World/RecordCameraTop")
    assert views.snapshot() == {
        "active": "top",
        "requested": "top",
        "camera_path": "/World/RecordCameraTop",
    }

    try:
        views.submit("unknown")
    except ValueError as exc:
        assert "unknown stream view" in str(exc)
    else:
        raise AssertionError("expected invalid stream view to fail")


def test_camera_frequencies_require_exact_control_decimation() -> None:
    assert _frequency_decimation(20.0, 10.0, label="preview-fps") == 2
    for target, message in ((0.0, "positive"), (30.0, "no greater"), (7.0, "integer multiple")):
        try:
            _frequency_decimation(20.0, target, label="preview-fps")
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("expected invalid camera frequency")


def test_lab_camera_profile_drives_dual_d435i_recording() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "robot-data-studio/config/camera_profiles/lab_d435i_dual_v1.json"
    )
    profile = _load_camera_profile(path, width=640, height=480)
    metadata = _camera_profile_metadata(profile)

    assert profile["engine_overrides"]["isaac_sim"]["wrist"]["prim_path"].endswith(
        "/panda_hand/wrist_cam"
    )
    assert profile["engine_overrides"]["isaac_sim"]["wrist"]["position_parent_m"] == [
        0.13,
        0.0,
        -0.19,
    ]
    assert profile["roles"]["wrist"]["nominal_vertical_fov_deg"] == 42.0
    assert profile["engine_overrides"]["isaac_sim"]["top"]["virtual_auxiliary"] is True
    assert profile["engine_overrides"]["isaac_sim"]["side"]["virtual_auxiliary"] is True
    assert metadata["match_level"] == "nominal_pending_intrinsics"
    assert metadata["roles"]["agentview"]["calibration_source_serial"] == "261722075807"
    assert metadata["roles"]["wrist"]["calibration_source_serial"] == "261622075579"


def test_nominal_and_calibrated_intrinsic_matrix_sources_are_explicit() -> None:
    nominal, source = _camera_intrinsic_matrix(
        {"intrinsics": None}, width=640, height=480, nominal_vertical_fov_deg=42.0
    )
    assert source == "nominal_vertical_fov"
    assert nominal[2] == 320.0 and nominal[5] == 240.0
    assert math.isclose(nominal[0], nominal[4])

    calibrated, source = _camera_intrinsic_matrix(
        {"intrinsics": {"fx": 600.0, "fy": 601.0, "cx": 319.5, "cy": 239.5}},
        width=640,
        height=480,
    )
    assert source == "calibrated_intrinsics"
    assert calibrated == [600.0, 0.0, 319.5, 0.0, 601.0, 239.5, 0.0, 0.0, 1.0]


class LegacyProtocol:
    def reject(self, status: HTTPStatus, text: str):
        return SimpleNamespace(status_code=int(status), body=text.encode(), headers={})


class FakePrim:
    def __init__(self, *, valid: bool = True, active: bool = True) -> None:
        self._valid = valid
        self._active = active

    def IsValid(self) -> bool:
        return self._valid

    def IsActive(self) -> bool:
        return self._active


class FakeStage:
    def __init__(self, prims: dict[str, FakePrim]) -> None:
        self._prims = prims

    def GetPrimAtPath(self, path: str) -> FakePrim:
        return self._prims.get(path, FakePrim(valid=False))


def test_gpu_id_configures_simulation_and_single_gpu_renderer(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    args = SimpleNamespace(gpu_id=6, device="cuda:0", kit_args="--/app/window/width=1280")

    _configure_gpu_selection(args, ["--gpu-id", "6"])

    assert args.device == "cuda:0"
    assert args.renderer_gpu_id == 6
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
    assert args.kit_args.split() == [
        "--/app/window/width=1280",
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/activeGpus=6",
        "--/renderer/multiGpu/maxGpuCount=1",
    ]


def test_split_gpu_launcher_config_uses_physical_renderer_and_logical_cuda() -> None:
    launcher_args: dict[str, object] = {"active_gpu": 0, "physics_gpu": 6}

    _apply_split_gpu_launcher_config(launcher_args, physical_gpu_id=6)

    assert launcher_args == {"active_gpu": 6, "physics_gpu": 0}


def test_gpu_id_rejects_negative_or_competing_device_selection() -> None:
    negative = SimpleNamespace(gpu_id=-1, device="cuda:0", kit_args="")
    competing_device = SimpleNamespace(gpu_id=6, device="cuda:2", kit_args="")

    for args, argv, message in (
        (negative, ["--gpu-id", "-1"], "gpu-id must be non-negative"),
        (competing_device, ["--device=cuda:2"], "use --gpu-id instead of --device"),
    ):
        try:
            _configure_gpu_selection(args, argv)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("expected invalid GPU selection to fail")


def test_gpu_id_rejects_gpu_routing_inside_kit_args() -> None:
    args = SimpleNamespace(
        gpu_id=6,
        device="cuda:0",
        kit_args="--/renderer/activeGpu=2 --/app/window/width=1280",
    )

    try:
        _configure_gpu_selection(args, ["--gpu-id", "6"])
    except ValueError as exc:
        assert "--/renderer/activeGpu" in str(exc)
    else:
        raise AssertionError("expected competing Kit GPU routing to fail")


def test_scene_prim_gate_reports_missing_and_inactive_prims() -> None:
    stage = FakeStage(
        {
            STREAM_SCENE_PRIMS[0]: FakePrim(),
            STREAM_SCENE_PRIMS[1]: FakePrim(active=False),
        }
    )

    assert _missing_scene_prims(stage) == [STREAM_SCENE_PRIMS[1], STREAM_SCENE_PRIMS[2]]


def test_telemetry_is_not_scene_ready_before_viewport_validation() -> None:
    telemetry = TelemetryStore().snapshot()

    assert telemetry["sim_ready"] is False
    assert telemetry["scene_prims_ready"] is False
    assert telemetry["viewport_ready"] is False
    assert telemetry["viewport_camera"] is None


def test_recording_commands_are_validated_and_drained_in_order() -> None:
    commands = RecordingCommandStore()
    commands.submit("start", "move safely")
    commands.submit("success")

    assert commands.drain() == [("start", "move safely"), ("success", "")]
    assert commands.drain() == []

    for command, task, message in (
        ("start", "", "non-empty task"),
        ("unknown", "task", "unknown recording command"),
    ):
        try:
            commands.submit(command, task)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("expected invalid recording command")


def test_kit_frame_pump_disables_automatic_physics_during_update() -> None:
    events: list[tuple[str, bool] | tuple[str]] = []

    class FakeSettings:
        def get(self, path: str) -> bool:
            assert path == "/app/player/playSimulations"
            return True

        def set_bool(self, path: str, value: bool) -> None:
            assert path == "/app/player/playSimulations"
            events.append(("play", value))

    class FakeApp:
        def update(self) -> None:
            events.append(("update",))

    _pump_kit_frame(FakeApp(), FakeSettings())

    assert events == [("play", False), ("update",), ("play", True)]


def test_process_request_supports_websockets_12_response_api() -> None:
    connection = SimpleNamespace(protocol=LegacyProtocol())
    request = SimpleNamespace(path="/", headers={})
    server = GelloWebServer(object(), TelemetryStore(), host="127.0.0.1", port=18081)

    response = server._process_request(connection, request)

    assert response.status_code == HTTPStatus.OK
    assert b"browser teleoperation console" in response.body.lower()
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert response.headers["Cache-Control"] == "no-store"


def test_set_header_replaces_websockets_default_content_type() -> None:
    response = SimpleNamespace(headers=Headers({"Content-Type": "text/plain; charset=utf-8"}))

    GelloWebServer._set_header(response, "Content-Type", "text/html; charset=utf-8")

    assert response.headers.get_all("Content-Type") == ["text/html; charset=utf-8"]


def test_process_request_serves_runtime_stream_config() -> None:
    connection = SimpleNamespace(protocol=LegacyProtocol())
    request = SimpleNamespace(path="/teleop-config.json", headers={})
    server = GelloWebServer(
        object(),
        TelemetryStore(),
        host="127.0.0.1",
        port=18081,
        stream_host="203.0.113.10",
        signaling_port=49100,
        media_port=47998,
        control_hz=20.0,
    )

    response = server._process_request(connection, request)

    assert response.status_code == HTTPStatus.OK
    assert b'"stream_host": "203.0.113.10"' in response.body
    assert b'"control_source": "gello"' in response.body
    assert b'"control_hz": 20.0' in response.body
    assert response.headers["Content-Type"] == "application/json"


def test_keyboard_server_requires_keyboard_state() -> None:
    try:
        GelloWebServer(
            object(),
            TelemetryStore(),
            control_source="keyboard",
            host="127.0.0.1",
            port=18081,
        )
    except ValueError as exc:
        assert "requires BrowserSe3KeyboardState" in str(exc)
    else:
        raise AssertionError("expected keyboard server without state to fail")


def test_process_request_serves_built_asset_and_rejects_traversal() -> None:
    connection = SimpleNamespace(protocol=LegacyProtocol())
    server = GelloWebServer(object(), TelemetryStore(), host="127.0.0.1", port=18081)
    script = next((WEB_CONSOLE_ROOT / "assets").glob("*.js"))

    response = server._process_request(
        connection,
        SimpleNamespace(path=f"/assets/{script.name}", headers={}),
    )
    rejected = server._process_request(
        connection,
        SimpleNamespace(path="/assets/../index.html", headers={}),
    )

    assert response.status_code == HTTPStatus.OK
    assert response.headers["Content-Type"] in {"text/javascript", "application/javascript"}
    assert rejected.status_code == HTTPStatus.NOT_FOUND


def test_server_starts_with_websockets_12_style_serve(monkeypatch) -> None:
    transport = SimpleNamespace(shutdown_called=False)

    def shutdown() -> None:
        transport.shutdown_called = True

    transport.shutdown = shutdown
    transport.serve_forever = lambda: None

    @contextmanager
    def legacy_serve(handler, host, port, *, process_request, max_size):
        del handler, host, port, process_request, max_size
        yield transport

    monkeypatch.setattr("websockets.sync.server.serve", legacy_serve)
    server = GelloWebServer(object(), TelemetryStore(), host="127.0.0.1", port=18081)

    server.start()
    server.stop()

    assert transport.shutdown_called is True
