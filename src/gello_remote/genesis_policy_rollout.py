"""Run a bounded OpenPI policy rollout in the Genesis Franka task scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np


MAX_GRIPPER_WIDTH_M = 0.08


def encode_policy_state(native_state: np.ndarray) -> np.ndarray:
    """Convert canonical Franka state to joint7 plus gripper-closed fraction."""
    from robot_data_contracts.profiles.franka import policy_state_from_native

    physical = policy_state_from_native(native_state)
    result = physical[:8].copy()
    result[7] = 1.0 - np.clip(result[7] / MAX_GRIPPER_WIDTH_M, 0.0, 1.0)
    return result.astype(np.float32)


def decode_policy_action(value: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    """Decode joint7 plus closed fraction into a physical requested command."""
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError(f"OpenPI action must contain 8 finite values, got {action.shape}")
    closed_fraction = float(np.clip(action[7], 0.0, 1.0))
    total_width = MAX_GRIPPER_WIDTH_M * (1.0 - closed_fraction)
    requested_native = np.concatenate(
        [action[:7], [total_width / 2.0, total_width / 2.0]]
    ).astype(np.float32)
    return action[:7].copy(), closed_fraction >= 0.5, requested_native


def validate_action_chunk(value: Any, *, replan_steps: int) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < replan_steps:
        raise ValueError(
            f"OpenPI action chunk must have shape (N, 8) with N >= {replan_steps}, "
            f"got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("OpenPI action chunk contains non-finite values")
    return actions[:replan_steps].copy()


def _encode_videos(episode_path: Path, camera_names: list[str], fps: int) -> dict[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode rollout videos")
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    encoder = next(
        (name for name in ("libx264", "libopenh264", "mpeg4") if name in encoders),
        None,
    )
    if encoder is None:
        raise RuntimeError("ffmpeg has no supported H.264 or MPEG-4 encoder")
    video_dir = episode_path / "videos"
    video_dir.mkdir(exist_ok=True)
    outputs: dict[str, str] = {}
    for camera in camera_names:
        output = video_dir / f"{camera}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(episode_path / f"{camera}_rgb" / "%06d.jpg"),
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
        outputs[camera] = str(output)
    return outputs


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import genesis as gs
    import gello_remote
    import franka_only
    from franka_only.camera_profiles import camera_profile_metadata
    from franka_only.scenes import build_scene
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from robot_data_contracts import FrankaEpisodeWriter

    gs.init(backend=gs.gpu, precision="32", logging_level=args.logging_level, seed=args.seed)
    task_scene = build_scene(args.scene, args, args.scene_overrides)
    task = args.task or task_scene.task_instruction
    geometry_metadata = task_scene.geometry_metadata()
    writer = FrankaEpisodeWriter(
        args.output_dir,
        fps=args.control_fps,
        engine_id="genesis",
        engine_version=getattr(gs, "__version__", "unknown"),
        scene_id=task_scene.scene_id,
        seed=args.seed,
        worker_id="gello_remote_genesis_policy_rollout",
        control_source="openpi_policy",
        state_names=task_scene.state_names,
        action_names=task_scene.action_names,
        operator_command_names=[
            *(f"model_target_joint/panda_joint{i}_rad" for i in range(1, 8)),
            "model_target_gripper_closed_fraction",
        ],
        camera_names=task_scene.camera_names,
        generation_method="policy_rollout",
        run_metadata={
            "robot": "Genesis built-in Franka Panda with original parallel gripper",
            "object_assets": "Genesis primitives defined by SceneSpec",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "sim_dt_s": args.sim_dt,
            "substeps": args.substeps,
            "physics_steps_per_control": args.physics_steps_per_control,
            "control_fps": args.control_fps,
            "camera_resolution": [args.width, args.height],
            "camera_roles": task_scene.camera_roles,
            "camera_profile": camera_profile_metadata(args.camera_profile),
            "scene_geometry": geometry_metadata,
            "action_bounds": geometry_metadata["action_bounds"],
            "resolved_scene_spec": task_scene.resolved_scene_spec(),
            "control_source": "openpi_policy",
            "policy": {
                "consumer": "openpi_franka_joint8",
                "config": args.policy_config,
                "checkpoint": str(args.policy_checkpoint),
                "server": f"ws://{args.policy_host}:{args.policy_port}",
                "replan_steps": args.replan_steps,
                "inference_steps": args.inference_steps,
                "gripper_threshold": 0.5,
            },
            "git_roots": {
                "gello_remote": Path(gello_remote.__file__).resolve().parents[2],
                "clothtwin": Path(franka_only.__file__).resolve().parents[1],
            },
            "dependency_packages": [
                "genesis-world",
                "numpy",
                "opencv-python",
                "websockets",
            ],
        },
    )

    client = WebsocketClientPolicy(args.policy_host, args.policy_port)
    metadata = client.get_server_metadata()
    if metadata.get("action_contract") != "joint7_absolute_plus_gripper_closed_fraction":
        raise RuntimeError(f"unexpected policy action contract: {metadata!r}")

    writer.start(task)
    action_plan: list[np.ndarray] = []
    replan_id = -1
    inference_timing = np.zeros(3, dtype=np.float32)
    command_timestamp_ns = -1
    command_receive_ns = -1
    sim_time = 0.0
    seen_closed = False
    released_after_close = False
    stable_success_frames = 0
    inference_calls = 0
    inference_samples: list[np.ndarray] = []
    error: str | None = None

    try:
        for frame_index in range(args.max_steps):
            observation = task_scene.observe()
            images = task_scene.render()
            capture_timestamp_ns = time.monotonic_ns()
            if not action_plan:
                command_timestamp_ns = time.time_ns()
                started = time.perf_counter()
                result = client.infer(
                    {
                        "observation/image": images["base"],
                        "observation/wrist_image": images["wrist"],
                        "observation/state": encode_policy_state(observation.state),
                        "prompt": task,
                    }
                )
                total_ms = (time.perf_counter() - started) * 1000.0
                command_receive_ns = time.monotonic_ns()
                chunk = validate_action_chunk(result["actions"], replan_steps=args.replan_steps)
                action_plan.extend(chunk)
                replan_id += 1
                inference_calls += 1
                server_ms = result.get("server_timing", {}).get("infer_ms", total_ms)
                policy_ms = result.get("policy_timing", {}).get("infer_ms", server_ms)
                inference_timing = np.asarray(
                    [total_ms, server_ms, policy_ms],
                    dtype=np.float32,
                )
                inference_samples.append(inference_timing.copy())

            model_action = action_plan.pop(0)
            joint_targets, gripper_closed, requested_native = decode_policy_action(model_action)
            safe_native, _ = task_scene.prepare_joint_action(joint_targets, gripper_closed)
            seen_closed = seen_closed or gripper_closed
            released_after_close = released_after_close or (seen_closed and not gripper_closed)
            writer.add_frame(
                images=images,
                state=observation.state,
                action=safe_native,
                requested_action=requested_native,
                safe_action=safe_native,
                applied_action=safe_native,
                operator_command=model_action,
                ee_pose=observation.ee_pose,
                target_ee_pose=task_scene.target_pose(),
                object_pose=observation.object_pose,
                object_velocity=observation.object_velocity,
                sim_time=sim_time,
                control_sequence=frame_index,
                control_timestamp_ns=command_timestamp_ns,
                control_receive_timestamp_ns=command_receive_ns,
                capture_timestamp_ns=capture_timestamp_ns,
                extra_arrays={
                    "model_action_raw": model_action,
                    "policy_replan_id": np.asarray([replan_id], dtype=np.int64),
                    "policy_inference_timing_ms": inference_timing,
                },
            )
            task_scene.apply_action(safe_native, args.physics_steps_per_control)
            sim_time += args.sim_dt * args.physics_steps_per_control

            post_observation = task_scene.observe()
            evaluator_success, _ = task_scene.evaluate()
            object_speed = float(np.linalg.norm(post_observation.object_velocity[:3]))
            if evaluator_success and released_after_close and object_speed <= args.stable_speed_m_s:
                stable_success_frames += 1
            else:
                stable_success_frames = 0
            if stable_success_frames >= args.success_hold_frames:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    evaluator_success, metrics = task_scene.evaluate()
    final_observation = task_scene.observe()
    final_object_speed = float(np.linalg.norm(final_observation.object_velocity[:3]))
    rollout_success = bool(
        evaluator_success
        and released_after_close
        and stable_success_frames >= args.success_hold_frames
        and error is None
    )
    metrics.update(
        {
            "policy_rollout_success": rollout_success,
            "policy_seen_closed": seen_closed,
            "policy_released_after_close": released_after_close,
            "stable_success_frames": stable_success_frames,
            "required_stable_success_frames": args.success_hold_frames,
            "stable_speed_threshold_m_s": args.stable_speed_m_s,
            "final_object_speed_m_s": final_object_speed,
            "inference_calls": inference_calls,
            "policy_error": error,
        }
    )
    if writer.frame_count == 0:
        writer.discard()
        raise RuntimeError(error or "policy rollout produced no frames")
    num_frames = writer.frame_count
    episode_path = writer.finish(
        operator_success=None,
        auto_success=rollout_success,
        episode_metrics=metrics,
        evaluator_id="rigid_pick_place_policy_rollout",
        evaluator_version="1",
    )
    report = {
        "status": "passed" if rollout_success else "failed",
        "closed_loop_validated": rollout_success,
        "episode_path": str(episode_path),
        "videos": {},
        "policy_config": args.policy_config,
        "policy_checkpoint": str(args.policy_checkpoint),
        "policy_server_metadata": metadata,
        "seed": args.seed,
        "frames": num_frames,
        "simulation_time_s": sim_time,
        "inference_timing_ms": (
            {
                "calls": len(inference_samples),
                "client_total_mean": float(np.mean(inference_samples, axis=0)[0]),
                "client_total_max": float(np.max(inference_samples, axis=0)[0]),
                "server_mean": float(np.mean(inference_samples, axis=0)[1]),
                "model_mean": float(np.mean(inference_samples, axis=0)[2]),
            }
            if inference_samples
            else {"calls": 0}
        ),
        "metrics": metrics,
    }
    _write_report(episode_path / "rollout_report.json", report)
    report["videos"] = _encode_videos(
        episode_path,
        task_scene.camera_names,
        args.control_fps,
    )
    _write_report(episode_path / "rollout_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if error is not None:
        raise RuntimeError(error)
    return report


def parse_args() -> argparse.Namespace:
    from franka_only.camera_profiles import load_camera_profile
    from franka_only.scene_registry import SCENE_IDS, apply_scene_overrides
    from franka_only.scenes import validate_control_timing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENE_IDS, default="rigid_pick_place")
    parser.add_argument("--scene-param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--task")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-config", default="pi05_franka_joint8_lora")
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--inference-steps", type=int, default=2)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--success-hold-frames", type=int, default=10)
    parser.add_argument("--stable-speed-m-s", type=float, default=0.01)
    parser.add_argument("--control-fps", type=int, default=10)
    parser.add_argument("--physics-steps-per-control", type=int, default=10)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-profile-file", type=Path)
    parser.add_argument(
        "--logging-level",
        choices=("debug", "info", "warning", "error"),
        default="warning",
    )
    args = parser.parse_args()
    if min(
        args.inference_steps,
        args.replan_steps,
        args.max_steps,
        args.success_hold_frames,
        args.control_fps,
        args.physics_steps_per_control,
        args.sim_dt,
        args.substeps,
        args.width,
        args.height,
        args.stable_speed_m_s,
    ) <= 0:
        parser.error("rollout, timing, image, and stability values must be positive")
    try:
        args.scene_overrides = {}
        for value in args.scene_param:
            key, raw = value.split("=", 1)
            if not key or key in args.scene_overrides:
                raise ValueError(f"invalid or repeated scene parameter: {key!r}")
            args.scene_overrides[key] = float(raw)
        apply_scene_overrides(args.scene, args.scene_overrides)
        validate_control_timing(args.control_fps, args.sim_dt, args.physics_steps_per_control)
        args.camera_profile = load_camera_profile(
            args.camera_profile_file,
            width=args.width,
            height=args.height,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir = args.output_dir.expanduser().resolve()
    args.policy_checkpoint = args.policy_checkpoint.expanduser().resolve()
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
