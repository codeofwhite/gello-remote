"""Engine-neutral OpenPI Franka action and rollout-video helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


MAX_GRIPPER_WIDTH_M = 0.08


def encode_policy_state(native_state: np.ndarray) -> np.ndarray:
    """Convert canonical Franka state to joint7 plus gripper-closed fraction."""
    from robot_data_contracts.profiles.franka import policy_state_from_native

    physical = policy_state_from_native(native_state)
    result = physical[:8].copy()
    total_width = float(result[7])
    if total_width >= MAX_GRIPPER_WIDTH_M - 1e-6:
        result[7] = 0.0
    elif total_width <= 1e-6:
        result[7] = 1.0
    else:
        result[7] = 1.0 - np.clip(total_width / MAX_GRIPPER_WIDTH_M, 0.0, 1.0)
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


def _video_encoder(ffmpeg: str) -> str:
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
    return encoder


def encode_videos(episode_path: Path, camera_names: list[str], fps: int) -> dict[str, str]:
    """Encode every canonical JPEG camera stream into an inspectable MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode rollout videos")
    encoder = _video_encoder(ffmpeg)
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
