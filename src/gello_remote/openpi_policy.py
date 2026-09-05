"""Engine-neutral OpenPI Franka action and rollout-video helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


MAX_GRIPPER_WIDTH_M = 0.08


@dataclass
class GripperCommandFilter:
    """Convert a continuous closed fraction to a stable binary command."""

    mode: str = "single_threshold"
    close_threshold: float = 0.5
    open_threshold: float = 0.5
    min_closed_frames: int = 0
    _closed: bool = field(init=False, default=False)
    _closed_frames: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mode not in {"single_threshold", "hysteresis"}:
            raise ValueError(f"unknown gripper filter mode: {self.mode}")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (self.close_threshold, self.open_threshold)
        ):
            raise ValueError("gripper thresholds must be finite values in [0, 1]")
        if self.mode == "hysteresis" and self.close_threshold <= self.open_threshold:
            raise ValueError("hysteresis close threshold must exceed open threshold")
        if self.min_closed_frames < 0:
            raise ValueError("minimum closed frames must be non-negative")

    @property
    def closed(self) -> bool:
        return self._closed

    def reset(self, *, initially_closed: bool = False) -> None:
        self._closed = bool(initially_closed)
        self._closed_frames = 1 if self._closed else 0

    def update(self, closed_fraction: float) -> bool:
        value = float(closed_fraction)
        if not math.isfinite(value):
            raise ValueError("gripper closed fraction must be finite")
        value = float(np.clip(value, 0.0, 1.0))
        if self.mode == "single_threshold":
            self._closed = value >= self.close_threshold
            self._closed_frames = self._closed_frames + 1 if self._closed else 0
            return self._closed

        if not self._closed:
            if value > self.close_threshold:
                self._closed = True
                self._closed_frames = 1
            return self._closed

        if self._closed_frames < self.min_closed_frames:
            self._closed_frames += 1
            return True
        if value < self.open_threshold:
            self._closed = False
            self._closed_frames = 0
            return False
        self._closed_frames += 1
        return True

    def configuration(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "close_threshold": self.close_threshold,
            "open_threshold": self.open_threshold,
            "min_closed_frames": self.min_closed_frames,
        }


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
