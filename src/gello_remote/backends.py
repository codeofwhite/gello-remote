"""Simulation-first Franka joint-target backends.

This module deliberately contains no real-robot integration.  Isaac Lab and
Torch are imported only when :class:`IsaacLabFrankaBackend` is constructed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

FRANKA_ARM_DOF = 7


def validate_joint_targets(joint_targets: Sequence[float]) -> tuple[float, ...]:
    """Return seven finite Franka arm targets as an immutable tuple."""
    if isinstance(joint_targets, (str, bytes)) or len(joint_targets) != FRANKA_ARM_DOF:
        raise ValueError("Franka joint_targets must contain exactly 7 values")
    try:
        targets = tuple(float(value) for value in joint_targets)
    except (TypeError, ValueError) as exc:
        raise ValueError("Franka joint_targets must be numeric") from exc
    if not all(math.isfinite(value) for value in targets):
        raise ValueError("Franka joint_targets must be finite")
    return targets


class FrankaBackend(ABC):
    """Backend contract for commanding seven simulated Franka arm joints."""

    @abstractmethod
    def set_joint_targets(self, joint_targets: Sequence[float]) -> None:
        """Validate and apply exactly seven finite joint-position targets."""


class MockFrankaBackend(FrankaBackend):
    """Safe in-memory backend for ordinary tests and transport development."""

    def __init__(self) -> None:
        self.last_joint_targets: tuple[float, ...] | None = None

    def set_joint_targets(self, joint_targets: Sequence[float]) -> None:
        self.last_joint_targets = validate_joint_targets(joint_targets)


class IsaacLabFrankaBackend(FrankaBackend):
    """Adapter for an already-created Isaac Lab Franka articulation.

    The caller owns simulator creation, stepping, and shutdown.  This backend
    only fills and writes the seven arm-joint target buffers.
    """

    _ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))

    def __init__(self, articulation: Any, *, device: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("IsaacLabFrankaBackend requires the Isaac Lab Python environment") from exc

        joint_ids, joint_names = articulation.find_joints(list(self._ARM_JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != self._ARM_JOINT_NAMES or len(joint_ids) != FRANKA_ARM_DOF:
            raise RuntimeError(
                "Expected Isaac Lab Franka joints panda_joint1 through panda_joint7 in order; "
                f"found {joint_names!r}"
            )
        self._articulation = articulation
        self._device = device
        self._joint_ids = joint_ids
        self._torch = torch

    def set_joint_targets(self, joint_targets: Sequence[float]) -> None:
        targets = validate_joint_targets(joint_targets)
        target_tensor = self._torch.tensor([targets], dtype=self._torch.float32, device=self._device)
        self._articulation.set_joint_position_target_index(target=target_tensor, joint_ids=self._joint_ids)
        self._articulation.write_data_to_sim()
