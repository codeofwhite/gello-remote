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
FRANKA_GRIPPER_DOF = 2
FRANKA_GRIPPER_OPEN_M = 0.04
FRANKA_GRIPPER_CLOSED_M = 0.0


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

    @abstractmethod
    def set_command(self, joint_targets: Sequence[float], *, gripper_closed: bool) -> None:
        """Apply seven arm targets and an independent binary gripper command."""


class MockFrankaBackend(FrankaBackend):
    """Safe in-memory backend for ordinary tests and transport development."""

    def __init__(self) -> None:
        self.last_joint_targets: tuple[float, ...] | None = None
        self.last_gripper_closed: bool | None = None

    def set_joint_targets(self, joint_targets: Sequence[float]) -> None:
        self.last_joint_targets = validate_joint_targets(joint_targets)

    def set_command(self, joint_targets: Sequence[float], *, gripper_closed: bool) -> None:
        if not isinstance(gripper_closed, bool):
            raise ValueError("gripper_closed must be boolean")
        self.last_joint_targets = validate_joint_targets(joint_targets)
        self.last_gripper_closed = gripper_closed


class IsaacLabFrankaBackend(FrankaBackend):
    """Adapter for an already-created Isaac Lab Franka articulation.

    The caller owns simulator creation, stepping, and shutdown.  This backend
    only fills and writes the seven arm-joint target buffers.
    """

    _ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
    _FINGER_JOINT_PATTERN = "panda_finger_joint.*"

    def __init__(self, articulation: Any, *, device: str, robot_model: str = "panda") -> None:
        if robot_model not in {"panda", "fr3"}:
            raise ValueError("robot_model must be panda or fr3")
        self._ARM_JOINT_NAMES = tuple(f"{robot_model}_joint{i}" for i in range(1, 8))
        self._FINGER_JOINT_PATTERN = f"{robot_model}_finger_joint.*"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("IsaacLabFrankaBackend requires the Isaac Lab Python environment") from exc

        joint_ids, joint_names = articulation.find_joints(list(self._ARM_JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != self._ARM_JOINT_NAMES or len(joint_ids) != FRANKA_ARM_DOF:
            raise RuntimeError(
                f"Expected ordered {robot_model} joints1 through7; "
                f"found {joint_names!r}"
            )
        finger_ids, finger_names = articulation.find_joints(self._FINGER_JOINT_PATTERN, preserve_order=True)
        if len(finger_ids) != FRANKA_GRIPPER_DOF:
            raise RuntimeError(f"Expected two Isaac Lab Franka finger joints; found {finger_names!r}")
        self._articulation = articulation
        self._device = device
        self._joint_ids = joint_ids
        self._finger_joint_ids = finger_ids
        self._torch = torch

    def set_joint_targets(self, joint_targets: Sequence[float]) -> None:
        targets = validate_joint_targets(joint_targets)
        target_tensor = self._torch.tensor([targets], dtype=self._torch.float32, device=self._device)
        self._articulation.set_joint_position_target_index(target=target_tensor, joint_ids=self._joint_ids)
        self._articulation.write_data_to_sim()

    def set_command(self, joint_targets: Sequence[float], *, gripper_closed: bool) -> None:
        if not isinstance(gripper_closed, bool):
            raise ValueError("gripper_closed must be boolean")
        targets = validate_joint_targets(joint_targets)
        arm_tensor = self._torch.tensor([targets], dtype=self._torch.float32, device=self._device)
        finger_target = FRANKA_GRIPPER_CLOSED_M if gripper_closed else FRANKA_GRIPPER_OPEN_M
        finger_tensor = self._torch.full(
            (1, FRANKA_GRIPPER_DOF), finger_target, dtype=self._torch.float32, device=self._device
        )
        self._articulation.set_joint_position_target_index(target=arm_tensor, joint_ids=self._joint_ids)
        self._articulation.set_joint_position_target_index(target=finger_tensor, joint_ids=self._finger_joint_ids)
        self._articulation.write_data_to_sim()

    def current_joint_positions(self) -> tuple[float, ...]:
        """Return current seven arm positions [rad]."""
        values = self._articulation.data.joint_pos.torch[0, self._joint_ids]
        return tuple(float(value) for value in values.detach().cpu().tolist())

    def current_gripper_positions(self) -> tuple[float, float]:
        """Return current two finger positions [m]."""
        values = self._articulation.data.joint_pos.torch[0, self._finger_joint_ids]
        result = values.detach().cpu().tolist()
        if len(result) != FRANKA_GRIPPER_DOF:
            raise RuntimeError(f"Expected two current finger positions; found {result!r}")
        return float(result[0]), float(result[1])

    def current_joint_velocities(self) -> tuple[float, ...]:
        """Return current seven arm velocities [rad/s]."""
        values = self._articulation.data.joint_vel.torch[0, self._joint_ids]
        return tuple(float(value) for value in values.detach().cpu().tolist())

    def current_gripper_velocities(self) -> tuple[float, float]:
        """Return current two finger velocities [m/s]."""
        values = self._articulation.data.joint_vel.torch[0, self._finger_joint_ids]
        result = values.detach().cpu().tolist()
        if len(result) != FRANKA_GRIPPER_DOF:
            raise RuntimeError(f"Expected two current finger velocities; found {result!r}")
        return float(result[0]), float(result[1])

    def joint_position_limits(self) -> tuple[tuple[float, float], ...]:
        """Return Isaac Lab soft limits for the seven arm joints [rad]."""
        limits = self._articulation.data.soft_joint_pos_limits.torch[0, self._joint_ids]
        return tuple((float(lower), float(upper)) for lower, upper in limits.detach().cpu().tolist())
