"""Dual Isaac keyboard/GELLO control; no simulator imports at module load."""

from dataclasses import replace
import math
import threading
import time

from .control import GelloSessionGate, DualGelloCommandController, arm_session_id
from .protocol import GelloCommand
from collections import deque

ARM_IDS = ("left", "right")


class BimanualKeyboardState:
    """One browser owner, two independent grippers, and no motion across switches."""

    def __init__(self, state_factory, *, clock=time.monotonic, **kwargs):
        self._states = {arm: state_factory(clock=clock, **kwargs) for arm in ARM_IDS}
        self._gate = GelloSessionGate(kwargs["watchdog_timeout"], clock=clock)
        self._clock = clock
        self._lock = threading.RLock()
        self.active_arm = "left"
        self._grippers = dict.fromkeys(ARM_IDS, False)
        self._accepted = 0
        self._rejected = 0

    def submit(self, data, *, received_at=None):
        arm = data.get("active_arm", "left")
        if arm not in ARM_IDS:
            raise ValueError("active_arm must be left or right")
        # Validate before touching ownership or arm state.
        command = self._states[arm]._validate(data)
        now = self._clock() if received_at is None else received_at
        if not math.isfinite(now):
            raise ValueError("received_at must be finite")
        with self._lock:
            if not self._gate.accept(command, received_at=now):
                self._rejected += 1
                return False
            changed = arm != self.active_arm
            if changed:
                self._states[self.active_arm].reset()
                self.active_arm = arm
            # A switching packet is always a hold, even if a client sends held keys.
            accepted = self._states[arm].submit(
                {**data, "pressed": [] if changed else data["pressed"]},
                received_at=now,
            )
            if not accepted:
                raise RuntimeError("accepted keyboard owner rejected by arm state")
            self._grippers[arm] = command.gripper_closed
            self._accepted += 1
            return True

    def advance_with_arm(self, *, now=None):
        with self._lock:
            arm = self.active_arm
            sample = self._states[arm].advance(now=now)
            return arm, replace(sample, gripper_closed=self._grippers[arm])

    def advance(self, *, now=None):
        return self.advance_with_arm(now=now)[1]

    def status(self, *, now=None):
        with self._lock:
            return {
                **self._states[self.active_arm].status(now=now),
                "dual_arm": True,
                "active_arm": self.active_arm,
                "grippers": dict(self._grippers),
                "accepted_commands": self._accepted,
                "rejected_commands": self._rejected,
            }

    def release_session(self, session_id):
        with self._lock:
            self._gate.release(session_id)
            for state in self._states.values():
                state.release_session(session_id)

    def reset(self):
        with self._lock:
            for state in self._states.values():
                state.reset()
            self.active_arm = "left"
            self._grippers = dict.fromkeys(ARM_IDS, False)


def keyboard_ik_targets(robot, ik, sample, hold_targets, *, device, world_axes=False, robot_model="panda"):
    """Use a base-frame Jacobian, including for rotated workstation bases."""
    if not any(sample.delta_pose):
        return hold_targets
    import torch
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    joint_ids = robot.find_joints(
        [f"{robot_model}_joint{i}" for i in range(1, 8)], preserve_order=True
    )[0]
    hand_id = robot.find_bodies(f"{robot_model}_hand")[0][0]
    jacobian_id = hand_id - 1 if robot.is_fixed_base else hand_id
    root = robot.data.root_pose_w.torch
    hand = robot.data.body_pose_w.torch[:, hand_id]
    pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7])
    command = torch.tensor([sample.delta_pose], dtype=torch.float32, device=device)
    if world_axes:
        command = torch.cat([
            quat_apply_inverse(root[:, 3:7], command[:, :3]),
            quat_apply_inverse(root[:, 3:7], command[:, 3:]),
        ], dim=-1)
    jacobian = robot.data.body_link_jacobian_w.torch[:, jacobian_id][
        :, :, [i + robot.num_base_dofs for i in joint_ids]
    ].clone()
    rotation = root[:, 3:7].unsqueeze(1).expand(-1, len(joint_ids), -1)
    for block in (slice(0, 3), slice(3, 6)):
        jacobian[:, block, :] = quat_apply_inverse(
            rotation, jacobian[:, block, :].transpose(1, 2)
        ).transpose(1, 2)
    ik.set_command(command, pos, quat)
    desired = ik.compute(pos, quat, jacobian, robot.data.joint_pos.torch[:, joint_ids])
    if not bool(torch.isfinite(desired).all().item()):
        raise RuntimeError("Differential IK produced a non-finite Franka target")
    return tuple(float(v) for v in desired[0].detach().cpu().tolist())


class BimanualGelloState:
    """One browser owner, two fresh leader streams, and coupled watchdog holds."""

    def __init__(self, left, right, *, watchdog_timeout=0.3, clock=time.monotonic):
        self._dual = DualGelloCommandController(left, right)
        self._gate = GelloSessionGate(watchdog_timeout, clock=clock)
        self._clock = clock
        self._lock = threading.RLock()
        self._latencies = {arm: deque(maxlen=240) for arm in ARM_IDS}
        self._applied_sequences = dict.fromkeys(ARM_IDS)

    def submit(self, command, *, received_at=None):
        command = GelloCommand.model_validate(command)
        if command.arm not in ARM_IDS:
            raise ValueError("dual GELLO requires an explicit left/right arm")
        # Reserve suffix space before taking ownership; inner schemas share the128 limit.
        inner_session = arm_session_id(command.session_id, command.arm)
        inner = command.model_copy(update={"session_id": inner_session, "arm": None})
        inner = GelloCommand.model_validate(inner.model_dump())
        now = self._clock() if received_at is None else received_at
        with self._lock:
            if not self._gate.accept(command, received_at=now):
                return False
            return self._dual.submit(inner, arm=command.arm, received_at=now)

    def tick(self, *, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            pair = self._dual.tick(now=now)
            for arm in ARM_IDS:
                sample = getattr(pair, arm)
                if (not pair.coupled_hold and sample.sequence is not None
                        and sample.sequence != self._applied_sequences[arm]):
                    self._latencies[arm].append(max(0., now *1000 - sample.receive_timestamp_ns/1e6))
                    self._applied_sequences[arm] = sample.sequence
            return pair

    def status(self, *, now=None):
        with self._lock:
            result = self._dual.status(now=now)
            result.update(dual_arm=True, session_id=self._gate.active_session_id)
            for arm in ARM_IDS:
                values = sorted(self._latencies[arm])
                result[arm]["receive_to_tick_ms_p50"] = values[len(values)//2] if values else None
                result[arm]["receive_to_tick_ms_p95"] = values[min(len(values)-1, math.ceil(.95*len(values))-1)] if values else None
            return result

    def release_session(self, session_id):
        with self._lock:
            self._gate.release(session_id)
            for arm in ARM_IDS:
                self._dual.release_session(arm_session_id(session_id, arm))


def fr3_writer_options():
    """Reuse numeric dual-arm adapters, with explicit FR3 identity and feature names."""
    from robot_data_contracts.profiles.franka_dual import DUAL_JOINT8_ACTION, dual_franka_feature_specs

    def features(operator_names):
        return tuple(spec.model_copy(update={"names": tuple(name.replace("panda_", "fr3_") for name in spec.names)})
                     for spec in dual_franka_feature_specs(operator_names))

    return {"robot_id": "franka_fr3_dual", "feature_builder": features,
            "action_profile": DUAL_JOINT8_ACTION.model_copy(update={
                "id": "dual_fr3_joint16_physical",
                "names": tuple(name.replace("panda_", "fr3_") for name in DUAL_JOINT8_ACTION.names)})}
