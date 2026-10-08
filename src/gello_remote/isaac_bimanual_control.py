"""Time-shared Isaac keyboard control; no simulator imports at module load."""

from dataclasses import replace
import math
import threading
import time

from .control import GelloSessionGate

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


def keyboard_ik_targets(robot, ik, sample, hold_targets, *, device, world_axes=False):
    """Use a base-frame Jacobian, including for rotated workstation bases."""
    if not any(sample.delta_pose):
        return hold_targets
    import torch
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

    joint_ids = robot.find_joints(
        [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True
    )[0]
    hand_id = robot.find_bodies("panda_hand")[0][0]
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
