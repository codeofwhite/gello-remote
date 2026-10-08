import pytest

from gello_remote.isaac_bimanual_control import BimanualKeyboardState
from gello_remote.isaaclab_gello import BrowserSe3KeyboardState


def message(sequence, arm="left", pressed=None, closed=False, session="operator"):
    return dict(type="keyboard", version=1, session_id=session, sequence=sequence,
                timestamp_ns=sequence + 1, pressed=pressed or [],
                gripper_closed=closed, active_arm=arm)


def keyboard():
    return BimanualKeyboardState(BrowserSe3KeyboardState, clock=lambda: 1.0,
                                position_step_m=0.01, rotation_step_rad=0.05,
                                watchdog_timeout=0.3)


def test_switch_clears_held_motion_and_keeps_independent_grippers():
    state = keyboard()
    state.submit(message(0, pressed=["KeyW"], closed=True), received_at=1.0)
    assert state.advance(now=1.1).delta_pose[0] == 0.01
    state.submit(message(1, "right", ["KeyW"]), received_at=1.1)
    arm, sample = state.advance_with_arm(now=1.2)
    assert arm == "right" and sample.delta_pose == (0.0,) * 6
    assert state.status(now=1.2)["grippers"] == {"left": True, "right": False}
    state.submit(message(2, "right", ["KeyQ"]), received_at=1.2)
    assert state.advance(now=1.3).delta_pose[2] == 0.01
    state.submit(message(3, "left", closed=True), received_at=1.3)
    assert state.advance(now=1.4).gripper_closed is True
    assert state.advance(now=1.4).delta_pose == (0.0,) * 6


def test_competing_or_stale_packets_cannot_switch_arm():
    state = keyboard()
    state.submit(message(5), received_at=1.0)
    assert not state.submit(message(4, "right"), received_at=1.1)
    assert not state.submit(message(6, "right", session="intruder"), received_at=1.1)
    assert state.active_arm == "left"
    state.release_session("operator")
    assert state.submit(message(0, "right", session="new-operator"), received_at=1.2)
    assert state.active_arm == "right"


def test_watchdog_and_disconnect_hold_motion_without_opening_gripper():
    state = keyboard()
    state.submit(message(0, pressed=["KeyW"], closed=True), received_at=1.0)
    sample = state.advance(now=1.4)
    assert sample.watchdog_holding and sample.delta_pose == (0.0,) * 6
    assert sample.gripper_closed
    state.release_session("operator")
    assert state.advance(now=1.5).delta_pose == (0.0,) * 6
    assert state.advance(now=1.5).gripper_closed
    state.reset()
    assert state.active_arm == "left"
    assert state.status()["grippers"] == {"left": False, "right": False}


def test_invalid_packet_cannot_take_ownership():
    state = keyboard()
    with pytest.raises(ValueError):
        state.submit(message(0, "third"))
    with pytest.raises(ValueError):
        state.submit(message(0, pressed=["Tab"]))
    assert state.submit(message(0, session="valid"))


def test_invalid_receive_time_cannot_take_ownership():
    state = keyboard()
    with pytest.raises(ValueError, match="finite"):
        state.submit(message(0, session="invalid"), received_at=float("nan"))
    assert state.submit(message(0, session="valid"))


def test_inactive_safety_controller_freezes_pending_slew():
    from gello_remote.control import GelloCommandController
    from gello_remote.protocol import GelloCommand

    controller = GelloCommandController((0.0,) * 7, ((-2.0, 2.0),) * 7,
                                        control_hz=20, max_joint_speed=1,
                                        watchdog_timeout=0.3)
    controller.submit(GelloCommand(sequence=0, timestamp_ns=1,
                                  joint_targets=(1.0,) * 7, gripper_closed=True), received_at=1.0)
    moving = controller.tick(now=1.01)
    holding = controller.tick(now=1.02, force_hold=True)
    assert holding.joint_targets == moving.joint_targets
    assert holding.gripper_closed is True
