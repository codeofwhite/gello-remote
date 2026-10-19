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


def dual_gello():
    from gello_remote.control import GelloCommandController
    from gello_remote.isaac_bimanual_control import BimanualGelloState
    arms = [GelloCommandController((0.,)*7, ((-2.,2.),)*7, clock=lambda:1.) for _ in range(2)]
    return BimanualGelloState(*arms, clock=lambda:1.), arms


def dual_message(seq, arm, session="operator", value=1.):
    return dict(type="gello", version=1, arm=arm, session_id=session, sequence=seq,
                timestamp_ns=seq+1, joint_targets=[value]*7, gripper_closed=True)


def test_two_usb_streams_route_and_freeze_together_on_one_stale_source():
    state, _ = dual_gello()
    assert state.submit(dual_message(0,"left"),received_at=1.)
    assert state.tick(now=1.01).left.joint_targets == (0.,)*7
    assert state.submit(dual_message(1,"right",value=-1.),received_at=1.02)
    pair = state.tick(now=1.03)
    assert pair.left.joint_targets[0] > 0 and pair.right.joint_targets[0] < 0
    assert not pair.coupled_hold
    assert state.submit(dual_message(2,"left"),received_at=1.4)
    held=state.tick(now=1.41)
    assert held.coupled_hold and held.left.joint_targets == pair.left.joint_targets
    assert held.right.joint_targets == pair.right.joint_targets
    assert held.left.gripper_closed and held.right.gripper_closed


def test_competing_operator_and_malformed_packets_cannot_mix_arms():
    state, _ = dual_gello()
    with pytest.raises(ValueError):state.submit(dual_message(0,None),received_at=1.)
    assert state.submit(dual_message(0,"left"),received_at=1.)
    assert not state.submit(dual_message(1,"right",session="other"),received_at=1.01)
    assert not state.submit(dual_message(0,"right"),received_at=1.02)
    assert state.tick(now=1.03).coupled_hold
    assert state.submit(dual_message(1,"right"),received_at=1.04)
    assert not state.tick(now=1.05).coupled_hold
    state.release_session("operator")
    assert state.tick(now=1.06).coupled_hold


def test_wire_handler_accepts_both_sides_and_disconnect_releases_both():
    import json
    from websockets.exceptions import ConnectionClosed
    from gello_remote.isaaclab_gello import GelloWebServer, TelemetryStore
    state, arms = dual_gello()
    results=[]
    class Wire:
        def __init__(self):self.messages=iter([dual_message(0,"left"),dual_message(1,"right",value=-1.)])
        def send(self,payload):
            json.loads(payload)
            results.append(state.tick(now=1.))
        def recv(self,timeout):
            try:return json.dumps(next(self.messages))
            except StopIteration:raise ConnectionClosed(None,None)
    server=GelloWebServer(arms[0],TelemetryStore(),dual_gello=state,host="127.0.0.1",port=0)
    server._handler(Wire())
    assert any(not pair.coupled_hold and pair.left.joint_targets[0]>0 and pair.right.joint_targets[0]<0 for pair in results)
    assert state.tick(now=1.).coupled_hold


def test_fr3_recording_identity_and_units_are_explicit():
    from gello_remote.isaac_bimanual_control import fr3_writer_options
    options=fr3_writer_options()
    assert options["robot_id"] == "franka_fr3_dual"
    profile=options["action_profile"]
    assert profile.dimension == 16 and profile.units[7] == profile.units[15] == "m"
    assert all("panda_" not in name for name in profile.names)
    features=options["feature_builder"](tuple(f"operator{i}" for i in range(16)))
    assert next(spec for spec in features if spec.key=="policy_state").shape == (32,)
    assert all("panda_" not in name for spec in features for name in spec.names)
