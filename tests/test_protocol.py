import time

from gello_remote.protocol import Action, Observation


def test_protocol_round_trip() -> None:
    observation = Observation(sequence=1, timestamp_ns=time.time_ns(), joint_positions=[0.1] * 7)
    action = Action(sequence=observation.sequence, joint_targets=observation.joint_positions, policy="test")
    assert action.sequence == 1
    assert len(action.joint_targets) == 7
