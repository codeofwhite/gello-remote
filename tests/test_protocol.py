import math
import time

import pytest
from pydantic import ValidationError

from gello_remote.protocol import Action, GelloCommand, Observation


def test_protocol_round_trip() -> None:
    observation = Observation(sequence=1, timestamp_ns=time.time_ns(), joint_positions=[0.1] * 7)
    action = Action(sequence=observation.sequence, joint_targets=observation.joint_positions, policy="test")
    assert action.sequence == 1
    assert len(action.joint_targets) == 7


def test_gello_command_accepts_franka_and_gripper_button() -> None:
    command = GelloCommand(
        sequence=4,
        timestamp_ns=time.time_ns(),
        joint_targets=[0.1] * 7,
        gripper_closed=True,
    )
    assert command.sequence == 4
    assert command.gripper_closed is True


@pytest.mark.parametrize("targets", [[0.0] * 6, [0.0] * 8, [0.0] * 6 + [math.nan]])
def test_gello_command_rejects_invalid_targets(targets: list[float]) -> None:
    with pytest.raises(ValidationError):
        GelloCommand(
            sequence=0,
            timestamp_ns=time.time_ns(),
            joint_targets=targets,
            gripper_closed=False,
        )
