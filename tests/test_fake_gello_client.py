from gello_remote.fake_gello_client import FRANKA_HOME, make_command


def test_fake_command_is_bounded_and_controls_gripper() -> None:
    open_command = make_command(0, 0.0, amplitude=0.15)
    closed_command = make_command(1, 3.1, amplitude=0.15)
    assert len(open_command.joint_targets) == 7
    assert max(abs(value - home) for value, home in zip(open_command.joint_targets, FRANKA_HOME)) <= 0.15
    assert open_command.gripper_closed is False
    assert closed_command.gripper_closed is True
