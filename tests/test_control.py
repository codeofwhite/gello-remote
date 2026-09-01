import pytest

from gello_remote.control import GelloCommandController
from gello_remote.protocol import GelloCommand


def command(sequence: int, targets: list[float], *, gripper_closed: bool = False) -> GelloCommand:
    return GelloCommand(
        sequence=sequence,
        timestamp_ns=sequence + 1,
        joint_targets=targets,
        gripper_closed=gripper_closed,
    )


def controller(**kwargs) -> GelloCommandController:
    return GelloCommandController([0.0] * 7, [(-1.0, 1.0)] * 7, **kwargs)


def test_controller_slews_at_configured_joint_speed() -> None:
    state = controller(control_hz=20.0, max_joint_speed=1.0)
    assert state.submit(command(0, [0.5] * 7), received_at=1.0)
    output = state.tick(now=1.0)
    assert output.joint_targets == pytest.approx((0.05,) * 7)


def test_controller_clips_to_articulation_limits() -> None:
    state = controller(control_hz=20.0, max_joint_speed=100.0)
    state.submit(command(0, [2.0] * 7), received_at=1.0)
    assert state.tick(now=1.0).joint_targets == (1.0,) * 7


def test_controller_rejects_stale_sequences_without_replacing_command() -> None:
    state = controller()
    assert state.submit(command(2, [0.2] * 7), received_at=1.0)
    assert not state.submit(command(1, [-0.2] * 7), received_at=1.01)
    output = state.tick(now=1.01)
    assert output.sequence == 2
    assert all(value > 0 for value in output.joint_targets)
    assert state.status(now=1.01)["rejected_stale_commands"] == 1


def test_watchdog_holds_once_and_does_not_continue_slewing() -> None:
    state = controller(control_hz=20.0, max_joint_speed=1.0, watchdog_timeout=0.3)
    state.submit(command(0, [0.5] * 7, gripper_closed=True), received_at=1.0)
    fresh = state.tick(now=1.0)
    held = state.tick(now=1.31)
    held_again = state.tick(now=1.5)
    assert fresh.gripper_closed is True
    assert held.watchdog_holding is True
    assert held.joint_targets == fresh.joint_targets
    assert held_again.joint_targets == fresh.joint_targets
    assert state.status(now=1.5)["watchdog_holds"] == 1


@pytest.mark.parametrize("bad_limits", [[(-1.0, 1.0)] * 6, [(1.0, -1.0)] * 7])
def test_controller_rejects_invalid_joint_limits(bad_limits: list[tuple[float, float]]) -> None:
    with pytest.raises(ValueError):
        GelloCommandController([0.0] * 7, bad_limits)


def test_reset_targets_clears_live_command_without_reopening_stale_sequences() -> None:
    state = controller(control_hz=20.0, max_joint_speed=1.0)
    state.submit(command(4, [0.5] * 7), received_at=1.0)
    output = state.tick(now=1.0)
    assert output.source_timestamp_ns == 5

    state.reset_targets([0.1] * 7)
    reset = state.tick(now=2.0)
    assert reset.joint_targets == pytest.approx((0.1,) * 7)
    assert reset.source_timestamp_ns is None
    assert not state.submit(command(3, [0.2] * 7), received_at=2.0)
    assert state.submit(command(5, [0.2] * 7), received_at=2.0)
