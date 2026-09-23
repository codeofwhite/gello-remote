import pytest

from gello_remote.control import (
    DualGelloCommandController,
    GelloCommandController,
    arm_session_id,
    parse_arm_session_id,
)
from gello_remote.protocol import GelloCommand


def command(
    sequence: int, targets: list[float], *, gripper_closed: bool = False, session_id: str = "legacy"
) -> GelloCommand:
    return GelloCommand(
        sequence=sequence,
        timestamp_ns=sequence + 1,
        joint_targets=targets,
        gripper_closed=gripper_closed,
        session_id=session_id,
    )


def controller(**kwargs) -> GelloCommandController:
    return GelloCommandController([0.0] * 7, [(-1.0, 1.0)] * 7, **kwargs)


def dual_controller(**kwargs) -> DualGelloCommandController:
    return DualGelloCommandController(controller(**kwargs), controller(**kwargs))


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


def test_force_hold_freezes_without_counting_as_a_watchdog_timeout() -> None:
    state = controller(control_hz=20.0, max_joint_speed=1.0, watchdog_timeout=0.3)
    state.submit(command(0, [0.5] * 7, gripper_closed=True), received_at=1.0)

    held = state.tick(now=1.0, force_hold=True)
    assert held.joint_targets == (0.0,) * 7
    assert held.gripper_closed is False
    assert held.requested_joint_targets == (0.5,) * 7
    assert held.watchdog_holding is False
    assert state.status(now=1.0)["force_holds"] == 1
    assert state.status(now=1.0)["watchdog_holds"] == 0

    released = state.tick(now=1.0, force_hold=False)
    assert released.joint_targets == pytest.approx((0.05,) * 7)
    assert released.gripper_closed is True
    assert state.status(now=1.0)["force_holding"] is False


def test_dual_controller_advances_both_arms_when_both_are_fresh() -> None:
    state = dual_controller(control_hz=20.0, max_joint_speed=1.0, watchdog_timeout=0.3)
    assert state.submit(command(0, [0.5] * 7, session_id="s:left"), arm="left", received_at=1.0)
    assert state.submit(command(0, [0.5] * 7, session_id="s:right"), arm="right", received_at=1.0)
    output = state.tick(now=1.0)
    assert output.coupled_hold is False
    assert output.left.joint_targets == pytest.approx((0.05,) * 7)
    assert output.right.joint_targets == pytest.approx((0.05,) * 7)
    assert state.status(now=1.0)["coupled_holds"] == 0


def test_dual_controller_freezes_the_fresh_arm_when_its_partner_goes_stale() -> None:
    state = dual_controller(control_hz=20.0, max_joint_speed=1.0, watchdog_timeout=0.3)
    state.submit(command(0, [0.5] * 7, session_id="s:left"), arm="left", received_at=1.0)
    state.submit(command(0, [0.5] * 7, session_id="s:right"), arm="right", received_at=1.0)
    first = state.tick(now=1.0)

    assert state.submit(command(1, [0.5] * 7, session_id="s:left"), arm="left", received_at=1.25)
    second = state.tick(now=1.35)

    assert second.coupled_hold is True
    assert second.left.joint_targets == first.left.joint_targets
    assert second.right.joint_targets == first.right.joint_targets
    assert second.left.requested_joint_targets == (0.5,) * 7
    status = state.status(now=1.35)
    assert status["left"]["force_holding"] is True
    assert status["left"]["watchdog_holding"] is False
    assert status["right"]["watchdog_holding"] is True
    assert status["coupled_holds"] == 1


def test_dual_controller_holds_both_arms_once_everything_is_stale() -> None:
    state = dual_controller(control_hz=20.0, max_joint_speed=1.0, watchdog_timeout=0.3)
    state.submit(command(0, [0.5] * 7, session_id="s:left"), arm="left", received_at=1.0)
    state.submit(command(0, [0.5] * 7, session_id="s:right"), arm="right", received_at=1.0)
    first = state.tick(now=1.0)
    second = state.tick(now=1.5)
    third = state.tick(now=2.0)
    assert second.coupled_hold is True and third.coupled_hold is True
    assert second.left.joint_targets == third.left.joint_targets == first.left.joint_targets
    assert state.status(now=2.0)["coupled_holds"] == 1
    assert state.status(now=2.0)["left"]["watchdog_holds"] == 1
    assert state.status(now=2.0)["right"]["watchdog_holds"] == 1


def test_dual_controller_clips_each_arm_against_its_own_limits() -> None:
    left = GelloCommandController([0.0] * 7, [(-1.0, 1.0)] * 7, max_joint_speed=100.0)
    right = GelloCommandController([0.0] * 7, [(-2.0, 2.0)] * 7, max_joint_speed=100.0)
    state = DualGelloCommandController(left, right)
    state.submit(command(0, [5.0] * 7, session_id="s:left"), arm="left", received_at=1.0)
    state.submit(command(0, [5.0] * 7, session_id="s:right"), arm="right", received_at=1.0)
    output = state.tick(now=1.0)
    assert output.left.joint_targets == (1.0,) * 7
    assert output.right.joint_targets == (2.0,) * 7


def test_dual_controller_rejects_unknown_arm() -> None:
    state = dual_controller()
    with pytest.raises(ValueError, match="unknown arm"):
        state.submit(command(0, [0.0] * 7), arm="both", received_at=1.0)


def test_dual_controller_release_session_clears_only_the_matching_arm_id() -> None:
    state = dual_controller(control_hz=20.0, max_joint_speed=1.0, watchdog_timeout=0.3)
    state.submit(command(0, [0.5] * 7, session_id="s:left"), arm="left", received_at=1.0)
    state.submit(command(0, [0.5] * 7, session_id="s:right"), arm="right", received_at=1.0)
    state.tick(now=1.0)
    state.release_session("s:left")
    status = state.status(now=1.0)
    assert status["left"]["session_id"] is None
    assert status["right"]["session_id"] == "s:right"


@pytest.mark.parametrize("arm", ["left", "right"])
def test_arm_session_id_round_trips(arm: str) -> None:
    assert parse_arm_session_id(arm_session_id("episode-7", arm)) == ("episode-7", arm)


@pytest.mark.parametrize(
    "session_id",
    ["legacy", "episode-7", "episode-7:both", ":left", "episode-7:", "a:b:left:x"],
)
def test_parse_arm_session_id_rejects_unsuffixed_or_unknown(session_id: str) -> None:
    assert parse_arm_session_id(session_id) is None


def test_parse_arm_session_id_keeps_colons_inside_the_base() -> None:
    assert parse_arm_session_id("ns:episode:1:right") == ("ns:episode:1", "right")


def test_arm_session_id_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        arm_session_id("episode-7", "both")
