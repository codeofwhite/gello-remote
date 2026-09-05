import numpy as np

from gello_remote.openpi_policy import (
    GripperCommandFilter,
    decode_policy_action,
    validate_action_chunk,
)


def test_decode_policy_action_maps_closed_fraction_to_native_fingers():
    joints, closed, native = decode_policy_action(np.asarray([1, 2, 3, 4, 5, 6, 7, 1]))

    np.testing.assert_allclose(joints, np.arange(1, 8))
    assert closed is True
    np.testing.assert_allclose(native[-2:], 0.0)


def test_validate_action_chunk_rejects_non_finite_and_short_chunks():
    for actions in (np.zeros((4, 8)), np.full((5, 8), np.nan)):
        try:
            validate_action_chunk(actions, replan_steps=5)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid policy chunk to fail")


def test_single_threshold_filter_preserves_current_behavior():
    command = GripperCommandFilter()

    assert command.update(0.49) is False
    assert command.update(0.50) is True
    assert command.update(0.49) is False


def test_hysteresis_requires_explicit_release_after_minimum_hold():
    command = GripperCommandFilter(
        mode="hysteresis",
        close_threshold=0.65,
        open_threshold=0.35,
        min_closed_frames=3,
    )

    assert command.update(0.65) is False
    assert command.update(0.66) is True
    assert command.update(0.10) is True
    assert command.update(0.10) is True
    assert command.update(0.34) is False
    assert command.update(0.50) is False


def test_hysteresis_reset_clears_episode_state():
    command = GripperCommandFilter(
        mode="hysteresis",
        close_threshold=0.65,
        open_threshold=0.35,
        min_closed_frames=5,
    )
    assert command.update(0.9) is True

    command.reset()

    assert command.closed is False
    assert command.update(0.5) is False
