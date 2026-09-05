import numpy as np

from gello_remote.openpi_policy import decode_policy_action, validate_action_chunk


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
