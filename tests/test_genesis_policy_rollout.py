import numpy as np
import pytest

from gello_remote.genesis_policy_rollout import (
    decode_policy_action,
    encode_policy_state,
    validate_action_chunk,
)


def test_policy_state_uses_closed_fraction() -> None:
    native = np.zeros(18, dtype=np.float32)
    native[7:9] = 0.04
    assert encode_policy_state(native).tolist() == pytest.approx([0.0] * 8)
    native[7:9] = 0.0
    assert encode_policy_state(native)[7] == pytest.approx(1.0)


def test_policy_action_decodes_gripper_and_preserves_joint_request() -> None:
    action = np.asarray([0.1] * 7 + [0.75], dtype=np.float32)
    joints, closed, requested_native = decode_policy_action(action)
    assert joints.tolist() == pytest.approx([0.1] * 7)
    assert closed is True
    assert requested_native[:7].tolist() == pytest.approx([0.1] * 7)
    assert requested_native[7:].tolist() == pytest.approx([0.01, 0.01])


def test_action_chunk_rejects_wrong_or_nonfinite_values() -> None:
    assert validate_action_chunk(np.zeros((10, 8)), replan_steps=5).shape == (5, 8)
    with pytest.raises(ValueError, match="shape"):
        validate_action_chunk(np.zeros((4, 8)), replan_steps=5)
    invalid = np.zeros((5, 8))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_action_chunk(invalid, replan_steps=5)
