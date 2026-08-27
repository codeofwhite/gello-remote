import importlib
import math
from types import SimpleNamespace

import pytest

from gello_remote.backends import IsaacLabFrankaBackend, MockFrankaBackend, validate_joint_targets


def test_validate_joint_targets_accepts_exactly_seven_finite_values() -> None:
    assert validate_joint_targets(range(7)) == tuple(float(value) for value in range(7))


@pytest.mark.parametrize("targets", [[], [0.0] * 6, [0.0] * 8])
def test_validate_joint_targets_rejects_wrong_length(targets: list[float]) -> None:
    with pytest.raises(ValueError, match="exactly 7"):
        validate_joint_targets(targets)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_validate_joint_targets_rejects_non_finite_values(bad_value: float) -> None:
    targets = [0.0] * 7
    targets[3] = bad_value
    with pytest.raises(ValueError, match="finite"):
        validate_joint_targets(targets)


def test_mock_backend_stores_an_immutable_copy() -> None:
    targets = [0.1] * 7
    backend = MockFrankaBackend()
    backend.set_joint_targets(targets)
    targets[0] = 1.0
    assert backend.last_joint_targets == (0.1,) * 7


def test_isaaclab_smoke_module_imports_without_isaaclab() -> None:
    module = importlib.import_module("gello_remote.isaaclab_smoke")
    assert callable(module.main)


def test_isaaclab_backend_uses_seven_ordered_arm_joints(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor_calls: list[tuple[object, object, str]] = []
    fake_torch = SimpleNamespace(
        float32="float32",
        tensor=lambda values, *, dtype, device: tensor_calls.append((values, dtype, device)) or "tensor",
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    class FakeArticulation:
        target_call: tuple[object, list[int]] | None = None
        writes = 0

        def find_joints(self, names: list[str], *, preserve_order: bool) -> tuple[list[int], list[str]]:
            assert preserve_order is True
            return list(range(7)), names

        def set_joint_position_target_index(self, *, target: object, joint_ids: list[int]) -> None:
            self.target_call = (target, joint_ids)

        def write_data_to_sim(self) -> None:
            self.writes += 1

    articulation = FakeArticulation()
    backend = IsaacLabFrankaBackend(articulation, device="cuda:3")
    backend.set_joint_targets([0.25] * 7)

    assert tensor_calls == [([(0.25,) * 7], "float32", "cuda:3")]
    assert articulation.target_call == ("tensor", list(range(7)))
    assert articulation.writes == 1
