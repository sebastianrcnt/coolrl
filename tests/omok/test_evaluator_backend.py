from __future__ import annotations

import pytest

from coolrl.omok.config import NetworkConfig
from coolrl.omok.torch_evaluator import TorchModelEvaluator, build_evaluator
from coolrl.omok.torch_network import PolicyValueNet


def _tiny_model() -> PolicyValueNet:
    return PolicyValueNet(9, NetworkConfig(channels=8, blocks=1, value_hidden=16))


def test_auto_evaluator_falls_back_to_torch_on_cpu() -> None:
    evaluator = build_evaluator(_tiny_model(), backend="auto", device="CPU")

    assert isinstance(evaluator, TorchModelEvaluator)


def test_tensorrt_evaluator_rejects_cpu_before_runtime_build() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        build_evaluator(_tiny_model(), backend="tensorrt", device="CPU")


def test_unknown_evaluator_backend_lists_supported_backends() -> None:
    with pytest.raises(ValueError, match="torch.*tensorrt.*auto"):
        build_evaluator(_tiny_model(), backend="bad-backend", device="CPU")
