from __future__ import annotations

import importlib.util

import numpy as np
from loguru import logger

from .board import GameState
from .evaluator import Evaluator
from .features import states_to_feature_planes
from .torch_network import PolicyValueNet as TorchPolicyValueNet

try:
    import torch
    import torch.nn as torch_nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed.
    torch = None
    torch_nn = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for Omok evaluation. "
            "Install torch or run with `uv run --extra omok ...`."
        )


def _torch_device(requested_device: str | None) -> "torch.device":
    _require_torch()
    name = (requested_device or "auto").upper()
    if name == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for torch evaluator, but torch.cuda is unavailable")
        return torch.device("cuda")
    if name in {"METAL", "GPU"}:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Metal/MPS requested for torch evaluator, but torch.backends.mps is unavailable")
    if name == "CPU":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _coerce_torch_model(model: object) -> TorchPolicyValueNet:
    _require_torch()
    if isinstance(model, TorchPolicyValueNet):
        return model
    if isinstance(model, torch_nn.Module):
        return model  # type: ignore[return-value]
    raise TypeError("torch evaluator expects a PyTorch nn.Module")


def _cuda_requested_or_available(device: str | None) -> bool:
    _require_torch()
    requested = (device or "auto").upper()
    if requested in {"CPU", "METAL", "GPU"}:
        return False
    if requested == "CUDA":
        return torch.cuda.is_available()
    return torch.cuda.is_available()


def _tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


class TorchModelEvaluator(Evaluator):
    def __init__(self, model: object, device: str | None = None) -> None:
        _require_torch()
        self.device = _torch_device(device)
        self.model = _coerce_torch_model(model)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        logger.info("Torch evaluator initialized: device={}", self.device)

    def effective_batch_size(self, batch_size: int) -> int:
        return batch_size

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        features = states_to_feature_planes(states)
        return self.evaluate_features(features)

    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.ascontiguousarray(features)
        with torch.inference_mode():
            tensor = torch.as_tensor(array, device=self.device)
            logits, values = self.model(tensor)
            priors = torch.softmax(logits, dim=1).detach().cpu().numpy()
            value_np = values.detach().cpu().numpy()

        return priors.astype(np.float32, copy=False), value_np.astype(np.float32, copy=False)


def build_evaluator(model: object, *, backend: str, device: str | None) -> Evaluator:
    token = backend.strip().lower()
    if token == "auto":
        if _cuda_requested_or_available(device) and _tensorrt_available():
            try:
                from .tensorrt_evaluator import TensorRTModelEvaluator

                return TensorRTModelEvaluator(model, device=device)
            except Exception as exc:
                logger.warning("TensorRT evaluator unavailable; falling back to torch: {}", exc)
        return TorchModelEvaluator(model, device=device)
    if token == "torch":
        return TorchModelEvaluator(model, device=device)
    if token in {"tensorrt", "trt"}:
        from .tensorrt_evaluator import TensorRTModelEvaluator

        return TensorRTModelEvaluator(model, device=device)
    raise ValueError(
        f"unsupported selfplay.evaluator_backend: {backend!r}; use 'torch', 'tensorrt', or 'auto'"
    )
