from __future__ import annotations

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
        return 1 << (max(batch_size, 1) - 1).bit_length()

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        features = states_to_feature_planes(states)
        return self.evaluate_features(features)

    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(features.shape[0])
        bucket = self.effective_batch_size(n)
        if bucket > n:
            pad_shape = (bucket - n, *features.shape[1:])
            features = np.concatenate(
                [features, np.zeros(pad_shape, dtype=features.dtype)],
                axis=0,
            )

        array = np.ascontiguousarray(features)
        with torch.inference_mode():
            tensor = torch.as_tensor(array, device=self.device)
            logits, values = self.model(tensor)
            priors = torch.softmax(logits, dim=1).detach().cpu().numpy()
            value_np = values.detach().cpu().numpy()

        if bucket > n:
            priors = priors[:n]
            value_np = value_np[:n]
        return priors.astype(np.float32, copy=False), value_np.astype(np.float32, copy=False)


def build_evaluator(model: object, *, backend: str, device: str | None) -> Evaluator:
    token = backend.strip().lower()
    if token in {"torch", "auto"}:
        return TorchModelEvaluator(model, device=device)
    raise ValueError(f"unsupported selfplay.evaluator_backend: {backend!r}; use 'torch'")
