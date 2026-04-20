from __future__ import annotations

import numpy as np
from loguru import logger

from .board import GameState
from .config import NetworkConfig
from .evaluator import Evaluator
from .features import states_to_feature_planes
from .torch_network import PolicyValueNet as TorchPolicyValueNet, load_tinygrad_state_dict

try:
    import torch
    import torch.nn as torch_nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed.
    torch = None
    torch_nn = None

try:
    from tinygrad import Tensor
except ModuleNotFoundError:
    Tensor = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "selfplay.evaluator_backend='torch' requires PyTorch. "
            "Install torch or run with `uv run --extra omok ...`."
        )


def _torch_device(tinygrad_device: str | None) -> "torch.device":
    _require_torch()
    name = (tinygrad_device or "auto").upper()
    if name == "CUDA":
        if not torch.cuda.is_available():
            raise RuntimeError("selfplay.evaluator_backend='torch' requested CUDA, but torch.cuda is unavailable")
        return torch.device("cuda")
    if name in {"CPU", "METAL", "GPU"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _tinygrad_state(model: object) -> dict[str, np.ndarray]:
    if Tensor is None or not hasattr(model, "state_dict"):
        raise RuntimeError("Expected a tinygrad-compatible model with state_dict()")
    return {key: np.array(value.realize().numpy(), copy=True) for key, value in model.state_dict().items()}


def _coerce_torch_model(model: object, device: "torch.device") -> TorchPolicyValueNet:
    _require_torch()
    if isinstance(model, TorchPolicyValueNet):
        return model

    if isinstance(model, torch_nn.Module):
        return model  # type: ignore[return-value]

    if not hasattr(model, "state_dict"):
        raise TypeError("Unsupported model type for torch evaluator")

    # Legacy tinygrad model: keep the same board/network architecture.
    board_size = int(getattr(model, "board_size"))
    config = getattr(model, "config")
    if not isinstance(config, NetworkConfig):
        raise TypeError("Model config must be a NetworkConfig for torch conversion")
    torch_model = TorchPolicyValueNet(board_size=board_size, cfg=config)
    load_tinygrad_state_dict(torch_model, _tinygrad_state(model))
    return torch_model


class TorchModelEvaluator(Evaluator):
    def __init__(self, model: object, device: str | None = None) -> None:
        _require_torch()
        self.device = _torch_device(device)
        self.model = _coerce_torch_model(model, self.device)
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
    if token == "tinygrad":
        if Tensor is None:
            raise RuntimeError("selfplay.evaluator_backend='tinygrad' requires tinygrad.")
        from .network import PolicyValueNet as TinyPolicyValueNet
        from .evaluator import ModelEvaluator

        if not isinstance(model, TinyPolicyValueNet):
            raise TypeError("tinygrad evaluator expects tinygrad PolicyValueNet")
        return ModelEvaluator(model, device=device)
    if token == "torch":
        return TorchModelEvaluator(model, device=device)
    if token == "auto":
        if (device or "").upper() == "CUDA":
            return TorchModelEvaluator(model, device=device)
        if torch is not None and torch.cuda.is_available():
            return TorchModelEvaluator(model, device=device)
        from .evaluator import ModelEvaluator
        from .network import PolicyValueNet as TinyPolicyValueNet

        if not isinstance(model, TinyPolicyValueNet):
            raise RuntimeError("auto evaluator selected tinygrad, but model is not tinygrad network")
        return ModelEvaluator(model, device=device)
    raise ValueError(f"unsupported selfplay.evaluator_backend: {backend!r}")
