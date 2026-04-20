from __future__ import annotations

import numpy as np
from loguru import logger
from tinygrad import Tensor

from .board import GameState
from .config import NetworkConfig
from .evaluator import Evaluator, ModelEvaluator
from .features import states_to_feature_planes
from .network import PolicyValueNet


try:
    import torch
    import torch.nn as torch_nn
    import torch.nn.functional as torch_f
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed.
    torch = None
    torch_nn = None
    torch_f = None


_TorchModuleBase = object if torch_nn is None else torch_nn.Module


class TorchSEBlock(_TorchModuleBase):  # type: ignore[misc, valid-type]
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(8, channels // max(1, reduction))
        self.fc1 = torch_nn.Linear(channels, hidden)
        self.fc2 = torch_nn.Linear(hidden, channels)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch, channels, _, _ = x.shape
        weights = torch.sigmoid(self.fc2(torch_f.relu(self.fc1(x.mean(dim=(2, 3))))))
        return x * weights.reshape(batch, channels, 1, 1)


class TorchResidualBlock(_TorchModuleBase):  # type: ignore[misc, valid-type]
    def __init__(self, channels: int, se_reduction: int) -> None:
        super().__init__()
        self.conv1 = torch_nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch_nn.BatchNorm2d(channels)
        self.conv2 = torch_nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch_nn.BatchNorm2d(channels)
        self.se = TorchSEBlock(channels, se_reduction)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = x
        x = torch_f.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return torch_f.relu(x + residual)


class TorchPolicyValueNet(_TorchModuleBase):  # type: ignore[misc, valid-type]
    def __init__(self, board_size: int, cfg: NetworkConfig) -> None:
        super().__init__()
        if board_size != 9:
            raise ValueError("TorchPolicyValueNet is fixed to 9x9")
        action_size = board_size * board_size
        channels = cfg.channels

        self.stem_conv = torch_nn.Conv2d(cfg.input_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = torch_nn.BatchNorm2d(channels)
        self.tower = torch_nn.ModuleList(
            [TorchResidualBlock(channels, cfg.se_reduction) for _ in range(cfg.blocks)]
        )

        self.policy_conv = torch_nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = torch_nn.BatchNorm2d(2)
        self.policy_fc = torch_nn.Linear(2 * action_size, action_size)

        self.value_conv = torch_nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = torch_nn.BatchNorm2d(1)
        self.value_fc1 = torch_nn.Linear(action_size, cfg.value_hidden)
        self.value_fc2 = torch_nn.Linear(cfg.value_hidden, 1)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        x = torch_f.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.tower:
            x = block(x)

        policy = torch_f.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(policy.reshape(policy.shape[0], -1))

        value = torch_f.relu(self.value_bn(self.value_conv(x)))
        value = torch.tanh(self.value_fc2(torch_f.relu(self.value_fc1(value.reshape(value.shape[0], -1)))))
        return policy_logits, value.reshape(value.shape[0])


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


def _numpy_state(model: PolicyValueNet) -> dict[str, np.ndarray]:
    return {
        key: np.array(value.realize().numpy(), copy=True)
        for key, value in model.state_dict().items()
        if isinstance(value, Tensor)
    }


def _load_tinygrad_state(torch_model: TorchPolicyValueNet, tinygrad_model: PolicyValueNet) -> None:
    source = _numpy_state(tinygrad_model)
    target = torch_model.state_dict()
    missing = sorted(set(target) - set(source))
    extra = sorted(set(source) - set(target))
    if missing or extra:
        raise RuntimeError(f"tinygrad/torch model state mismatch: missing={missing} extra={extra}")

    converted = {}
    for key, target_tensor in target.items():
        array = source[key]
        if target_tensor.dtype == torch.long:
            converted[key] = torch.as_tensor(array, dtype=torch.long)
        else:
            converted[key] = torch.as_tensor(array, dtype=target_tensor.dtype)
    torch_model.load_state_dict(converted, strict=True)


class TorchModelEvaluator(Evaluator):
    def __init__(self, model: PolicyValueNet, device: str | None = None) -> None:
        _require_torch()
        self.device = _torch_device(device)
        self.model = TorchPolicyValueNet(model.board_size, model.config).to(self.device)
        _load_tinygrad_state(self.model, model)
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


def build_evaluator(model: PolicyValueNet, *, backend: str, device: str | None) -> Evaluator:
    token = backend.strip().lower()
    if token == "tinygrad":
        return ModelEvaluator(model, device=device)
    if token == "torch":
        return TorchModelEvaluator(model, device=device)
    if token == "auto":
        if (device or "").upper() == "CUDA":
            return TorchModelEvaluator(model, device=device)
        return ModelEvaluator(model, device=device)
    raise ValueError(f"unsupported selfplay.evaluator_backend: {backend!r}")
