from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f

from .config import NetworkConfig


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "realize"):
        value = value.realize().numpy()
    return np.array(value)


class TorchSEBlock(torch_nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(8, channels // max(1, reduction))
        self.fc1 = torch_nn.Linear(channels, hidden)
        self.fc2 = torch_nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        pooled = x.mean(dim=(2, 3))
        weights = torch.sigmoid(self.fc2(self.fc1(torch_f.relu(pooled))))
        return x * weights.reshape(batch, channels, 1, 1)


class TorchResidualBlock(torch_nn.Module):
    def __init__(self, channels: int, se_reduction: int) -> None:
        super().__init__()
        self.conv1 = torch_nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch_nn.BatchNorm2d(channels)
        self.conv2 = torch_nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch_nn.BatchNorm2d(channels)
        self.se = TorchSEBlock(channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch_f.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return torch_f.relu(x + residual)


class PolicyValueNet(torch_nn.Module):
    def __init__(self, board_size: int, cfg: NetworkConfig) -> None:
        super().__init__()
        if board_size != 9:
            raise ValueError("PolicyValueNet is fixed to 9x9")
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch_f.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.tower:
            x = block(x)

        policy = torch_f.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(policy.reshape(policy.shape[0], -1))

        value = torch_f.relu(self.value_bn(self.value_conv(x)))
        value = torch_f.tanh(self.value_fc2(torch_f.relu(self.value_fc1(value.reshape(value.shape[0], -1)))))
        return policy_logits, value.reshape(value.shape[0])


# Backward compatibility alias used in plan wording and some runtime code paths.
TorchPolicyValueNet = PolicyValueNet


def load_legacy_state_dict(model: PolicyValueNet, legacy_state: Mapping[str, Any]) -> None:
    target_state = model.state_dict()
    source_state = {key: _as_numpy(value) for key, value in legacy_state.items()}
    missing = sorted(set(target_state) - set(source_state))
    extra = sorted(set(source_state) - set(target_state))
    if missing or extra:
        raise RuntimeError(f"legacy/torch state mismatch: missing={missing} extra={extra}")

    converted: dict[str, torch.Tensor] = {}
    for key, target_tensor in target_state.items():
        source = source_state[key]
        dtype = target_tensor.dtype
        if dtype == torch.long:
            converted[key] = torch.as_tensor(source, dtype=dtype)
        else:
            converted[key] = torch.as_tensor(source, dtype=dtype)
    model.load_state_dict(converted, strict=True)


def clone_model(model: PolicyValueNet) -> PolicyValueNet:
    return copy.deepcopy(model)
