from __future__ import annotations

import numpy as np
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict

from .config import NetworkConfig


class SEBlock:
    def __init__(self, channels: int, reduction: int) -> None:
        hidden = max(8, channels // max(1, reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def __call__(self, x: Tensor) -> Tensor:
        batch, channels, _, _ = x.shape
        weights = self.fc2(self.fc1(x.mean(axis=(2, 3))).relu()).sigmoid()
        return x * weights.reshape(batch, channels, 1, 1)


class ResidualBlock:
    def __init__(self, channels: int, se_reduction: int) -> None:
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_reduction)

    def __call__(self, x: Tensor) -> Tensor:
        residual = x
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return (x + residual).relu()


class PolicyValueNet:
    def __init__(self, board_size: int, cfg: NetworkConfig) -> None:
        if board_size != 9:
            raise ValueError("PolicyValueNet is fixed to 9x9")
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.config = cfg
        channels = cfg.channels

        self.stem_conv = nn.Conv2d(cfg.input_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.tower = [ResidualBlock(channels, cfg.se_reduction) for _ in range(cfg.blocks)]

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.action_size, self.action_size)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.action_size, cfg.value_hidden)
        self.value_fc2 = nn.Linear(cfg.value_hidden, 1)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem_bn(self.stem_conv(x)).relu()
        for block in self.tower:
            x = block(x)

        policy = self.policy_bn(self.policy_conv(x)).relu()
        policy_logits = self.policy_fc(policy.reshape(policy.shape[0], -1))

        value = self.value_bn(self.value_conv(x)).relu()
        value = self.value_fc2(self.value_fc1(value.reshape(value.shape[0], -1)).relu()).tanh()
        return policy_logits, value.reshape(value.shape[0])

    def parameters(self) -> list[Tensor]:
        return get_parameters(self)

    def state_dict(self) -> dict[str, Tensor]:
        return get_state_dict(self)


def clone_model(model: PolicyValueNet) -> PolicyValueNet:
    cloned = PolicyValueNet(model.board_size, model.config)
    copied = {
        key: Tensor(np.array(value.realize().numpy(), copy=True))
        for key, value in model.state_dict().items()
    }
    load_state_dict(cloned, copied, strict=True, verbose=False)
    return cloned

