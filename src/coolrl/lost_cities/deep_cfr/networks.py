from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import NetworkConfig


def _activation(name: str) -> nn.Module:
    token = name.lower()
    if token == "relu":
        return nn.ReLU()
    if token == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name!r}")


class _MLP(nn.Module):
    def __init__(self, input_dim: int, action_size: int, cfg: NetworkConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(cfg.num_layers):
            layers.append(nn.Linear(last_dim, cfg.hidden_size))
            layers.append(_activation(cfg.activation))
            last_dim = cfg.hidden_size
        layers.append(nn.Linear(last_dim, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvantageNet(_MLP):
    """Predicts instantaneous regrets/advantages for unified actions."""


class StrategyNet(_MLP):
    """Predicts unnormalized policy logits for unified actions."""


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = legal_mask.to(dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked, dim=dim)
    return torch.where(mask, probs, torch.zeros_like(probs))


def _regret_matching_torch(
    advantages: torch.Tensor,
    legal_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    mask = legal_mask.to(dtype=torch.bool, device=advantages.device)
    positive = torch.clamp(advantages, min=0.0).masked_fill(~mask, 0.0)
    denom = positive.sum(dim=-1, keepdim=True)
    legal_count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform = mask.to(dtype=advantages.dtype) / legal_count.to(dtype=advantages.dtype)
    return torch.where(denom > epsilon, positive / denom.clamp_min(epsilon), uniform)


def regret_matching(
    advantages: np.ndarray | torch.Tensor,
    legal_mask: np.ndarray | torch.Tensor,
    epsilon: float = 1.0e-8,
) -> np.ndarray | torch.Tensor:
    """Return a regret-matched policy with the same array type as the input."""
    if isinstance(advantages, torch.Tensor):
        mask = legal_mask if isinstance(legal_mask, torch.Tensor) else torch.as_tensor(legal_mask)
        return _regret_matching_torch(advantages, mask, epsilon)
    adv = np.asarray(advantages, dtype=np.float32)
    mask_np = np.asarray(legal_mask, dtype=bool)
    positive = np.where(mask_np, np.maximum(adv, 0.0), 0.0)
    total = float(positive.sum())
    if total > epsilon:
        return (positive / total).astype(np.float32)
    legal_count = int(mask_np.sum())
    if legal_count == 0:
        return np.zeros_like(adv, dtype=np.float32)
    return (mask_np.astype(np.float32) / legal_count).astype(np.float32)
