from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from ..game import GameState
from .config import RunConfig, config_from_dict
from .encoding import encode_information_state, infer_input_dim, legal_mask_array
from .evaluate import make_opponent
from .networks import AdvantageNet, StrategyNet
from .trainer import _torch_device, set_seed


@dataclass(frozen=True)
class PolicyGradientResult:
    output_path: Path
    games: int
    wins: int
    losses: int
    draws: int
    avg_diff: float
    avg_loss: float
    avg_kl: float
    avg_entropy: float


def _masked_dist(logits: torch.Tensor, legal_mask: np.ndarray) -> torch.distributions.Categorical:
    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.distributions.Categorical(logits=masked)


def _policy_kl_to_anchor(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    legal_mask: np.ndarray,
) -> torch.Tensor:
    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    anchor_masked = anchor_logits.masked_fill(~mask, torch.finfo(anchor_logits.dtype).min)
    log_probs = F.log_softmax(masked, dim=-1)
    anchor_probs = F.softmax(anchor_masked, dim=-1)
    anchor_log_probs = F.log_softmax(anchor_masked, dim=-1)
    return (anchor_probs * (anchor_log_probs - log_probs)).masked_fill(~mask, 0.0).sum()


def fine_tune_policy_gradient_checkpoint(
    config: RunConfig,
    *,
    checkpoint_path: Path,
    output_path: Path,
    games: int,
    opponent: str = "safe_heuristic",
    max_steps: int = 1000,
    learning_rate: float = 1.0e-6,
    reward_scale: float = 100.0,
    reward_clip: float = 2.0,
    kl_coef: float = 0.05,
    entropy_coef: float = 0.001,
    grad_clip: float = 0.5,
    seed: int | None = None,
) -> PolicyGradientResult:
    if games <= 0:
        raise ValueError("games must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    if reward_clip <= 0.0:
        raise ValueError("reward_clip must be positive")

    train_seed = config.seed + 300_000 if seed is None else seed
    set_seed(train_seed)
    rng = np.random.default_rng(train_seed)
    device = _torch_device(config.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = config_from_dict(checkpoint["config"])
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    input_dim = infer_input_dim(lc_config)
    action_size = lc_config.action_size
    if int(checkpoint["input_dim"]) != input_dim or int(checkpoint["action_size"]) != action_size:
        raise ValueError(
            "checkpoint rules/action space do not match the fine-tune config: "
            f"checkpoint input_dim={checkpoint['input_dim']} action_size={checkpoint['action_size']}, "
            f"config input_dim={input_dim} action_size={action_size}"
        )

    strategy_net = StrategyNet(input_dim, action_size, checkpoint_config.network).to(device)
    anchor_net = StrategyNet(input_dim, action_size, checkpoint_config.network).to(device)
    strategy_net.load_state_dict(checkpoint["strategy_net"])
    anchor_net.load_state_dict(checkpoint["strategy_net"])
    anchor_net.eval()
    for parameter in anchor_net.parameters():
        parameter.requires_grad_(False)

    advantage_nets = [
        AdvantageNet(input_dim, action_size, checkpoint_config.network).to(device),
        AdvantageNet(input_dim, action_size, checkpoint_config.network).to(device),
    ]
    for net, state_dict in zip(advantage_nets, checkpoint["advantage_nets"], strict=True):
        net.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(
        strategy_net.parameters(),
        lr=learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    wins = losses = draws = 0
    diffs: list[int] = []
    losses_seen: list[float] = []
    kls_seen: list[float] = []
    entropies_seen: list[float] = []
    for game_index in range(games):
        state = GameState.new_game(lc_config, seed=train_seed + game_index)
        policy_player = game_index % 2
        opponent_bot = make_opponent(opponent, seed=int(rng.integers(0, 2**31 - 1)))
        log_probs: list[torch.Tensor] = []
        kls: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        timed_out = True

        for _ in range(max_steps):
            if state.terminal:
                timed_out = False
                break
            if state.current_player != policy_player:
                state.apply_action(opponent_bot.act(state))
                continue

            info = encode_information_state(state, policy_player)
            legal = legal_mask_array(state)
            x = torch.as_tensor(info, dtype=torch.float32, device=device).unsqueeze(0)
            logits = strategy_net(x).squeeze(0)
            with torch.no_grad():
                anchor_logits = anchor_net(x).squeeze(0)
            dist = _masked_dist(logits, legal)
            unified_action = dist.sample()
            log_probs.append(dist.log_prob(unified_action))
            entropies.append(dist.entropy())
            kls.append(_policy_kl_to_anchor(logits, anchor_logits, legal))
            state.apply_unified_action(int(unified_action.detach().cpu()))

        diff = state.score_diff(policy_player)
        if timed_out and not state.terminal:
            # Treat timeout as the current score difference, matching eval semantics.
            diff = state.score_diff(policy_player)
        diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1

        if not log_probs:
            continue
        reward = float(np.clip(diff / reward_scale, -reward_clip, reward_clip))
        log_prob_term = torch.stack(log_probs).mean()
        kl_term = torch.stack(kls).mean()
        entropy_term = torch.stack(entropies).mean()
        loss = -(reward * log_prob_term) + kl_coef * kl_term - entropy_coef * entropy_term

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), grad_clip)
        optimizer.step()

        losses_seen.append(float(loss.detach().cpu()))
        kls_seen.append(float(kl_term.detach().cpu()))
        entropies_seen.append(float(entropy_term.detach().cpu()))
        if (game_index + 1) % 100 == 0:
            logger.info(
                "Policy gradient fine-tune progress: games={} win_rate={:.3f} avg_diff={:.2f} avg_loss={:.4f} avg_kl={:.4f}",
                game_index + 1,
                wins / max(1, game_index + 1),
                float(np.mean(diffs)),
                float(np.mean(losses_seen)) if losses_seen else 0.0,
                float(np.mean(kls_seen)) if kls_seen else 0.0,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "lost_cities_config": lc_config.to_snapshot(),
            "resume_semantics": "networks_optimizers_iteration_only",
            "iteration": int(checkpoint.get("iteration", 0)),
            "input_dim": input_dim,
            "action_size": action_size,
            "advantage_nets": [net.state_dict() for net in advantage_nets],
            "strategy_net": strategy_net.state_dict(),
            "advantage_optimizers": checkpoint.get("advantage_optimizers", []),
            "strategy_optimizer": optimizer.state_dict(),
            "metrics": {
                "policy_gradient_source_checkpoint": str(checkpoint_path),
                "policy_gradient_games": games,
                "policy_gradient_opponent": opponent,
                "policy_gradient_wins": wins,
                "policy_gradient_losses": losses,
                "policy_gradient_draws": draws,
                "policy_gradient_avg_diff": float(np.mean(diffs)) if diffs else 0.0,
                "policy_gradient_avg_loss": float(np.mean(losses_seen)) if losses_seen else 0.0,
                "policy_gradient_avg_kl": float(np.mean(kls_seen)) if kls_seen else 0.0,
                "policy_gradient_avg_entropy": float(np.mean(entropies_seen)) if entropies_seen else 0.0,
            },
        },
        output_path,
    )
    logger.info("Saved policy-gradient fine-tuned checkpoint: {}", output_path)
    return PolicyGradientResult(
        output_path=output_path,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        avg_diff=float(np.mean(diffs)) if diffs else 0.0,
        avg_loss=float(np.mean(losses_seen)) if losses_seen else 0.0,
        avg_kl=float(np.mean(kls_seen)) if kls_seen else 0.0,
        avg_entropy=float(np.mean(entropies_seen)) if entropies_seen else 0.0,
    )
