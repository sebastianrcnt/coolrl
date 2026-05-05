from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from ..bots.heuristic import SafeHeuristicBot
from ..interfaces import LostCitiesBot
from ..game import GameState
from .config import RunConfig, config_from_dict
from .encoding import encode_information_state, infer_input_dim, legal_mask_array
from .evaluate import StrategyNetBot
from .networks import AdvantageNet, StrategyNet
from .trainer import _torch_device, set_seed


@dataclass(frozen=True)
class HeuristicPretrainResult:
    output_path: Path
    dataset_mode: str
    games: int
    states: int
    strategy_loss: float
    strategy_accuracy: float
    advantage_loss_p0: float
    advantage_loss_p1: float


def _split_counts(total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    if total <= 0:
        raise ValueError("total must be positive")
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError(f"ratios must be nonnegative, got {ratios}")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0.0:
        raise ValueError("at least one ratio must be positive")
    raw = [total * ratio / ratio_sum for ratio in ratios]
    counts = [int(value) for value in raw]
    remainders = sorted(
        ((raw[index] - counts[index], index) for index in range(len(ratios))),
        reverse=True,
    )
    for _, index in remainders[: total - sum(counts)]:
        counts[index] += 1
    return tuple(counts)


def _load_policy_behavior_bot(
    checkpoint_path: Path,
    config: RunConfig,
    *,
    device: torch.device,
    sample: bool,
    seed: int,
) -> StrategyNetBot:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = config_from_dict(checkpoint["config"])
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    input_dim = infer_input_dim(lc_config)
    action_size = lc_config.action_size
    if int(checkpoint["input_dim"]) != input_dim or int(checkpoint["action_size"]) != action_size:
        raise ValueError(
            "base checkpoint rules/action space do not match the pretrain config: "
            f"checkpoint input_dim={checkpoint['input_dim']} action_size={checkpoint['action_size']}, "
            f"config input_dim={input_dim} action_size={action_size}"
        )
    strategy_net = StrategyNet(input_dim, action_size, checkpoint_config.network).to(device)
    strategy_net.load_state_dict(checkpoint["strategy_net"])
    strategy_net.eval()
    return StrategyNetBot(
        strategy_net,
        lc_config,
        device=device,
        sample=sample,
        seed=seed,
    )


def _append_example(
    state: GameState,
    player: int,
    target_action: int,
    *,
    infos: list[np.ndarray],
    legal_masks: list[np.ndarray],
    target_actions: list[int],
    players: list[int],
) -> None:
    infos.append(encode_information_state(state, player))
    legal_masks.append(legal_mask_array(state))
    target_actions.append(state.to_unified_action(target_action))
    players.append(player)


def _collect_games_with_behavior(
    config: RunConfig,
    *,
    games: int,
    max_steps: int,
    seed: int,
    behavior_factories: tuple[tuple[str, Callable[[int], list[LostCitiesBot]]], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    infos: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    target_actions: list[int] = []
    players: list[int] = []
    expert = SafeHeuristicBot()
    for game_index in range(games):
        state = GameState.new_game(lc_config, seed=seed + game_index)
        _, factory = behavior_factories[game_index % len(behavior_factories)]
        bots = factory(seed + game_index)
        for _ in range(max_steps):
            if state.terminal:
                break
            player = state.current_player
            target_action = expert.act(state)
            action = bots[player].act(state)
            _append_example(
                state,
                player,
                target_action,
                infos=infos,
                legal_masks=legal_masks,
                target_actions=target_actions,
                players=players,
            )
            state.apply_action(action)

    if not infos:
        raise RuntimeError("safe heuristic pretraining produced no states")
    return (
        np.stack(infos).astype(np.float32),
        np.stack(legal_masks).astype(bool),
        np.asarray(target_actions, dtype=np.int64),
        np.asarray(players, dtype=np.int64),
    )


def _collect_safe_heuristic_dataset(
    config: RunConfig,
    *,
    games: int,
    max_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _collect_games_with_behavior(
        config,
        games=games,
        max_steps=max_steps,
        seed=seed,
        behavior_factories=(("safe_self_play", lambda _: [SafeHeuristicBot(), SafeHeuristicBot()]),),
    )


def _collect_aggregated_safe_heuristic_dataset(
    config: RunConfig,
    *,
    games: int,
    max_steps: int,
    seed: int,
    base_checkpoint: Path,
    device: torch.device,
    policy_sample: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    safe_games, policy_vs_safe_games, policy_self_play_games = _split_counts(games, (0.30, 0.40, 0.30))
    policy_template = _load_policy_behavior_bot(
        base_checkpoint,
        config,
        device=device,
        sample=policy_sample,
        seed=seed,
    )

    def policy_bot(game_seed: int) -> StrategyNetBot:
        return StrategyNetBot(
            policy_template.strategy_net,
            policy_template.config,
            device=device,
            sample=policy_sample,
            seed=game_seed,
        )

    def safe_self_play(_: int) -> list[LostCitiesBot]:
        return [SafeHeuristicBot(), SafeHeuristicBot()]

    def policy_vs_safe(game_seed: int) -> list[LostCitiesBot]:
        policy = policy_bot(game_seed)
        if game_seed % 2 == 0:
            return [policy, SafeHeuristicBot()]
        return [SafeHeuristicBot(), policy]

    def policy_self_play(game_seed: int) -> list[LostCitiesBot]:
        return [
            policy_bot(game_seed),
            policy_bot(game_seed + 100_000),
        ]

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    cursor = seed
    for label, count, factory in (
        ("safe_self_play", safe_games, safe_self_play),
        ("policy_vs_safe", policy_vs_safe_games, policy_vs_safe),
        ("policy_self_play", policy_self_play_games, policy_self_play),
    ):
        if count <= 0:
            continue
        logger.info("Collecting aggregated imitation states: mode={} games={}", label, count)
        parts.append(
            _collect_games_with_behavior(
                config,
                games=count,
                max_steps=max_steps,
                seed=cursor,
                behavior_factories=((label, factory),),
            )
        )
        cursor += count

    if not parts:
        raise RuntimeError("aggregated safe heuristic pretraining produced no states")
    return tuple(np.concatenate(values, axis=0) for values in zip(*parts))


def _collect_successful_policy_vs_safe_dataset(
    config: RunConfig,
    *,
    games: int,
    max_steps: int,
    seed: int,
    base_checkpoint: Path,
    device: torch.device,
    policy_sample: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    policy_template = _load_policy_behavior_bot(
        base_checkpoint,
        config,
        device=device,
        sample=policy_sample,
        seed=seed,
    )
    infos: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    target_actions: list[int] = []
    players: list[int] = []
    wins = 0
    timeouts = 0

    def policy_bot(game_seed: int) -> StrategyNetBot:
        return StrategyNetBot(
            policy_template.strategy_net,
            policy_template.config,
            device=device,
            sample=policy_sample,
            seed=game_seed,
        )

    for game_index in range(games):
        state = GameState.new_game(lc_config, seed=seed + game_index)
        policy_player = game_index % 2
        bots: list[LostCitiesBot]
        if policy_player == 0:
            bots = [policy_bot(seed + game_index), SafeHeuristicBot()]
        else:
            bots = [SafeHeuristicBot(), policy_bot(seed + game_index)]
        trajectory: list[tuple[GameState, int, int]] = []
        timed_out = True
        for _ in range(max_steps):
            if state.terminal:
                timed_out = False
                break
            player = state.current_player
            action = bots[player].act(state)
            if player == policy_player:
                trajectory.append((state.clone(), player, action))
            state.apply_action(action)
        if timed_out and not state.terminal:
            timeouts += 1
        diff = state.score_diff(policy_player)
        if diff <= 0:
            continue
        wins += 1
        for example_state, player, action in trajectory:
            _append_example(
                example_state,
                player,
                action,
                infos=infos,
                legal_masks=legal_masks,
                target_actions=target_actions,
                players=players,
            )

    if not infos:
        raise RuntimeError("successful policy-vs-safe collection produced no winning states")
    logger.info(
        "Collected successful policy-vs-safe examples: games={} wins={} timeouts={} states={}",
        games,
        wins,
        timeouts,
        len(target_actions),
    )
    return (
        np.stack(infos).astype(np.float32),
        np.stack(legal_masks).astype(bool),
        np.asarray(target_actions, dtype=np.int64),
        np.asarray(players, dtype=np.int64),
    )


def _advantage_targets(
    legal_masks: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    action_size: int,
) -> torch.Tensor:
    targets = torch.zeros((target_actions.shape[0], action_size), dtype=torch.float32)
    targets[legal_masks] = -1.0
    targets[torch.arange(target_actions.shape[0]), target_actions] = 1.0
    return targets


def pretrain_safe_heuristic_checkpoint(
    config: RunConfig,
    *,
    output_path: Path,
    games: int,
    epochs: int,
    batch_size: int,
    max_steps: int,
    seed: int | None = None,
    dataset_mode: str = "safe_self_play",
    base_checkpoint: Path | None = None,
    init_checkpoint: Path | None = None,
    policy_sample: bool = False,
) -> HeuristicPretrainResult:
    if games <= 0:
        raise ValueError("games must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    pretrain_seed = config.seed + 200_000 if seed is None else seed
    set_seed(pretrain_seed)
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    input_dim = infer_input_dim(lc_config)
    action_size = lc_config.action_size
    device = _torch_device(config.device)
    dataset_mode = dataset_mode.strip().lower()
    if dataset_mode == "safe_self_play":
        info_np, legal_np, actions_np, players_np = _collect_safe_heuristic_dataset(
            config,
            games=games,
            max_steps=max_steps,
            seed=pretrain_seed,
        )
    elif dataset_mode == "aggregated":
        if base_checkpoint is None:
            raise ValueError("base_checkpoint is required when dataset_mode='aggregated'")
        info_np, legal_np, actions_np, players_np = _collect_aggregated_safe_heuristic_dataset(
            config,
            games=games,
            max_steps=max_steps,
            seed=pretrain_seed,
            base_checkpoint=base_checkpoint,
            device=device,
            policy_sample=policy_sample,
        )
    elif dataset_mode == "successful_policy_vs_safe":
        if base_checkpoint is None:
            raise ValueError("base_checkpoint is required when dataset_mode='successful_policy_vs_safe'")
        info_np, legal_np, actions_np, players_np = _collect_successful_policy_vs_safe_dataset(
            config,
            games=games,
            max_steps=max_steps,
            seed=pretrain_seed,
            base_checkpoint=base_checkpoint,
            device=device,
            policy_sample=policy_sample,
        )
    else:
        raise ValueError(
            "dataset_mode must be one of 'safe_self_play', 'aggregated', "
            "or 'successful_policy_vs_safe'"
        )
    logger.info(
        "Collected safe heuristic imitation dataset: mode={} games={} states={} input_dim={} actions={}",
        dataset_mode,
        games,
        len(actions_np),
        input_dim,
        action_size,
    )

    strategy_net = StrategyNet(input_dim, action_size, config.network).to(device)
    advantage_nets = [
        AdvantageNet(input_dim, action_size, config.network).to(device),
        AdvantageNet(input_dim, action_size, config.network).to(device),
    ]
    if init_checkpoint is not None:
        init_payload = torch.load(init_checkpoint, map_location="cpu")
        if int(init_payload["input_dim"]) != input_dim or int(init_payload["action_size"]) != action_size:
            raise ValueError(
                "init checkpoint rules/action space do not match the pretrain config: "
                f"checkpoint input_dim={init_payload['input_dim']} action_size={init_payload['action_size']}, "
                f"config input_dim={input_dim} action_size={action_size}"
            )
        try:
            strategy_net.load_state_dict(init_payload["strategy_net"])
            for net, state_dict in zip(advantage_nets, init_payload["advantage_nets"], strict=True):
                net.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ValueError(
                "init checkpoint network architecture does not match the pretrain config"
            ) from exc
        logger.info("Initialized safe heuristic pretrain networks from {}", init_checkpoint)
    strategy_optimizer = torch.optim.AdamW(
        strategy_net.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    advantage_optimizers = [
        torch.optim.AdamW(
            net.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        for net in advantage_nets
    ]

    info = torch.as_tensor(info_np, dtype=torch.float32)
    legal_masks = torch.as_tensor(legal_np, dtype=torch.bool)
    target_actions = torch.as_tensor(actions_np, dtype=torch.long)
    players = torch.as_tensor(players_np, dtype=torch.long)
    advantage_targets = _advantage_targets(
        legal_masks,
        target_actions,
        action_size=action_size,
    )
    loader = DataLoader(
        TensorDataset(info, legal_masks, target_actions, advantage_targets, players),
        batch_size=batch_size,
        shuffle=True,
    )

    last_strategy_loss = 0.0
    last_strategy_accuracy = 0.0
    last_advantage_losses = [0.0, 0.0]
    for epoch in range(1, epochs + 1):
        total = 0
        correct = 0
        strategy_loss_sum = 0.0
        advantage_loss_sums = [0.0, 0.0]
        advantage_counts = [0, 0]
        strategy_net.train()
        for net in advantage_nets:
            net.train()

        for x_cpu, mask_cpu, action_cpu, advantage_cpu, player_cpu in loader:
            x = x_cpu.to(device)
            mask = mask_cpu.to(device)
            actions = action_cpu.to(device)
            advantage_target = advantage_cpu.to(device)
            player_batch = player_cpu.to(device)

            logits = strategy_net(x).masked_fill(~mask, torch.finfo(torch.float32).min)
            strategy_loss = F.cross_entropy(logits, actions)
            strategy_optimizer.zero_grad(set_to_none=True)
            strategy_loss.backward()
            if config.optimization.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(strategy_net.parameters(), config.optimization.grad_clip)
            strategy_optimizer.step()

            batch_size_actual = int(actions.shape[0])
            total += batch_size_actual
            correct += int((logits.argmax(dim=-1) == actions).sum().detach().cpu())
            strategy_loss_sum += float(strategy_loss.detach().cpu()) * batch_size_actual

            for player in (0, 1):
                player_mask = player_batch == player
                if not bool(player_mask.any()):
                    continue
                pred = advantage_nets[player](x[player_mask])
                legal = mask[player_mask]
                diff = (pred - advantage_target[player_mask]).masked_fill(~legal, 0.0)
                advantage_loss = diff.square().sum() / legal.sum().clamp_min(1)
                advantage_optimizers[player].zero_grad(set_to_none=True)
                advantage_loss.backward()
                if config.optimization.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        advantage_nets[player].parameters(),
                        config.optimization.grad_clip,
                    )
                advantage_optimizers[player].step()
                count = int(player_mask.sum().detach().cpu())
                advantage_loss_sums[player] += float(advantage_loss.detach().cpu()) * count
                advantage_counts[player] += count

        last_strategy_loss = strategy_loss_sum / max(1, total)
        last_strategy_accuracy = correct / max(1, total)
        last_advantage_losses = [
            advantage_loss_sums[player] / max(1, advantage_counts[player])
            for player in (0, 1)
        ]
        logger.info(
            "Safe heuristic pretrain epoch={}/{} strategy_loss={:.4f} strategy_accuracy={:.4f} advantage_loss=({:.4f},{:.4f})",
            epoch,
            epochs,
            last_strategy_loss,
            last_strategy_accuracy,
            last_advantage_losses[0],
            last_advantage_losses[1],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "lost_cities_config": lc_config.to_snapshot(),
            "resume_semantics": "networks_optimizers_iteration_only",
            "iteration": 0,
            "input_dim": input_dim,
            "action_size": action_size,
            "advantage_nets": [net.state_dict() for net in advantage_nets],
            "strategy_net": strategy_net.state_dict(),
            "advantage_optimizers": [optimizer.state_dict() for optimizer in advantage_optimizers],
            "strategy_optimizer": strategy_optimizer.state_dict(),
            "metrics": {
                "pretrain_dataset_mode": dataset_mode,
                "pretrain_base_checkpoint": str(base_checkpoint) if base_checkpoint is not None else None,
                "pretrain_init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
                "pretrain_policy_sample": bool(policy_sample),
                "pretrain_games": games,
                "pretrain_states": int(len(actions_np)),
                "pretrain_strategy_loss": last_strategy_loss,
                "pretrain_strategy_accuracy": last_strategy_accuracy,
                "pretrain_advantage_loss_p0": last_advantage_losses[0],
                "pretrain_advantage_loss_p1": last_advantage_losses[1],
            },
        },
        output_path,
    )
    logger.info("Saved safe heuristic pretrain checkpoint: {}", output_path)
    return HeuristicPretrainResult(
        output_path=output_path,
        dataset_mode=dataset_mode,
        games=games,
        states=int(len(actions_np)),
        strategy_loss=last_strategy_loss,
        strategy_accuracy=last_strategy_accuracy,
        advantage_loss_p0=last_advantage_losses[0],
        advantage_loss_p1=last_advantage_losses[1],
    )
