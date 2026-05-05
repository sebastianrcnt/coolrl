from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..bots.base import first_legal
from ..bots.heuristic import SafeHeuristicBot, SafeHeuristicParams
from ..bots.passive import PassiveDiscardBot
from ..bots.random import RandomBot
from ..game import GameState, LostCitiesConfig
from ..interfaces import BotInput, LostCitiesBot
from .config import config_from_dict
from .encoding import encode_information_state, legal_mask_array
from .networks import StrategyNet

SUPPORTED_OPPONENTS = (
    "random",
    "safe_heuristic",
    "safe_heuristic_loose",
    "safe_heuristic_strict",
    "noisy_safe",
    "passive_discard",
)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    token = str(device).strip().lower()
    if token == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if token == "cpu":
        return torch.device("cpu")
    if token == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is unavailable")
        return torch.device("cuda")
    raise ValueError(f"unsupported device: {device!r}")


class StrategyNetBot(LostCitiesBot):
    def __init__(
        self,
        strategy_net: StrategyNet,
        config: LostCitiesConfig,
        *,
        device: torch.device | str = "cpu",
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        self.strategy_net = strategy_net
        self.config = config
        self.device = torch.device(device)
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.last_policy_entropy: float | None = None

    def act(self, obs_or_state: BotInput) -> int:
        if not isinstance(obs_or_state, GameState):
            self.last_policy_entropy = None
            legal = np.asarray(obs_or_state["legal_mask"], dtype=bool) if isinstance(obs_or_state, dict) else np.asarray(obs_or_state.legal_mask, dtype=bool)
            return first_legal(legal)
        state = obs_or_state
        info = encode_information_state(state, state.current_player)
        legal = legal_mask_array(state)
        legal_indices = np.flatnonzero(legal)
        if len(legal_indices) == 0:
            raise RuntimeError("no legal action available")
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_net(x).squeeze(0).detach().cpu().numpy()
        masked = np.where(legal, logits, -np.inf)
        if self.sample:
            stable = masked[legal_indices] - np.max(masked[legal_indices])
            probs = np.exp(stable)
            probs = probs / probs.sum()
            unified = int(self.rng.choice(legal_indices, p=probs))
        else:
            stable = masked[legal_indices] - np.max(masked[legal_indices])
            probs = np.exp(stable)
            probs = probs / probs.sum()
            unified = int(np.argmax(masked))
        self.last_policy_entropy = float(-(probs * np.log(np.clip(probs, 1.0e-12, 1.0))).sum())
        return state.from_unified_action(unified)


class NoisySafeHeuristicBot(LostCitiesBot):
    def __init__(self, *, seed: int | None = None, noise: float = 0.15) -> None:
        if not 0.0 <= noise <= 1.0:
            raise ValueError(f"noise must be between 0 and 1, got {noise}")
        self.safe_bot = SafeHeuristicBot()
        self.random_bot = RandomBot(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def act(self, obs_or_state: BotInput) -> int:
        if self.rng.random() < self.noise:
            return self.random_bot.act(obs_or_state)
        return self.safe_bot.act(obs_or_state)


def load_strategy_bot_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    sample: bool = False,
    seed: int | None = None,
) -> tuple[StrategyNetBot, LostCitiesConfig]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = config_from_dict(payload["config"])
    resolved_device = _resolve_device(device)
    strategy_net = StrategyNet(
        int(payload["input_dim"]),
        int(payload["action_size"]),
        config.network,
    ).to(resolved_device)
    strategy_net.load_state_dict(payload["strategy_net"])
    strategy_net.eval()
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    return (
        StrategyNetBot(
            strategy_net,
            lc_config,
            device=resolved_device,
            sample=sample,
            seed=seed,
        ),
        lc_config,
    )


def make_opponent(name: str, seed: int | None = None) -> LostCitiesBot:
    token = name.lower()
    if token == "random":
        return RandomBot(seed=seed)
    if token == "safe_heuristic":
        return SafeHeuristicBot()
    if token == "safe_heuristic_loose":
        return SafeHeuristicBot(
            SafeHeuristicParams(
                open_target_ratio=0.42,
                open_min_card_ratio=0.34,
                handshake_target_multiplier=1.05,
                late_deck_ratio=0.15,
                gift_penalty_weight=0.85,
            )
        )
    if token == "safe_heuristic_strict":
        return SafeHeuristicBot(
            SafeHeuristicParams(
                open_target_ratio=0.58,
                open_min_card_ratio=0.50,
                handshake_target_multiplier=1.30,
                late_deck_ratio=0.25,
                gift_penalty_weight=1.25,
            )
        )
    if token == "noisy_safe":
        return NoisySafeHeuristicBot(seed=seed, noise=0.15)
    if token == "passive_discard":
        return PassiveDiscardBot()
    raise ValueError(f"unsupported opponent: {name!r}")


def _opened_color_count(state: GameState, player: int) -> int:
    return sum(1 for expedition in state.expeditions[player] if len(expedition) > 0)


def _expedition_card_count(state: GameState, player: int) -> int:
    return sum(len(expedition) for expedition in state.expeditions[player])


def is_card_play_action(action_id: int) -> bool:
    """Card-phase phase-local actions use even ids for play and odd ids for discard."""
    return action_id % 2 == 0


def is_card_discard_action(action_id: int) -> bool:
    return not is_card_play_action(action_id)


def is_draw_deck_action(action_id: int) -> bool:
    return action_id == 0


def _play_game_for_evaluation(
    bot0: LostCitiesBot,
    bot1: LostCitiesBot,
    config: LostCitiesConfig,
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
    tracked_player: int | None = None,
) -> tuple[GameState, bool, dict[str, float | int], int]:
    game_config = replace(config, seed=seed) if seed is not None else config
    state = GameState.new_game(game_config)
    bots = [bot0, bot1]
    action_counts: dict[str, float | int] = {
        "play_actions": 0,
        "discard_actions": 0,
        "draw_deck_actions": 0,
        "draw_pile_actions": 0,
        "policy_entropy_sum": 0.0,
        "policy_entropy_actions": 0,
    }
    for step in range(max_steps):
        if state.terminal:
            return state, False, action_counts, step
        acting_player = state.current_player
        phase = state.phase
        bot = bots[state.current_player]
        action = bot.act(state)
        if tracked_player is not None and acting_player == tracked_player and phase == "card":
            if is_card_play_action(action):
                action_counts["play_actions"] = int(action_counts["play_actions"]) + 1
            elif is_card_discard_action(action):
                action_counts["discard_actions"] = int(action_counts["discard_actions"]) + 1
        if tracked_player is not None and acting_player == tracked_player and phase == "draw":
            if is_draw_deck_action(action):
                action_counts["draw_deck_actions"] = int(action_counts["draw_deck_actions"]) + 1
            else:
                action_counts["draw_pile_actions"] = int(action_counts["draw_pile_actions"]) + 1
        if tracked_player is not None and acting_player == tracked_player and isinstance(bot, StrategyNetBot):
            if bot.last_policy_entropy is not None:
                action_counts["policy_entropy_sum"] = float(action_counts["policy_entropy_sum"]) + bot.last_policy_entropy
                action_counts["policy_entropy_actions"] = int(action_counts["policy_entropy_actions"]) + 1
        state.apply_action(action)
    return state, not state.terminal, action_counts, max_steps


def evaluate_against_bot(
    strategy_net: StrategyNet,
    opponent_bot: LostCitiesBot,
    config: LostCitiesConfig,
    games: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
    max_steps: int = 10_000,
    on_max_steps: str = "score_diff",
    sample: bool = False,
) -> dict[str, float | int]:
    strategy_net.eval()
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    timeout_mode = str(on_max_steps).strip().lower()
    if timeout_mode not in {"score_diff", "loss", "draw"}:
        raise ValueError(
            "on_max_steps must be one of 'score_diff', 'loss', or 'draw'"
        )
    diffs: list[int] = []
    final_scores: list[int] = []
    opponent_scores: list[int] = []
    opened_colors: list[int] = []
    opponent_opened_colors: list[int] = []
    expedition_cards: list[int] = []
    play_actions = 0
    discard_actions = 0
    draw_deck_actions = 0
    draw_pile_actions = 0
    game_lengths: list[int] = []
    policy_entropy_sum = 0.0
    policy_entropy_actions = 0
    wins = losses = draws = 0
    max_step_timeouts = 0
    for index in range(games):
        net_bot = StrategyNetBot(strategy_net, config, device=device, sample=sample, seed=seed + index)
        game_seed = seed + index
        if index % 2 == 0:
            final_state, timed_out, action_counts, game_length = _play_game_for_evaluation(
                net_bot,
                opponent_bot,
                config,
                seed=game_seed,
                max_steps=max_steps,
                tracked_player=0,
            )
            deep_cfr_player = 0
        else:
            final_state, timed_out, action_counts, game_length = _play_game_for_evaluation(
                opponent_bot,
                net_bot,
                config,
                seed=game_seed,
                max_steps=max_steps,
                tracked_player=1,
            )
            deep_cfr_player = 1
        opponent_player = 1 - deep_cfr_player
        final_scores.append(final_state.total_score(deep_cfr_player))
        opponent_scores.append(final_state.total_score(opponent_player))
        opened_colors.append(_opened_color_count(final_state, deep_cfr_player))
        opponent_opened_colors.append(_opened_color_count(final_state, opponent_player))
        expedition_cards.append(_expedition_card_count(final_state, deep_cfr_player))
        game_lengths.append(game_length)
        play_actions += int(action_counts["play_actions"])
        discard_actions += int(action_counts["discard_actions"])
        draw_deck_actions += int(action_counts["draw_deck_actions"])
        draw_pile_actions += int(action_counts["draw_pile_actions"])
        policy_entropy_sum += float(action_counts["policy_entropy_sum"])
        policy_entropy_actions += int(action_counts["policy_entropy_actions"])
        if timed_out:
            max_step_timeouts += 1
            if timeout_mode == "score_diff":
                diff = final_state.score_diff(deep_cfr_player)
            elif timeout_mode == "loss":
                diff = -1
            else:
                diff = 0
        else:
            diff = final_state.score_diff(deep_cfr_player)
        diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1
    card_actions = play_actions + discard_actions
    draw_actions = draw_deck_actions + draw_pile_actions
    return {
        "games": int(games),
        "win_rate": float(wins / max(1, games)),
        "avg_diff": float(np.mean(diffs)) if diffs else 0.0,
        "avg_final_score": float(np.mean(final_scores)) if final_scores else 0.0,
        "avg_opponent_score": float(np.mean(opponent_scores)) if opponent_scores else 0.0,
        "avg_opened_colors": float(np.mean(opened_colors)) if opened_colors else 0.0,
        "avg_opponent_opened_colors": float(np.mean(opponent_opened_colors)) if opponent_opened_colors else 0.0,
        "avg_expedition_cards": float(np.mean(expedition_cards)) if expedition_cards else 0.0,
        "avg_discard_actions": float(discard_actions / max(1, games)),
        "avg_play_actions": float(play_actions / max(1, games)),
        "avg_draw_deck_actions": float(draw_deck_actions / max(1, games)),
        "avg_draw_pile_actions": float(draw_pile_actions / max(1, games)),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "policy_entropy": float(policy_entropy_sum / max(1, policy_entropy_actions)),
        "play_action_rate": float(play_actions / max(1, card_actions)),
        "discard_action_rate": float(discard_actions / max(1, card_actions)),
        "draw_deck_rate": float(draw_deck_actions / max(1, draw_actions)),
        "draw_pile_rate": float(draw_pile_actions / max(1, draw_actions)),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "max_step_timeouts": max_step_timeouts,
    }
