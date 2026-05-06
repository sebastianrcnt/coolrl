from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from .bots.heuristic import SafeHeuristicBot, SafeHeuristicParams
from .bots.passive import PassiveDiscardBot
from .bots.random import RandomBot
from .deep_cfr.encoding import color_playability_summary
from .game import GameState, LostCitiesConfig
from .interfaces import BotInput, LostCitiesBot

SUPPORTED_OPPONENTS = (
    "random",
    "safe_heuristic",
    "safe_heuristic_loose",
    "safe_heuristic_strict",
    "noisy_safe",
    "passive_discard",
)


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


def is_card_play_action(action_id: int) -> bool:
    """Card-phase phase-local actions use even ids for play and odd ids for discard."""
    return action_id % 2 == 0


def is_card_discard_action(action_id: int) -> bool:
    return not is_card_play_action(action_id)


def is_draw_deck_action(action_id: int) -> bool:
    return action_id == 0


def _opened_color_count(state: GameState, player: int) -> int:
    return sum(1 for expedition in state.expeditions[player] if len(expedition) > 0)


def _expedition_card_count(state: GameState, player: int) -> int:
    return sum(len(expedition) for expedition in state.expeditions[player])


def _opened_expedition_scores(state: GameState, player: int) -> list[float]:
    return [
        float(state.expedition_score(player, color))
        for color, expedition in enumerate(state.expeditions[player])
        if expedition
    ]


def _opening_play_summary(
    state: GameState,
    player: int,
    action: int,
) -> dict[str, float | bool] | None:
    if state.phase != "card" or not is_card_play_action(action):
        return None
    slot = action // 2
    hand = state.hands[player]
    if slot < 0 or slot >= len(hand):
        return None
    card = hand[slot]
    if len(state.expeditions[player][card.color]) > 0:
        return None
    summary = color_playability_summary(state, player, card.color)
    recoverable_score = float(summary["recoverable_score_no_bonus"])
    margin = float(summary["recoverable_margin_no_bonus"])
    has_bonus_path = bool(summary["has_bonus_path"])
    return {
        "color": int(card.color),
        "recoverable_score_no_bonus": recoverable_score,
        "recoverable_margin_no_bonus": margin,
        "has_bonus_path": has_bonus_path,
        "is_bad_open": recoverable_score < 0.0 and not has_bonus_path,
        "is_weak_open": recoverable_score < 0.0 and has_bonus_path,
        "is_good_open": recoverable_score >= 0.0,
    }


def play_game_for_evaluation(
    bot0: LostCitiesBot,
    bot1: LostCitiesBot,
    config: LostCitiesConfig,
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
    tracked_player: int | None = None,
) -> tuple[GameState, bool, dict[str, Any], int]:
    game_config = replace(config, seed=seed) if seed is not None else config
    state = GameState.new_game(game_config)
    bots = [bot0, bot1]
    action_counts: dict[str, Any] = {
        "play_actions": 0,
        "discard_actions": 0,
        "draw_deck_actions": 0,
        "draw_pile_actions": 0,
        "policy_entropy_sum": 0.0,
        "policy_entropy_actions": 0,
        "opening_play_actions": 0,
        "bad_open_actions": 0,
        "weak_open_actions": 0,
        "good_open_actions": 0,
        "opening_recoverable_score_sum": 0.0,
        "opening_margin_sum": 0.0,
        "opening_recoverable_scores": [],
        "bad_open_colors": set(),
        "my_discarded_cards": 0,
        "opponent_discarded_cards": 0,
        "my_took_opponent_discards": 0,
        "opponent_took_my_discards": 0,
    }
    discard_owner_stacks: list[list[int]] = [[] for _ in range(config.n_colors)]
    for step in range(max_steps):
        if state.terminal:
            return state, False, action_counts, step
        acting_player = state.current_player
        phase = state.phase
        bot = bots[state.current_player]
        action = bot.act(state)
        if tracked_player is not None and phase == "card" and is_card_discard_action(action):
            slot = action // 2
            hand = state.hands[acting_player]
            if 0 <= slot < len(hand):
                card = hand[slot]
                discard_owner_stacks[card.color].append(acting_player)
                if acting_player == tracked_player:
                    action_counts["my_discarded_cards"] = int(action_counts["my_discarded_cards"]) + 1
                else:
                    action_counts["opponent_discarded_cards"] = (
                        int(action_counts["opponent_discarded_cards"]) + 1
                    )
        if tracked_player is not None and phase == "draw" and not is_draw_deck_action(action):
            color = action - 1
            if 0 <= color < len(discard_owner_stacks):
                top_owner = discard_owner_stacks[color][-1] if discard_owner_stacks[color] else None
                if top_owner is not None and top_owner != acting_player:
                    if acting_player == tracked_player:
                        action_counts["my_took_opponent_discards"] = (
                            int(action_counts["my_took_opponent_discards"]) + 1
                        )
                    elif top_owner == tracked_player:
                        action_counts["opponent_took_my_discards"] = (
                            int(action_counts["opponent_took_my_discards"]) + 1
                        )
                if discard_owner_stacks[color]:
                    discard_owner_stacks[color].pop()
        if tracked_player is not None and acting_player == tracked_player and phase == "card":
            if is_card_play_action(action):
                action_counts["play_actions"] = int(action_counts["play_actions"]) + 1
                opening = _opening_play_summary(state, acting_player, action)
                if opening is not None:
                    action_counts["opening_play_actions"] = int(action_counts["opening_play_actions"]) + 1
                    if opening["is_bad_open"]:
                        action_counts["bad_open_actions"] = int(action_counts["bad_open_actions"]) + 1
                        colors = action_counts["bad_open_colors"]
                        if isinstance(colors, set):
                            colors.add(int(opening["color"]))
                    if opening["is_weak_open"]:
                        action_counts["weak_open_actions"] = int(action_counts["weak_open_actions"]) + 1
                    if opening["is_good_open"]:
                        action_counts["good_open_actions"] = int(action_counts["good_open_actions"]) + 1
                    action_counts["opening_recoverable_score_sum"] = float(
                        action_counts["opening_recoverable_score_sum"]
                    ) + float(opening["recoverable_score_no_bonus"])
                    action_counts["opening_margin_sum"] = float(action_counts["opening_margin_sum"]) + float(
                        opening["recoverable_margin_no_bonus"]
                    )
                    scores = action_counts["opening_recoverable_scores"]
                    if isinstance(scores, list):
                        scores.append(float(opening["recoverable_score_no_bonus"]))
            elif is_card_discard_action(action):
                action_counts["discard_actions"] = int(action_counts["discard_actions"]) + 1
        if tracked_player is not None and acting_player == tracked_player and phase == "draw":
            if is_draw_deck_action(action):
                action_counts["draw_deck_actions"] = int(action_counts["draw_deck_actions"]) + 1
            else:
                action_counts["draw_pile_actions"] = int(action_counts["draw_pile_actions"]) + 1
        if tracked_player is not None and acting_player == tracked_player:
            last_policy_entropy = getattr(bot, "last_policy_entropy", None)
            if last_policy_entropy is not None:
                action_counts["policy_entropy_sum"] = (
                    float(action_counts["policy_entropy_sum"]) + float(last_policy_entropy)
                )
                action_counts["policy_entropy_actions"] = int(action_counts["policy_entropy_actions"]) + 1
        state.apply_action(action)
    return state, not state.terminal, action_counts, max_steps


def evaluate_agent_against_bot(
    agent_factory: Callable[[int], LostCitiesBot],
    opponent_bot: LostCitiesBot,
    config: LostCitiesConfig,
    games: int,
    seed: int,
    *,
    max_steps: int = 10_000,
    on_max_steps: str = "score_diff",
) -> dict[str, float | int]:
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    timeout_mode = str(on_max_steps).strip().lower()
    if timeout_mode not in {"score_diff", "loss", "draw"}:
        raise ValueError("on_max_steps must be one of 'score_diff', 'loss', or 'draw'")
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
    opening_play_actions = 0
    bad_open_actions = 0
    weak_open_actions = 0
    good_open_actions = 0
    opening_recoverable_score_sum = 0.0
    opening_margin_sum = 0.0
    opening_recoverable_scores: list[float] = []
    expedition_final_scores: list[float] = []
    bad_open_final_scores: list[float] = []
    positive_expeditions_per_game: list[int] = []
    negative_expeditions_per_game: list[int] = []
    five_color_diffs: list[int] = []
    five_color_expedition_scores: list[float] = []
    my_discarded_cards = 0
    opponent_discarded_cards = 0
    my_took_opponent_discards = 0
    opponent_took_my_discards = 0
    wins = losses = draws = 0
    max_step_timeouts = 0
    for index in range(games):
        agent_bot = agent_factory(index)
        game_seed = seed + index
        if index % 2 == 0:
            final_state, timed_out, action_counts, game_length = play_game_for_evaluation(
                agent_bot,
                opponent_bot,
                config,
                seed=game_seed,
                max_steps=max_steps,
                tracked_player=0,
            )
            agent_player = 0
        else:
            final_state, timed_out, action_counts, game_length = play_game_for_evaluation(
                opponent_bot,
                agent_bot,
                config,
                seed=game_seed,
                max_steps=max_steps,
                tracked_player=1,
            )
            agent_player = 1
        opponent_player = 1 - agent_player
        final_scores.append(final_state.total_score(agent_player))
        opponent_scores.append(final_state.total_score(opponent_player))
        opened_color_count = _opened_color_count(final_state, agent_player)
        opened_colors.append(opened_color_count)
        opponent_opened_colors.append(_opened_color_count(final_state, opponent_player))
        expedition_cards.append(_expedition_card_count(final_state, agent_player))
        game_expedition_scores = _opened_expedition_scores(final_state, agent_player)
        expedition_final_scores.extend(game_expedition_scores)
        positive_expeditions_per_game.append(sum(1 for score in game_expedition_scores if score >= 0.0))
        negative_expeditions_per_game.append(sum(1 for score in game_expedition_scores if score < 0.0))
        bad_open_colors = action_counts.get("bad_open_colors")
        if isinstance(bad_open_colors, set):
            for color in bad_open_colors:
                if 0 <= color < config.n_colors and final_state.expeditions[agent_player][color]:
                    bad_open_final_scores.append(float(final_state.expedition_score(agent_player, color)))
        game_lengths.append(game_length)
        play_actions += int(action_counts["play_actions"])
        discard_actions += int(action_counts["discard_actions"])
        draw_deck_actions += int(action_counts["draw_deck_actions"])
        draw_pile_actions += int(action_counts["draw_pile_actions"])
        policy_entropy_sum += float(action_counts["policy_entropy_sum"])
        policy_entropy_actions += int(action_counts["policy_entropy_actions"])
        opening_play_actions += int(action_counts["opening_play_actions"])
        bad_open_actions += int(action_counts["bad_open_actions"])
        weak_open_actions += int(action_counts["weak_open_actions"])
        good_open_actions += int(action_counts["good_open_actions"])
        opening_recoverable_score_sum += float(action_counts["opening_recoverable_score_sum"])
        opening_margin_sum += float(action_counts["opening_margin_sum"])
        scores = action_counts["opening_recoverable_scores"]
        if isinstance(scores, list):
            opening_recoverable_scores.extend(float(score) for score in scores)
        my_discarded_cards += int(action_counts["my_discarded_cards"])
        opponent_discarded_cards += int(action_counts["opponent_discarded_cards"])
        my_took_opponent_discards += int(action_counts["my_took_opponent_discards"])
        opponent_took_my_discards += int(action_counts["opponent_took_my_discards"])
        if timed_out:
            max_step_timeouts += 1
            if timeout_mode == "score_diff":
                diff = final_state.score_diff(agent_player)
            elif timeout_mode == "loss":
                diff = -1
            else:
                diff = 0
        else:
            diff = final_state.score_diff(agent_player)
        diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1
        if opened_color_count == config.n_colors:
            five_color_diffs.append(diff)
            five_color_expedition_scores.extend(game_expedition_scores)
    card_actions = play_actions + discard_actions
    draw_actions = draw_deck_actions + draw_pile_actions
    bad_or_weak_open_actions = bad_open_actions + weak_open_actions
    positive_expedition_scores = [score for score in expedition_final_scores if score >= 0.0]
    negative_expedition_scores = [score for score in expedition_final_scores if score < 0.0]
    bad_open_final_positive_scores = [score for score in bad_open_final_scores if score >= 0.0]
    five_color_positive_scores = [score for score in five_color_expedition_scores if score >= 0.0]
    return {
        "games": int(games),
        "win_rate": float(wins / max(1, games)),
        "avg_diff": float(np.mean(diffs)) if diffs else 0.0,
        "avg_final_score": float(np.mean(final_scores)) if final_scores else 0.0,
        "avg_opponent_score": float(np.mean(opponent_scores)) if opponent_scores else 0.0,
        "avg_opened_colors": float(np.mean(opened_colors)) if opened_colors else 0.0,
        "opened_colors_std": float(np.std(opened_colors)) if opened_colors else 0.0,
        "opened_colors_min": int(min(opened_colors)) if opened_colors else 0,
        "opened_colors_max": int(max(opened_colors)) if opened_colors else 0,
        **{f"opened_colors_count_{count}": int(opened_colors.count(count)) for count in range(config.n_colors + 1)},
        "avg_opponent_opened_colors": float(np.mean(opponent_opened_colors)) if opponent_opened_colors else 0.0,
        "opponent_opened_colors_std": float(np.std(opponent_opened_colors)) if opponent_opened_colors else 0.0,
        "opponent_opened_colors_min": int(min(opponent_opened_colors)) if opponent_opened_colors else 0,
        "opponent_opened_colors_max": int(max(opponent_opened_colors)) if opponent_opened_colors else 0,
        **{
            f"opponent_opened_colors_count_{count}": int(opponent_opened_colors.count(count))
            for count in range(config.n_colors + 1)
        },
        "avg_expedition_cards": float(np.mean(expedition_cards)) if expedition_cards else 0.0,
        "positive_expedition_rate": float(
            len(positive_expedition_scores) / max(1, len(expedition_final_scores))
        ),
        "negative_expedition_rate": float(
            len(negative_expedition_scores) / max(1, len(expedition_final_scores))
        ),
        "avg_positive_expeditions": float(np.mean(positive_expeditions_per_game))
        if positive_expeditions_per_game
        else 0.0,
        "avg_negative_expeditions": float(np.mean(negative_expeditions_per_game))
        if negative_expeditions_per_game
        else 0.0,
        "final_expedition_score_mean": float(np.mean(expedition_final_scores))
        if expedition_final_scores
        else 0.0,
        "final_expedition_score_p25": float(np.percentile(expedition_final_scores, 25))
        if expedition_final_scores
        else 0.0,
        "final_expedition_score_median": float(np.percentile(expedition_final_scores, 50))
        if expedition_final_scores
        else 0.0,
        "avg_discard_actions": float(discard_actions / max(1, games)),
        "avg_play_actions": float(play_actions / max(1, games)),
        "avg_draw_deck_actions": float(draw_deck_actions / max(1, games)),
        "avg_draw_pile_actions": float(draw_pile_actions / max(1, games)),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "policy_entropy": float(policy_entropy_sum / max(1, policy_entropy_actions)),
        "opening_play_actions": opening_play_actions,
        "bad_open_actions": bad_open_actions,
        "weak_open_actions": weak_open_actions,
        "good_open_actions": good_open_actions,
        "bad_open_rate": float(bad_open_actions / max(1, opening_play_actions)),
        "weak_open_rate": float(weak_open_actions / max(1, opening_play_actions)),
        "bad_or_weak_open_rate": float(bad_or_weak_open_actions / max(1, opening_play_actions)),
        "bad_open_per_game": float(bad_open_actions / max(1, games)),
        "bad_or_weak_open_per_game": float(bad_or_weak_open_actions / max(1, games)),
        "good_open_rate": float(good_open_actions / max(1, opening_play_actions)),
        "bad_open_final_positive_rate": float(
            len(bad_open_final_positive_scores) / max(1, len(bad_open_final_scores))
        ),
        "bad_open_final_score_mean": float(np.mean(bad_open_final_scores))
        if bad_open_final_scores
        else 0.0,
        "opening_recoverable_score_mean": float(
            opening_recoverable_score_sum / max(1, opening_play_actions)
        ),
        "opening_recoverable_score_p25": float(np.percentile(opening_recoverable_scores, 25))
        if opening_recoverable_scores
        else 0.0,
        "opening_margin_mean": float(opening_margin_sum / max(1, opening_play_actions)),
        "avg_score_per_opened_color": float(
            np.mean(
                [
                    score / max(1, opened)
                    for score, opened in zip(final_scores, opened_colors, strict=True)
                ]
            )
        )
        if final_scores
        else 0.0,
        "play_action_rate": float(play_actions / max(1, card_actions)),
        "discard_action_rate": float(discard_actions / max(1, card_actions)),
        "draw_deck_rate": float(draw_deck_actions / max(1, draw_actions)),
        "draw_pile_rate": float(draw_pile_actions / max(1, draw_actions)),
        "my_took_opponent_discard_rate": float(
            my_took_opponent_discards / max(1, opponent_discarded_cards)
        ),
        "opponent_took_my_discard_rate": float(
            opponent_took_my_discards / max(1, my_discarded_cards)
        ),
        "net_discard_take_rate": float(
            (my_took_opponent_discards / max(1, opponent_discarded_cards))
            - (opponent_took_my_discards / max(1, my_discarded_cards))
        ),
        "five_color_positive_expedition_rate": float(
            len(five_color_positive_scores) / max(1, len(five_color_expedition_scores))
        ),
        "five_color_avg_diff": float(np.mean(five_color_diffs)) if five_color_diffs else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "max_step_timeouts": max_step_timeouts,
    }
