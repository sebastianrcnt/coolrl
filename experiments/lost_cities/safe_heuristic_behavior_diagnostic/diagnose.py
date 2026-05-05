#!/usr/bin/env python3
"""Safe heuristic 계열 bot 행동 분포 진단."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from coolrl.lost_cities.evaluation import (
    is_card_discard_action,
    is_card_play_action,
    is_draw_deck_action,
    make_opponent,
)
from coolrl.lost_cities.game import GameState, tier_config

DEFAULT_OPPONENTS = (
    "safe_heuristic",
    "safe_heuristic_loose",
    "safe_heuristic_strict",
    "random",
    "passive_discard",
    "noisy_safe",
)


@dataclass
class PlayerGameStats:
    scores: list[int] = field(default_factory=list)
    score_diffs: list[int] = field(default_factory=list)
    opened_colors: list[int] = field(default_factory=list)
    play_actions: int = 0
    discard_actions: int = 0
    draw_deck_actions: int = 0
    draw_pile_actions: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="safe heuristic 계열 bot 행동 분포를 진단합니다.")
    parser.add_argument("--games", type=int, default=100, help="matchup당 게임 수")
    parser.add_argument("--seed", type=int, default=61, help="기본 seed")
    parser.add_argument("--tier", default="tier3", help="Lost Cities rules tier")
    parser.add_argument("--max-steps", type=int, default=1000, help="게임당 최대 step")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=list(DEFAULT_OPPONENTS),
        help="anchor safe_heuristic와 대결할 opponent 목록",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("/tmp/safe_heuristic_behavior_diagnostic.json"),
        help="JSON 결과 경로",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("/tmp/safe_heuristic_behavior_diagnostic.md"),
        help="Markdown 요약 경로",
    )
    return parser.parse_args()


def opened_color_count(state: GameState, player: int) -> int:
    return sum(1 for expedition in state.expeditions[player] if len(expedition) > 0)


def new_bot(name: str, seed: int):
    return make_opponent(name, seed=seed)


def record_action(stats: PlayerGameStats, phase: str, action: int) -> None:
    if phase == "card":
        if is_card_play_action(action):
            stats.play_actions += 1
        elif is_card_discard_action(action):
            stats.discard_actions += 1
    elif phase == "draw":
        if is_draw_deck_action(action):
            stats.draw_deck_actions += 1
        else:
            stats.draw_pile_actions += 1


def play_matchup_game(
    safe_name: str,
    opponent_name: str,
    *,
    safe_player: int,
    config,
    seed: int,
    max_steps: int,
) -> tuple[GameState, bool, int, dict[str, PlayerGameStats]]:
    state = GameState.new_game(config, seed=seed)
    safe_bot = new_bot(safe_name, seed + 10_000)
    opponent_bot = new_bot(opponent_name, seed + 20_000)
    bots = [None, None]
    bots[safe_player] = safe_bot
    bots[1 - safe_player] = opponent_bot
    stats_by_role = {
        "anchor_safe": PlayerGameStats(),
        "opponent": PlayerGameStats(),
    }
    role_by_player = {
        safe_player: "anchor_safe",
        1 - safe_player: "opponent",
    }

    for step in range(max_steps):
        if state.terminal:
            return state, False, step, stats_by_role
        acting_player = state.current_player
        phase = state.phase
        action = bots[acting_player].act(state)
        record_action(stats_by_role[role_by_player[acting_player]], phase, action)
        state.apply_action(action)
    return state, not state.terminal, max_steps, stats_by_role


def summarize_player_stats(stats: PlayerGameStats, games: int, game_lengths: list[int]) -> dict[str, Any]:
    opened = np.asarray(stats.opened_colors, dtype=float)
    scores = np.asarray(stats.scores, dtype=float)
    diffs = np.asarray(stats.score_diffs, dtype=float)
    opened_counter = Counter(stats.opened_colors)
    card_actions = stats.play_actions + stats.discard_actions
    draw_actions = stats.draw_deck_actions + stats.draw_pile_actions
    return {
        "games": games,
        "avg_score": float(scores.mean()) if len(scores) else 0.0,
        "avg_diff": float(diffs.mean()) if len(diffs) else 0.0,
        "win_rate": stats.wins / max(1, games),
        "loss_rate": stats.losses / max(1, games),
        "draw_rate": stats.draws / max(1, games),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "timeouts": stats.timeouts,
        "timeout_rate": stats.timeouts / max(1, games),
        "terminal_rate": (games - stats.timeouts) / max(1, games),
        "opened_colors_mean": float(opened.mean()) if len(opened) else 0.0,
        "opened_colors_std": float(opened.std(ddof=0)) if len(opened) else 0.0,
        "opened_colors_min": int(opened.min()) if len(opened) else 0,
        "opened_colors_max": int(opened.max()) if len(opened) else 0,
        "opened_colors_histogram": {str(key): opened_counter.get(key, 0) for key in range(6)},
        "play_action_rate": stats.play_actions / max(1, card_actions),
        "discard_action_rate": stats.discard_actions / max(1, card_actions),
        "draw_deck_rate": stats.draw_deck_actions / max(1, draw_actions),
        "draw_pile_rate": stats.draw_pile_actions / max(1, draw_actions),
        "play_actions": stats.play_actions,
        "discard_actions": stats.discard_actions,
        "draw_deck_actions": stats.draw_deck_actions,
        "draw_pile_actions": stats.draw_pile_actions,
    }


def run_matchup(opponent_name: str, *, games: int, seed: int, config, max_steps: int) -> dict[str, Any]:
    role_stats = {
        "anchor_safe": PlayerGameStats(),
        "opponent": PlayerGameStats(),
    }
    game_lengths: list[int] = []
    for index in range(games):
        safe_player = index % 2
        game_seed = seed + index
        final_state, timed_out, game_length, game_role_stats = play_matchup_game(
            "safe_heuristic",
            opponent_name,
            safe_player=safe_player,
            config=config,
            seed=game_seed,
            max_steps=max_steps,
        )
        game_lengths.append(game_length)
        player_by_role = {
            "anchor_safe": safe_player,
            "opponent": 1 - safe_player,
        }
        for role, player in player_by_role.items():
            opponent_player = 1 - player
            stats = role_stats[role]
            stats.play_actions += game_role_stats[role].play_actions
            stats.discard_actions += game_role_stats[role].discard_actions
            stats.draw_deck_actions += game_role_stats[role].draw_deck_actions
            stats.draw_pile_actions += game_role_stats[role].draw_pile_actions
            score = final_state.total_score(player)
            diff = final_state.score_diff(player)
            stats.scores.append(score)
            stats.score_diffs.append(diff)
            stats.opened_colors.append(opened_color_count(final_state, player))
            if timed_out:
                stats.timeouts += 1
            if diff > 0:
                stats.wins += 1
            elif diff < 0:
                stats.losses += 1
            else:
                stats.draws += 1
            assert final_state.score_diff(opponent_player) == -diff

    return {
        "opponent": opponent_name,
        "games": games,
        "anchor_safe": summarize_player_stats(role_stats["anchor_safe"], games, game_lengths),
        "opponent_policy": summarize_player_stats(role_stats["opponent"], games, game_lengths),
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Safe Heuristic Behavior Diagnostic",
        "",
        f"- games per matchup: `{payload['games']}`",
        f"- seed: `{payload['seed']}`",
        f"- max steps: `{payload['max_steps']}`",
        "",
        "| opponent | role | avg score | avg diff | win rate | opened mean | opened std | opened hist | play rate | discard rate | timeout rate | avg length |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for matchup in payload["matchups"]:
        for role in ("anchor_safe", "opponent_policy"):
            stats = matchup[role]
            hist = ", ".join(
                f"{key}:{value}" for key, value in stats["opened_colors_histogram"].items() if value
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        matchup["opponent"],
                        role,
                        format_value(stats["avg_score"]),
                        format_value(stats["avg_diff"]),
                        format_value(stats["win_rate"]),
                        format_value(stats["opened_colors_mean"]),
                        format_value(stats["opened_colors_std"]),
                        hist,
                        format_value(stats["play_action_rate"]),
                        format_value(stats["discard_action_rate"]),
                        format_value(stats["timeout_rate"]),
                        format_value(stats["avg_game_length"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    config = tier_config(args.tier)
    matchups = [
        run_matchup(
            opponent,
            games=args.games,
            seed=args.seed + opponent_index * 100_000,
            config=config,
            max_steps=args.max_steps,
        )
        for opponent_index, opponent in enumerate(args.opponents)
    ]
    payload = {
        "games": args.games,
        "seed": args.seed,
        "tier": args.tier,
        "max_steps": args.max_steps,
        "matchups": matchups,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"- json: {args.json_output}")
    print(f"- markdown: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
