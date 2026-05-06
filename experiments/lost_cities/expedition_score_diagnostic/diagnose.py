#!/usr/bin/env python3
"""Opened expedition final-score diagnostic for Lost Cities Deep CFR checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coolrl.lost_cities.deep_cfr.config import config_from_dict
from coolrl.lost_cities.deep_cfr.encoding import color_playability_summary
from coolrl.lost_cities.deep_cfr.evaluate import StrategyNetBot
from coolrl.lost_cities.deep_cfr.networks import StrategyNet
from coolrl.lost_cities.evaluation import make_opponent
from coolrl.lost_cities.game import GameState, LostCitiesConfig
from coolrl.lost_cities.interfaces import LostCitiesBot

DEFAULT_OPPONENTS = (
    "noisy_safe",
    "safe_heuristic",
    "safe_heuristic_loose",
    "safe_heuristic_strict",
    "random",
    "passive_discard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StrategyNet checkpoint의 opened expedition final score 분포를 진단합니다."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="진단할 checkpoint path. 여러 번 지정 가능.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON 출력 경로")
    parser.add_argument("--jsonl-output", type=Path, default=None, help="선택적 JSONL 출력 경로")
    parser.add_argument("--games", type=int, default=100, help="opponent별 평가 게임 수")
    parser.add_argument("--seed", type=int, default=123_000, help="offline eval base seed")
    parser.add_argument(
        "--opponent",
        action="append",
        choices=DEFAULT_OPPONENTS,
        default=None,
        help="평가할 opponent. 생략하면 기본 opponent 전체를 사용.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--sample", action="store_true", help="StrategyNet policy를 sample로 실행")
    parser.add_argument("--max-steps", type=int, default=None, help="평가 max steps override")
    parser.add_argument(
        "--on-max-steps",
        choices=("score_diff", "loss", "draw"),
        default=None,
        help="checkpoint config의 evaluation.on_max_steps override",
    )
    return parser.parse_args()


def resolve_device(token: str) -> torch.device:
    if token == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if token == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda is unavailable")
    return torch.device(token)


def load_checkpoint_bot(
    checkpoint_path: Path,
    *,
    device: torch.device,
    sample: bool,
    seed: int,
) -> tuple[dict[str, Any], StrategyNetBot, LostCitiesConfig, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    run_config = config_from_dict(payload["config"])
    strategy_net = StrategyNet(
        int(payload["input_dim"]),
        int(payload["action_size"]),
        run_config.network,
    ).to(device)
    strategy_net.load_state_dict(payload["strategy_net"])
    strategy_net.eval()
    lc_config = run_config.rules.to_lost_cities_config(seed=run_config.seed)
    bot = StrategyNetBot(
        strategy_net,
        lc_config,
        device=device,
        encoding=run_config.encoding,
        sample=sample,
        seed=seed,
    )
    return payload, bot, lc_config, run_config


def is_card_play_action(action_id: int) -> bool:
    return action_id % 2 == 0


def play_diagnostic_game(
    agent_bot: LostCitiesBot,
    opponent_bot: LostCitiesBot,
    config: LostCitiesConfig,
    *,
    seed: int,
    agent_player: int,
    max_steps: int,
) -> tuple[GameState, bool, dict[int, float], int]:
    state = GameState.new_game(config, seed=seed)
    bots = [agent_bot, opponent_bot] if agent_player == 0 else [opponent_bot, agent_bot]
    first_open_recoverable_by_color: dict[int, float] = {}

    for step in range(max_steps):
        if state.terminal:
            return state, False, first_open_recoverable_by_color, step
        acting_player = state.current_player
        phase = state.phase
        action = bots[acting_player].act(state)
        if (
            acting_player == agent_player
            and phase == "card"
            and is_card_play_action(action)
        ):
            slot = action // 2
            if 0 <= slot < len(state.hands[agent_player]):
                card = state.hands[agent_player][slot]
                if len(state.expeditions[agent_player][card.color]) == 0:
                    summary = color_playability_summary(state, agent_player, card.color)
                    first_open_recoverable_by_color.setdefault(
                        int(card.color),
                        float(summary["recoverable_score_no_bonus"]),
                    )
        state.apply_action(action)
    return state, not state.terminal, first_open_recoverable_by_color, max_steps


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_games(
    *,
    checkpoint_path: Path,
    checkpoint_iteration: int,
    opponent_name: str,
    games: int,
    seed: int,
    final_scores: list[float],
    bonus_flags: list[bool],
    first_open_recoverable_positive: list[float],
    first_open_recoverable_negative: list[float],
    timed_out_games: int,
    game_lengths: list[int],
) -> dict[str, Any]:
    positive = [score for score in final_scores if score > 0]
    negative = [score for score in final_scores if score < 0]
    breakeven = [score for score in final_scores if score == 0]
    below_minus_20 = [score for score in final_scores if score < -20]
    opened_count = len(final_scores)
    bonus_count = sum(1 for flag in bonus_flags if flag)

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.parent.name,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_iteration": int(checkpoint_iteration),
        "opponent": opponent_name,
        "games": int(games),
        "seed": int(seed),
        "timed_out_games": int(timed_out_games),
        "avg_game_length": mean_or_none([float(length) for length in game_lengths]),
        "opened_expeditions": int(opened_count),
        "per_game_positive_expeditions": len(positive) / max(1, games),
        "per_game_negative_expeditions": len(negative) / max(1, games),
        "per_game_breakeven_expeditions": len(breakeven) / max(1, games),
        "per_game_bonus_expeditions": bonus_count / max(1, games),
        "per_game_opened_but_negative_expeditions": len(negative) / max(1, games),
        "per_game_below_minus_20_expeditions": len(below_minus_20) / max(1, games),
        "positive_expedition_rate": len(positive) / max(1, opened_count),
        "negative_expedition_rate": len(negative) / max(1, opened_count),
        "bonus_expedition_rate": bonus_count / max(1, opened_count),
        "avg_final_score_per_opened_expedition": mean_or_none(final_scores),
        "final_expedition_score_p25": percentile(final_scores, 25),
        "final_expedition_score_median": percentile(final_scores, 50),
        "final_expedition_score_p75": percentile(final_scores, 75),
        "final_expedition_score_p90": percentile(final_scores, 90),
        "positive_expedition_score_mean": mean_or_none(positive),
        "negative_expedition_score_mean": mean_or_none(negative),
        "first_open_recoverable_score_mean_for_positive_final": mean_or_none(
            first_open_recoverable_positive
        ),
        "first_open_recoverable_score_mean_for_negative_final": mean_or_none(
            first_open_recoverable_negative
        ),
    }


def diagnose_checkpoint(
    checkpoint_path: Path,
    *,
    opponents: tuple[str, ...],
    games: int,
    seed: int,
    device: torch.device,
    sample: bool,
    max_steps_override: int | None,
    on_max_steps_override: str | None,
) -> list[dict[str, Any]]:
    payload, base_bot, lc_config, run_config = load_checkpoint_bot(
        checkpoint_path,
        device=device,
        sample=sample,
        seed=seed,
    )
    checkpoint_iteration = int(payload.get("iteration", 0))
    max_steps = int(max_steps_override or run_config.evaluation.max_steps)
    _on_max_steps = on_max_steps_override or run_config.evaluation.on_max_steps

    rows: list[dict[str, Any]] = []
    for opponent_index, opponent_name in enumerate(opponents):
        final_scores: list[float] = []
        bonus_flags: list[bool] = []
        first_open_recoverable_positive: list[float] = []
        first_open_recoverable_negative: list[float] = []
        timed_out_games = 0
        game_lengths: list[int] = []
        opponent_seed = seed + opponent_index * 1_000_003

        for game_index in range(games):
            # Recreate bots per game to mirror existing eval's per-game StrategyNetBot factory.
            agent_bot = StrategyNetBot(
                base_bot.strategy_net,
                lc_config,
                device=device,
                encoding=base_bot.encoding,
                sample=sample,
                seed=opponent_seed + game_index,
            )
            opponent_bot = make_opponent(opponent_name, seed=opponent_seed + 50_000 + game_index)
            agent_player = 0 if game_index % 2 == 0 else 1
            final_state, timed_out, first_open_recoverable_by_color, game_length = (
                play_diagnostic_game(
                    agent_bot,
                    opponent_bot,
                    lc_config,
                    seed=opponent_seed + 100_000 + game_index,
                    agent_player=agent_player,
                    max_steps=max_steps,
                )
            )
            if timed_out:
                timed_out_games += 1
            game_lengths.append(game_length)

            for color in range(lc_config.n_colors):
                expedition = final_state.expeditions[agent_player][color]
                if not expedition:
                    continue
                score = float(final_state.expedition_score(agent_player, color))
                final_scores.append(score)
                bonus_flags.append(len(expedition) >= lc_config.bonus_threshold)
                first_open_recoverable = first_open_recoverable_by_color.get(color)
                if first_open_recoverable is None:
                    continue
                if score > 0:
                    first_open_recoverable_positive.append(first_open_recoverable)
                elif score < 0:
                    first_open_recoverable_negative.append(first_open_recoverable)

        rows.append(
            summarize_games(
                checkpoint_path=checkpoint_path,
                checkpoint_iteration=checkpoint_iteration,
                opponent_name=opponent_name,
                games=games,
                seed=opponent_seed,
                final_scores=final_scores,
                bonus_flags=bonus_flags,
                first_open_recoverable_positive=first_open_recoverable_positive,
                first_open_recoverable_negative=first_open_recoverable_negative,
                timed_out_games=timed_out_games,
                game_lengths=game_lengths,
            )
        )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    opponents = tuple(args.opponent or DEFAULT_OPPONENTS)
    device = resolve_device(args.device)
    rows: list[dict[str, Any]] = []
    for checkpoint in args.checkpoint:
        rows.extend(
            diagnose_checkpoint(
                checkpoint,
                opponents=opponents,
                games=args.games,
                seed=args.seed,
                device=device,
                sample=args.sample,
                max_steps_override=args.max_steps,
                on_max_steps_override=args.on_max_steps,
            )
        )

    payload = {
        "diagnostic": "expedition_score_distribution",
        "games": int(args.games),
        "seed": int(args.seed),
        "opponents": list(opponents),
        "checkpoints": [str(path) for path in args.checkpoint],
        "rows": rows,
    }
    write_json(args.output, payload)
    if args.jsonl_output is not None:
        write_jsonl(args.jsonl_output, rows)

    print(f"wrote JSON: {args.output}")
    if args.jsonl_output is not None:
        print(f"wrote JSONL: {args.jsonl_output}")
    for row in rows:
        print(
            "{checkpoint_name} iter={checkpoint_iteration} opponent={opponent} "
            "positive_rate={positive_expedition_rate:.3f} "
            "avg_score={avg_final_score_per_opened_expedition} "
            "p50={final_expedition_score_median}".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
