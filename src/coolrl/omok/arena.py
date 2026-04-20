from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from loguru import logger
from tqdm import tqdm

from .board import GameState
from .evaluator import ModelEvaluator
from .mcts import MCTS
from .network import PolicyValueNet
from .openings import build_opening_sequences


@dataclass(slots=True)
class ArenaResult:
    games: int
    candidate_wins: int
    best_wins: int
    draws: int
    candidate_black_wins: int
    candidate_white_wins: int

    @property
    def candidate_win_rate(self) -> float:
        return 0.0 if self.games == 0 else self.candidate_wins / self.games


class Arena:
    def __init__(
        self,
        candidate_model: PolicyValueNet,
        best_model: PolicyValueNet,
        device: str | None,
        board_size: int,
        exactly_five: bool,
        simulations: int,
        c_puct: float,
        leaves_per_batch: int = 1,
    ) -> None:
        self.board_size = board_size
        self.exactly_five = exactly_five
        self.simulations = simulations
        self.c_puct = c_puct
        self.leaves_per_batch = leaves_per_batch
        self.candidate_evaluator = ModelEvaluator(candidate_model, device=device)
        self.best_evaluator = ModelEvaluator(best_model, device=device)

    def evaluate(self, games: int) -> ArenaResult:
        target_games = max(2, games)
        pair_count = max(1, target_games // 2)
        openings = build_opening_sequences(self.board_size, pair_count)
        candidate_wins = 0
        best_wins = 0
        draws = 0
        candidate_black_wins = 0
        candidate_white_wins = 0

        for opening in tqdm(openings, desc="Arena", unit="pair", leave=False):
            wins, losses, pair_draws, black_wins, white_wins = self._play_opening_pair(opening)
            candidate_wins += wins
            best_wins += losses
            draws += pair_draws
            candidate_black_wins += black_wins
            candidate_white_wins += white_wins

        result = ArenaResult(
            games=pair_count * 2,
            candidate_wins=candidate_wins,
            best_wins=best_wins,
            draws=draws,
            candidate_black_wins=candidate_black_wins,
            candidate_white_wins=candidate_white_wins,
        )
        logger.info(
            "Arena finished: games={} candidate_wins={} best_wins={} draws={} win_rate={:.3f}",
            result.games,
            result.candidate_wins,
            result.best_wins,
            result.draws,
            result.candidate_win_rate,
        )
        return result

    def _play_opening_pair(self, opening: Sequence[int]) -> tuple[int, int, int, int, int]:
        candidate_wins = 0
        best_wins = 0
        draws = 0
        candidate_black_wins = 0
        candidate_white_wins = 0
        for candidate_color in (1, -1):
            winner = self._play_game(opening, candidate_color)
            if winner == 0:
                draws += 1
            elif winner == candidate_color:
                candidate_wins += 1
                if candidate_color == 1:
                    candidate_black_wins += 1
                else:
                    candidate_white_wins += 1
            else:
                best_wins += 1
        return candidate_wins, best_wins, draws, candidate_black_wins, candidate_white_wins

    def _play_game(self, opening: Sequence[int], candidate_color: int) -> int:
        candidate_search = MCTS(
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=self.candidate_evaluator,
        )
        best_search = MCTS(
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=self.best_evaluator,
        )
        state = GameState(board_size=self.board_size, exactly_five=self.exactly_five)
        for action in opening:
            state.apply_action(action)

        candidate_root = None
        best_root = None
        while not state.terminal:
            if state.to_play == candidate_color:
                result = candidate_search.search_batch(
                    [state],
                    self.simulations,
                    [0.0],
                    add_noise=False,
                    roots=[candidate_root],
                    leaves_per_batch=self.leaves_per_batch,
                )[0]
                candidate_root = result.next_root
                if best_root is not None:
                    best_root = best_root.children.get(result.action)
            else:
                result = best_search.search_batch(
                    [state],
                    self.simulations,
                    [0.0],
                    add_noise=False,
                    roots=[best_root],
                    leaves_per_batch=self.leaves_per_batch,
                )[0]
                best_root = result.next_root
                if candidate_root is not None:
                    candidate_root = candidate_root.children.get(result.action)
            state.apply_action(result.action)
        return state.winner

