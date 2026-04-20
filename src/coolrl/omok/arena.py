from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from loguru import logger

from coolrl.progress import make_progress

from .board import GameState
from .evaluator import ModelEvaluator
from .mcts_backend import resolve_mcts_backend
from .mcts_types import MCTSBackend
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


@dataclass(slots=True)
class ArenaGame:
    state: GameState
    candidate_color: int
    candidate_root: object | None = None
    best_root: object | None = None


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
        search_threads: int = 1,
        mcts_backend: str = "python",
        candidate_evaluator: ModelEvaluator | None = None,
        best_evaluator: ModelEvaluator | None = None,
    ) -> None:
        self.board_size = board_size
        self.exactly_five = exactly_five
        self.simulations = simulations
        self.c_puct = c_puct
        self.leaves_per_batch = leaves_per_batch
        self.search_threads = max(1, int(search_threads))
        self.mcts_module = resolve_mcts_backend(mcts_backend)
        self.candidate_evaluator = candidate_evaluator or ModelEvaluator(candidate_model, device=device)
        self.best_evaluator = best_evaluator or ModelEvaluator(best_model, device=device)

    def evaluate(self, games: int) -> ArenaResult:
        target_games = max(2, games)
        pair_count = max(1, target_games // 2)
        openings = build_opening_sequences(self.board_size, pair_count)
        active_games = self._build_games(openings)
        result = ArenaResult(
            games=pair_count * 2,
            candidate_wins=0,
            best_wins=0,
            draws=0,
            candidate_black_wins=0,
            candidate_white_wins=0,
        )

        candidate_search = self.mcts_module.MCTS(
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=self.candidate_evaluator,
            search_threads=self.search_threads,
        )
        best_search = self.mcts_module.MCTS(
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=self.best_evaluator,
            search_threads=self.search_threads,
        )

        with make_progress() as progress:
            task = progress.add_task("Arena", total=len(active_games), status="games")
            while active_games:
                candidate_turns = [
                    game
                    for game in active_games
                    if not game.state.terminal and game.state.to_play == game.candidate_color
                ]
                best_turns = [
                    game
                    for game in active_games
                    if not game.state.terminal and game.state.to_play != game.candidate_color
                ]
                if candidate_turns:
                    self._advance_games(candidate_turns, candidate_search, candidate=True)
                if best_turns:
                    self._advance_games(best_turns, best_search, candidate=False)

                still_active: list[ArenaGame] = []
                for game in active_games:
                    if game.state.terminal:
                        self._record_result(result, game)
                        progress.update(task, advance=1, status="games")
                    else:
                        still_active.append(game)
                active_games = still_active

        logger.info(
            "Arena finished: games={} candidate_wins={} best_wins={} draws={} win_rate={:.3f}",
            result.games,
            result.candidate_wins,
            result.best_wins,
            result.draws,
            result.candidate_win_rate,
        )
        return result

    def _build_games(self, openings: Sequence[Sequence[int]]) -> list[ArenaGame]:
        games: list[ArenaGame] = []
        for opening in openings:
            for candidate_color in (1, -1):
                state = GameState(board_size=self.board_size, exactly_five=self.exactly_five)
                for action in opening:
                    state.apply_action(action)
                games.append(ArenaGame(state=state, candidate_color=candidate_color))
        return games

    def _advance_games(self, games: list[ArenaGame], search: MCTSBackend, candidate: bool) -> None:
        states = [game.state for game in games]
        roots = [game.candidate_root if candidate else game.best_root for game in games]
        results = search.search_batch(
            states,
            self.simulations,
            [0.0] * len(states),
            add_noise=False,
            roots=roots,
            leaves_per_batch=self.leaves_per_batch,
        )
        for game, result in zip(games, results, strict=True):
            if candidate:
                game.candidate_root = result.next_root
                if game.best_root is not None:
                    game.best_root = game.best_root.children.get(result.action)
            else:
                game.best_root = result.next_root
                if game.candidate_root is not None:
                    game.candidate_root = game.candidate_root.children.get(result.action)
            game.state.apply_action(result.action)

    def _record_result(self, result: ArenaResult, game: ArenaGame) -> None:
        winner = game.state.winner
        if winner == 0:
            result.draws += 1
        elif winner == game.candidate_color:
            result.candidate_wins += 1
            if game.candidate_color == 1:
                result.candidate_black_wins += 1
            else:
                result.candidate_white_wins += 1
        else:
            result.best_wins += 1
