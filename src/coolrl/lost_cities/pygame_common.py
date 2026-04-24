from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Literal

from .game import Card, GameState, LostCitiesConfig, build_deck, score_expedition

BackendName = Literal["python", "rust"]
LOGGER = logging.getLogger("coolrl.lost_cities.pygame")


@dataclass
class Snapshot:
    config: LostCitiesConfig
    deck: list[Card]
    hands: list[list[Card]]
    expeditions: list[list[list[Card]]]
    discards: list[list[Card]]
    current_player: int
    phase: str
    pending_discarded_color: int | None
    turn_count: int
    terminal: bool
    legal_mask: list[bool]

    @property
    def card_action_size(self) -> int:
        return self.config.card_action_size

    @property
    def draw_action_size(self) -> int:
        return self.config.draw_action_size

    def expedition_score(self, player: int, color: int) -> int:
        return score_expedition(self.expeditions[player][color], self.config)

    def total_score(self, player: int) -> int:
        return sum(
            self.expedition_score(player, color)
            for color in range(self.config.n_colors)
        )

    def score_diff(self, player: int = 0) -> int:
        return self.total_score(player) - self.total_score(1 - player)


class GameBackend:
    name: BackendName

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        self.config = config
        self.seed = seed

    def snapshot(self) -> Snapshot:
        raise NotImplementedError

    def apply(self, action_id: int) -> None:
        raise NotImplementedError

    def can_undo(self) -> bool:
        raise NotImplementedError

    def undo(self) -> bool:
        raise NotImplementedError


class PythonBackend(GameBackend):
    name: BackendName = "python"

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        super().__init__(config, seed)
        self.state = GameState.new_game(config, seed=seed)
        self.history: list[GameState] = []
        LOGGER.debug("파이썬 백엔드 초기화: %s", snapshot_summary(self.snapshot()))

    def snapshot(self) -> Snapshot:
        return _snapshot_from_state(self.state)

    def apply(self, action_id: int) -> None:
        before = self.snapshot()
        self.history.append(self.state.clone())
        self.state.apply_unified_action(action_id)
        LOGGER.debug(
            "파이썬 액션 적용: 액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            action_id,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.history),
        )

    def can_undo(self) -> bool:
        return bool(self.history)

    def undo(self) -> bool:
        if not self.history:
            LOGGER.debug("파이썬 되돌리기 무시: 기록이 비어 있음")
            return False
        before = self.snapshot()
        self.state = self.history.pop()
        LOGGER.debug(
            "파이썬 되돌리기: 이전={%s} 이후={%s} 되돌리기깊이=%s",
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.history),
        )
        return True


class RustBackend(GameBackend):
    name: BackendName = "rust"

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        super().__init__(config, seed)
        self.initial_deck = _shuffled_deck(config, seed)
        self.actions: list[int] = []
        self._snapshot = self._run_trace()
        LOGGER.debug("러스트 백엔드 초기화: %s", snapshot_summary(self.snapshot()))

    def snapshot(self) -> Snapshot:
        return self._snapshot

    def apply(self, action_id: int) -> None:
        before = self.snapshot()
        self.actions.append(action_id)
        try:
            self._snapshot = self._run_trace()
        except Exception:
            self.actions.pop()
            raise
        LOGGER.debug(
            "러스트 액션 적용: 액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            action_id,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.actions),
        )

    def can_undo(self) -> bool:
        return bool(self.actions)

    def undo(self) -> bool:
        if not self.actions:
            LOGGER.debug("러스트 되돌리기 무시: 액션 기록이 비어 있음")
            return False
        before = self.snapshot()
        removed = self.actions.pop()
        self._snapshot = self._run_trace()
        LOGGER.debug(
            "러스트 되돌리기: 제거한액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            removed,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.actions),
        )
        return True

    def _run_trace(self) -> Snapshot:
        fixture = {
            "config": self.config.to_snapshot(),
            "initial_deck": [card.to_snapshot() for card in self.initial_deck],
            "steps": [{"action": None}]
            + [{"action": action} for action in self.actions],
        }
        rust_core = Path(__file__).resolve().parent / "rust_core"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(fixture, handle)
            fixture_path = Path(handle.name)
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--bin",
                    "lost_cities_probe",
                    "--",
                    "trace",
                    str(fixture_path),
                ],
                cwd=rust_core,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(f"rust backend failed: {message}") from exc
        finally:
            fixture_path.unlink(missing_ok=True)

        trace = json.loads(result.stdout)
        return _snapshot_from_trace(trace["config"], trace["steps"][-1])


def build_backend(
    backend: BackendName,
    config: LostCitiesConfig,
    seed: int | None,
) -> GameBackend:
    if backend == "python":
        return PythonBackend(config, seed)
    if backend == "rust":
        return RustBackend(config, seed)
    raise ValueError(f"unknown backend: {backend}")


def snapshot_summary(snapshot: Snapshot) -> str:
    scores = [snapshot.total_score(0), snapshot.total_score(1)]
    hand_sizes = [len(hand) for hand in snapshot.hands]
    discard_sizes = [len(discard) for discard in snapshot.discards]
    phase = "카드" if snapshot.phase == "card" else "뽑기"
    return (
        f"플레이어={snapshot.current_player} 단계={phase} "
        f"턴={snapshot.turn_count} 종료={snapshot.terminal} "
        f"덱={len(snapshot.deck)} 손패수={hand_sizes} 점수={scores} "
        f"직전버린색={snapshot.pending_discarded_color} "
        f"버린더미수={discard_sizes}"
    )


def _shuffled_deck(config: LostCitiesConfig, seed: int | None) -> list[Card]:
    deck = build_deck(config)
    rng = random.Random(config.seed if seed is None else seed)
    rng.shuffle(deck)
    return deck


def _snapshot_from_state(state: GameState) -> Snapshot:
    return Snapshot(
        config=state.config,
        deck=list(state.deck),
        hands=[list(hand) for hand in state.hands],
        expeditions=[
            [list(expedition) for expedition in player_expeditions]
            for player_expeditions in state.expeditions
        ],
        discards=[list(discard) for discard in state.discards],
        current_player=state.current_player,
        phase=state.phase,
        pending_discarded_color=state.pending_discarded_color,
        turn_count=state.turn_count,
        terminal=state.terminal,
        legal_mask=state.unified_legal_mask(),
    )


def _snapshot_from_trace(config_data: dict[str, Any], step: dict[str, Any]) -> Snapshot:
    config = LostCitiesConfig(**config_data)
    return Snapshot(
        config=config,
        deck=_cards_from_json(step["deck"]),
        hands=[_cards_from_json(hand) for hand in step["hands"]],
        expeditions=[
            [_cards_from_json(expedition) for expedition in player_expeditions]
            for player_expeditions in step["expeditions"]
        ],
        discards=[_cards_from_json(discard) for discard in step["discards"]],
        current_player=int(step["current_player"]),
        phase=str(step["phase"]),
        pending_discarded_color=step.get("pending_discarded_color"),
        turn_count=int(step["turn_count"]),
        terminal=bool(step["terminal"]),
        legal_mask=list(step["legal_mask"]),
    )


def _cards_from_json(cards: list[dict[str, int]]) -> list[Card]:
    return [Card.from_snapshot(card) for card in cards]
