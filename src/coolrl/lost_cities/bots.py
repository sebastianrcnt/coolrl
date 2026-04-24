from __future__ import annotations

from dataclasses import replace
from typing import Protocol, TypeAlias, runtime_checkable

from .game import Card, GameState, LostCitiesConfig

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for Lost Cities bots") from exc


BotInput: TypeAlias = dict | GameState


@runtime_checkable
class LostCitiesBot(Protocol):
    def act(self, obs_or_state: BotInput) -> int:
        """Choose an action id from the current state or observation."""


def _legal_from_obs(obs_or_state: BotInput) -> np.ndarray:
    if isinstance(obs_or_state, GameState):
        return np.asarray(obs_or_state.legal_mask(), dtype=bool)
    return np.asarray(obs_or_state["legal_mask"], dtype=bool)


class RandomBot(LostCitiesBot):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs_or_state: BotInput) -> int:
        legal = _legal_from_obs(obs_or_state)
        legal_indices = np.nonzero(legal)[0]
        if len(legal_indices) == 0:
            raise RuntimeError("no legal action available")
        return int(self.rng.choice(legal_indices))


class SafeHeuristicBot(LostCitiesBot):
    def act(self, obs_or_state: BotInput) -> int:
        if not isinstance(obs_or_state, GameState):
            legal = _legal_from_obs(obs_or_state)
            legal_indices = np.nonzero(legal)[0]
            if len(legal_indices) == 0:
                raise RuntimeError("no legal action available")
            return int(legal_indices[0])
        if obs_or_state.phase == "card":
            return self._act_card(obs_or_state)
        return self._act_draw(obs_or_state)

    def _act_card(self, state: GameState) -> int:
        player = state.current_player
        hand = state.hands[player]
        legal = state.legal_card_mask()
        middle_rank = (state.config.n_ranks + 1) // 2

        for slot, card in enumerate(hand):
            action = 2 * slot
            if not legal[action] or not card.is_handshake:
                continue
            support = 0
            for other_slot, other in enumerate(hand):
                if other_slot == slot or other.color != card.color:
                    continue
                if other.is_handshake or other.rank >= middle_rank:
                    support += 1
            if support >= 2:
                return action

        play_candidates: list[tuple[int, int, int]] = []
        for slot, card in enumerate(hand):
            action = 2 * slot
            if not legal[action] or card.is_handshake:
                continue
            expedition_started = len(state.expeditions[player][card.color]) > 0
            if expedition_started:
                play_candidates.append((0, card.rank, action))
                continue
            same_color_numbers = [
                other for other in hand
                if other.color == card.color and not other.is_handshake
            ]
            high_numbers = [
                other for other in same_color_numbers
                if other.rank >= middle_rank
            ]
            if len(same_color_numbers) >= 3 and len(high_numbers) >= 2:
                play_candidates.append((1, card.rank, action))
        if play_candidates:
            return min(play_candidates)[2]

        discard_candidates: list[tuple[int, int, int, int]] = []
        opponent = 1 - player
        for slot, card in enumerate(hand):
            action = 2 * slot + 1
            if not legal[action]:
                continue
            opponent_started = int(len(state.expeditions[opponent][card.color]) > 0)
            handshake_risk = 10 if card.is_handshake else 0
            discard_candidates.append(
                (opponent_started, card.rank + handshake_risk, card.color, action)
            )
        if discard_candidates:
            return min(discard_candidates)[3]
        return self._first_legal(legal)

    def _act_draw(self, state: GameState) -> int:
        legal = state.legal_draw_mask()
        player = state.current_player
        for color in range(state.config.n_colors):
            action = 1 + color
            if not legal[action] or not state.discards[color]:
                continue
            card = state.discards[color][-1]
            if self._can_use_now(state, player, card):
                return action
        if legal[0]:
            return 0
        return self._first_legal(legal)

    def _can_use_now(self, state: GameState, player: int, card: Card) -> bool:
        return state.can_play_card(player, card)

    def _first_legal(self, legal: list[bool] | np.ndarray) -> int:
        legal_indices = np.nonzero(np.asarray(legal, dtype=bool))[0]
        if len(legal_indices) == 0:
            raise RuntimeError("no legal action available")
        return int(legal_indices[0])


def play_game(
    bot0: LostCitiesBot,
    bot1: LostCitiesBot,
    config: LostCitiesConfig,
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> GameState:
    game_config = replace(config, seed=seed) if seed is not None else config
    state = GameState.new_game(game_config)
    bots = [bot0, bot1]
    for _ in range(max_steps):
        if state.terminal:
            return state
        action = bots[state.current_player].act(state)
        state.apply_action(action)
    raise RuntimeError(f"game exceeded max_steps={max_steps}")


def run_series(
    bot0: LostCitiesBot,
    bot1: LostCitiesBot,
    config: LostCitiesConfig,
    *,
    games: int = 100,
    seed: int = 0,
) -> dict:
    diffs: list[int] = []
    wins0 = 0
    wins1 = 0
    draws = 0
    for index in range(games):
        state = play_game(bot0, bot1, config, seed=seed + index)
        diff = state.score_diff(0)
        diffs.append(diff)
        if diff > 0:
            wins0 += 1
        elif diff < 0:
            wins1 += 1
        else:
            draws += 1
    return {
        "games": games,
        "avg_diff": float(np.mean(diffs)) if diffs else 0.0,
        "wins0": wins0,
        "wins1": wins1,
        "draws": draws,
    }
