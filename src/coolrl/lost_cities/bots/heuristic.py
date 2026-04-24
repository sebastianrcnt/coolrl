from __future__ import annotations

from ..game import Card, GameState
from ..interfaces import BotInput, LostCitiesBot
from .base import first_legal, legal_from_obs

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for Lost Cities bots") from exc


class SafeHeuristicBot(LostCitiesBot):
    def act(self, obs_or_state: BotInput) -> int:
        if not isinstance(obs_or_state, GameState):
            return first_legal(legal_from_obs(obs_or_state))
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
        return first_legal(legal)

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
        return first_legal(legal)

    def _can_use_now(self, state: GameState, player: int, card: Card) -> bool:
        return state.can_play_card(player, card)
