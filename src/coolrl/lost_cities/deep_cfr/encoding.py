from __future__ import annotations

import numpy as np

from ..game import Card, GameState, LostCitiesConfig


def _card_type_index(card: Card, config: LostCitiesConfig) -> int:
    return card.color * (config.n_ranks + 1) + card.rank


def _append_card_type(out: list[float], card: Card | None, config: LostCitiesConfig) -> None:
    size = config.n_colors * (config.n_ranks + 1)
    values = [0.0] * (size + 1)
    if card is None:
        values[-1] = 1.0
    else:
        values[_card_type_index(card, config)] = 1.0
    out.extend(values)


def _card_counts(cards: list[Card], config: LostCitiesConfig) -> list[float]:
    size = config.n_colors * (config.n_ranks + 1)
    counts = [0.0] * size
    denom = max(1, config.n_handshakes)
    for card in cards:
        scale = denom if card.is_handshake else 1
        counts[_card_type_index(card, config)] += 1.0 / scale
    return counts


def _append_expeditions(out: list[float], state: GameState, player: int) -> None:
    for color in range(state.config.n_colors):
        cards = state.expeditions[player][color]
        out.extend(_card_counts(cards, state.config))
        out.append(len(cards) / max(1, state.config.n_handshakes + state.config.n_ranks))
        out.append(state.last_numeric_rank(player, color) / max(1, state.config.n_ranks))


def _append_discards(out: list[float], state: GameState) -> None:
    for color in range(state.config.n_colors):
        pile = state.discards[color]
        out.extend(_card_counts(pile, state.config))
        out.append(len(pile) / max(1, state.config.n_handshakes + state.config.n_ranks))
        _append_card_type(out, pile[-1] if pile else None, state.config)


def encode_information_state(state: GameState, player: int) -> np.ndarray:
    config = state.config
    other = 1 - player
    values: list[float] = []
    values.extend([1.0, 0.0] if state.phase == "card" else [0.0, 1.0])
    values.append(1.0 if state.current_player == player else 0.0)
    values.append(float(player))

    hand = state.hand_slots(player)
    for card in hand:
        _append_card_type(values, card, config)

    _append_expeditions(values, state, player)
    _append_expeditions(values, state, other)
    _append_discards(values, state)

    public_cards: list[Card] = []
    for p in range(2):
        for expedition in state.expeditions[p]:
            public_cards.extend(expedition)
    for pile in state.discards:
        public_cards.extend(pile)
    values.extend(_card_counts(public_cards, config))

    values.append(len(state.deck) / max(1, config.deck_size))
    values.append(state.turn_count / max(1, config.deck_size * 2))

    pending = [0.0] * (config.n_colors + 1)
    if state.pending_discarded_color is None:
        pending[-1] = 1.0
    else:
        pending[state.pending_discarded_color] = 1.0
    values.extend(pending)
    return np.asarray(values, dtype=np.float32)


def legal_mask_array(state: GameState) -> np.ndarray:
    return np.asarray(state.unified_legal_mask(), dtype=bool)


def infer_input_dim(config: LostCitiesConfig) -> int:
    state = GameState.new_game(config, seed=config.seed)
    return int(encode_information_state(state, 0).shape[0])
