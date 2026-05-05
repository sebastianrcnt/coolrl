from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..game import Card, GameState, LostCitiesConfig


@lru_cache(maxsize=None)
def _encoding_layout(
    n_colors: int,
    n_ranks: int,
    n_handshakes: int,
    hand_size: int,
) -> dict[str, int | float]:
    card_type_size = n_colors * (n_ranks + 1)
    card_slot_size = card_type_size + 1
    expedition_block = card_type_size + 2
    discard_block = card_type_size + 1 + card_slot_size
    total = (
        2  # phase one-hot
        + 1  # current_player matches
        + 1  # player id
        + hand_size * card_slot_size
        + 2 * n_colors * expedition_block
        + n_colors * discard_block
        + card_type_size  # public counts
        + 2  # deck ratio + turn count ratio
        + n_colors + 1  # pending discard
    )
    return {
        "n_colors": n_colors,
        "n_ranks": n_ranks,
        "rank_plus_one": n_ranks + 1,
        "card_type_size": card_type_size,
        "card_slot_size": card_slot_size,
        "total": total,
        "denom_handshake": float(max(1, n_handshakes)),
        "len_denom": float(max(1, n_handshakes + n_ranks)),
        "rank_denom": float(max(1, n_ranks)),
        "deck_denom": float(max(1, n_colors * (n_ranks + n_handshakes))),
        "turn_denom": float(max(1, 2 * n_colors * (n_ranks + n_handshakes))),
    }


def _layout_for(config: LostCitiesConfig) -> dict[str, int | float]:
    return _encoding_layout(
        config.n_colors,
        config.n_ranks,
        config.n_handshakes,
        config.hand_size,
    )


def _accumulate_card_counts(
    out: np.ndarray,
    base: int,
    cards: list[Card],
    rank_plus_one: int,
    denom_handshake: float,
) -> None:
    for card in cards:
        pos = base + card.color * rank_plus_one + card.rank
        if card.rank == 0:
            out[pos] += 1.0 / denom_handshake
        else:
            out[pos] += 1.0


def encode_information_state(state: GameState, player: int) -> np.ndarray:
    config = state.config
    layout = _layout_for(config)
    out = np.zeros(layout["total"], dtype=np.float32)
    n_colors = layout["n_colors"]
    rank_plus_one = layout["rank_plus_one"]
    card_type_size = layout["card_type_size"]
    card_slot_size = layout["card_slot_size"]
    denom_handshake = layout["denom_handshake"]
    len_denom = layout["len_denom"]
    rank_denom = layout["rank_denom"]

    idx = 0
    if state.phase == "card":
        out[idx] = 1.0
    else:
        out[idx + 1] = 1.0
    idx += 2

    if state.current_player == player:
        out[idx] = 1.0
    idx += 1

    out[idx] = float(player)
    idx += 1

    hand = state.hand_slots(player)
    for card in hand:
        if card is None:
            out[idx + card_type_size] = 1.0
        else:
            out[idx + card.color * rank_plus_one + card.rank] = 1.0
        idx += card_slot_size

    other = 1 - player
    for p in (player, other):
        for color in range(n_colors):
            cards = state.expeditions[p][color]
            _accumulate_card_counts(out, idx, cards, rank_plus_one, denom_handshake)
            idx += card_type_size
            out[idx] = len(cards) / len_denom
            idx += 1
            out[idx] = state.last_numeric_rank(p, color) / rank_denom
            idx += 1

    for color in range(n_colors):
        pile = state.discards[color]
        _accumulate_card_counts(out, idx, pile, rank_plus_one, denom_handshake)
        idx += card_type_size
        out[idx] = len(pile) / len_denom
        idx += 1
        if pile:
            top = pile[-1]
            out[idx + top.color * rank_plus_one + top.rank] = 1.0
        else:
            out[idx + card_type_size] = 1.0
        idx += card_slot_size

    public_base = idx
    for p in range(2):
        for color in range(n_colors):
            _accumulate_card_counts(
                out, public_base, state.expeditions[p][color], rank_plus_one, denom_handshake
            )
    for color in range(n_colors):
        _accumulate_card_counts(
            out, public_base, state.discards[color], rank_plus_one, denom_handshake
        )
    idx += card_type_size

    out[idx] = len(state.deck) / layout["deck_denom"]
    idx += 1
    out[idx] = state.turn_count / layout["turn_denom"]
    idx += 1

    if state.pending_discarded_color is None:
        out[idx + n_colors] = 1.0
    else:
        out[idx + state.pending_discarded_color] = 1.0

    return out


def legal_mask_array(state: GameState) -> np.ndarray:
    return state.unified_legal_mask_np()


def infer_input_dim(config: LostCitiesConfig) -> int:
    return int(_layout_for(config)["total"])
