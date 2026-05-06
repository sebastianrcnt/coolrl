from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..game import Card, GameState, LostCitiesConfig
from .config import EncodingConfig

DERIVED_PLAYABILITY_PER_COLOR = 19
DERIVED_PLAYABILITY_COMMON = 3
SLOT_AWARE_PLAYABILITY_PER_SLOT = 12


@lru_cache(maxsize=None)
def _encoding_layout(
    n_colors: int,
    n_ranks: int,
    n_handshakes: int,
    hand_size: int,
    min_rank: int,
    expedition_penalty: int,
    bonus_threshold: int,
    derived_playability: bool,
    slot_aware_playability: bool,
) -> dict[str, int | float | bool]:
    card_type_size = n_colors * (n_ranks + 1)
    card_slot_size = card_type_size + 1
    expedition_block = card_type_size + 2
    discard_block = card_type_size + 1 + card_slot_size
    base_total = (
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
    max_numeric_sum = sum(min_rank + rank - 1 for rank in range(1, n_ranks + 1))
    max_cards_per_color = n_ranks + n_handshakes
    max_wagers = n_handshakes
    break_even = abs(expedition_penalty)
    max_score_estimate = max(1, (max_numeric_sum - break_even) * (max_wagers + 1))
    derived_total = (
        n_colors * DERIVED_PLAYABILITY_PER_COLOR + DERIVED_PLAYABILITY_COMMON
        if derived_playability
        else 0
    )
    slot_aware_total = hand_size * SLOT_AWARE_PLAYABILITY_PER_SLOT if slot_aware_playability else 0
    return {
        "n_colors": n_colors,
        "n_ranks": n_ranks,
        "min_rank": min_rank,
        "expedition_penalty": expedition_penalty,
        "bonus_threshold": bonus_threshold,
        "rank_plus_one": n_ranks + 1,
        "card_type_size": card_type_size,
        "card_slot_size": card_slot_size,
        "base_total": base_total,
        "derived_playability": derived_playability,
        "slot_aware_playability": slot_aware_playability,
        "derived_total": derived_total,
        "slot_aware_total": slot_aware_total,
        "total": base_total + derived_total + slot_aware_total,
        "denom_handshake": float(max(1, n_handshakes)),
        "len_denom": float(max(1, n_handshakes + n_ranks)),
        "rank_denom": float(max(1, n_ranks)),
        "deck_denom": float(max(1, n_colors * (n_ranks + n_handshakes))),
        "turn_denom": float(max(1, 2 * n_colors * (n_ranks + n_handshakes))),
        "max_numeric_sum": float(max(1, max_numeric_sum)),
        "max_cards_per_color": float(max(1, max_cards_per_color)),
        "max_wagers": float(max(1, max_wagers)),
        "break_even": float(break_even),
        "max_score_estimate": float(max_score_estimate),
    }


def _layout_for(
    config: LostCitiesConfig,
    encoding: EncodingConfig | None = None,
) -> dict[str, int | float | bool]:
    encoding = encoding or EncodingConfig()
    return _encoding_layout(
        config.n_colors,
        config.n_ranks,
        config.n_handshakes,
        config.hand_size,
        config.min_rank,
        config.expedition_penalty,
        config.bonus_threshold,
        bool(encoding.derived_playability),
        bool(encoding.slot_aware_playability),
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


def _numeric_value(card: Card, min_rank: int) -> int:
    return 0 if card.rank == 0 else min_rank + card.rank - 1


def color_playability_summary(state: GameState, player: int, color: int) -> dict[str, Any]:
    config = state.config
    min_rank = int(config.min_rank)
    hand_cards = [card for card in state.hands[player] if card.color == color]
    expedition = state.expeditions[player][color]
    last_numeric_rank = state.last_numeric_rank(player, color)

    current_numeric_sum = sum(_numeric_value(card, min_rank) for card in expedition if card.rank > 0)
    current_wager_count = sum(1 for card in expedition if card.rank == 0)
    playable_hand_numeric = [
        card for card in hand_cards if card.rank > 0 and card.rank > last_numeric_rank
    ]
    dead_hand_numeric = [
        card for card in hand_cards if card.rank > 0 and card.rank <= last_numeric_rank
    ]
    playable_hand_wager_count = sum(
        1 for card in hand_cards if card.rank == 0 and last_numeric_rank == 0
    )
    playable_hand_numeric_sum = sum(_numeric_value(card, min_rank) for card in playable_hand_numeric)
    dead_hand_numeric_sum = sum(_numeric_value(card, min_rank) for card in dead_hand_numeric)

    projected_numeric_sum = current_numeric_sum + playable_hand_numeric_sum
    projected_wager_count = current_wager_count + playable_hand_wager_count
    break_even = abs(int(config.expedition_penalty))
    recoverable_margin_no_bonus = projected_numeric_sum - break_even
    recoverable_score_no_bonus = recoverable_margin_no_bonus * (projected_wager_count + 1)
    min_needed_to_break_even = max(0, break_even - projected_numeric_sum)
    projected_len = len(expedition) + len(playable_hand_numeric) + playable_hand_wager_count
    has_bonus_path = projected_len >= int(config.bonus_threshold)
    cards_needed_for_bonus = max(0, int(config.bonus_threshold) - projected_len)

    discard_top_playable_flag = False
    discard_top_playable_value = 0
    pile = state.discards[color]
    if pile:
        top = pile[-1]
        if top.color == color and top.rank > 0 and top.rank > last_numeric_rank:
            discard_top_playable_flag = True
            discard_top_playable_value = _numeric_value(top, min_rank)

    known_color_count = len(hand_cards)
    known_color_count += sum(1 for card in expedition if card.color == color)
    known_color_count += sum(
        1 for card in state.expeditions[1 - player][color] if card.color == color
    )
    known_color_count += sum(1 for card in state.discards[color] if card.color == color)
    total_color_count = int(config.n_ranks + config.n_handshakes)
    unknown_remaining_count = max(0, total_color_count - known_color_count)

    return {
        "is_unopened": len(expedition) == 0,
        "has_only_wagers_opened": len(expedition) > 0 and all(card.rank == 0 for card in expedition),
        "current_expedition_numeric_sum": current_numeric_sum,
        "current_expedition_wager_count": current_wager_count,
        "current_expedition_len": len(expedition),
        "last_numeric_rank": last_numeric_rank,
        "hand_count": len(hand_cards),
        "hand_wager_count": sum(1 for card in hand_cards if card.rank == 0),
        "playable_hand_wager_count": playable_hand_wager_count,
        "playable_hand_numeric_sum": playable_hand_numeric_sum,
        "playable_hand_numeric_count": len(playable_hand_numeric),
        "dead_hand_numeric_count": len(dead_hand_numeric),
        "dead_hand_numeric_sum": dead_hand_numeric_sum,
        "recoverable_margin_no_bonus": recoverable_margin_no_bonus,
        "recoverable_score_no_bonus": recoverable_score_no_bonus,
        "min_needed_to_break_even": min_needed_to_break_even,
        "discard_top_playable_flag": discard_top_playable_flag,
        "discard_top_playable_value": discard_top_playable_value,
        "unknown_remaining_count": unknown_remaining_count,
        "has_bonus_path": has_bonus_path,
        "cards_needed_for_bonus": cards_needed_for_bonus,
    }


def _append_derived_playability_features(
    out: np.ndarray,
    idx: int,
    state: GameState,
    player: int,
    layout: dict[str, int | float | bool],
) -> int:
    max_numeric_sum = float(layout["max_numeric_sum"])
    max_cards_per_color = float(layout["max_cards_per_color"])
    max_wagers = float(layout["max_wagers"])
    max_score_estimate = float(layout["max_score_estimate"])

    for color in range(int(layout["n_colors"])):
        summary = color_playability_summary(state, player, color)
        values = (
            float(summary["is_unopened"]),
            float(summary["has_only_wagers_opened"]),
            float(summary["current_expedition_numeric_sum"]) / max_numeric_sum,
            float(summary["current_expedition_wager_count"]) / max_wagers,
            float(summary["current_expedition_len"]) / max_cards_per_color,
            float(summary["last_numeric_rank"]) / max_numeric_sum,
            float(summary["hand_count"]) / max_cards_per_color,
            float(summary["hand_wager_count"]) / max_wagers,
            float(summary["playable_hand_numeric_sum"]) / max_numeric_sum,
            float(summary["playable_hand_numeric_count"]) / max_cards_per_color,
            float(summary["dead_hand_numeric_count"]) / max_cards_per_color,
            float(summary["dead_hand_numeric_sum"]) / max_numeric_sum,
            float(summary["recoverable_margin_no_bonus"]) / max_numeric_sum,
            float(summary["recoverable_score_no_bonus"]) / max_score_estimate,
            float(summary["min_needed_to_break_even"]) / max_numeric_sum,
            float(summary["discard_top_playable_flag"]),
            float(summary["discard_top_playable_value"]) / max_numeric_sum,
            float(summary["unknown_remaining_count"]) / max_cards_per_color,
            float(summary["cards_needed_for_bonus"]) / max_cards_per_color,
        )
        if len(values) != DERIVED_PLAYABILITY_PER_COLOR:
            raise AssertionError(f"derived playability feature count mismatch: {len(values)}")
        for value in values:
            out[idx] = value
            idx += 1

    deck_remaining = len(state.deck)
    out[idx] = deck_remaining / float(layout["deck_denom"])
    idx += 1
    out[idx] = state.turn_count / float(layout["turn_denom"])
    idx += 1
    out[idx] = deck_remaining / float(layout["turn_denom"])
    idx += 1
    return idx


def _slot_playability_values(
    state: GameState,
    player: int,
    card: Card,
    layout: dict[str, int | float | bool],
) -> tuple[float, ...]:
    min_rank = int(layout["min_rank"])
    color = card.color
    summary = color_playability_summary(state, player, color)
    last_numeric_rank = int(summary["last_numeric_rank"])
    has_numeric_started = last_numeric_rank > 0
    legal_play = state.can_play_card(player, card)
    is_numeric = card.rank > 0
    is_wager = card.rank == 0

    # Opening selectivity is about starting or confirming a color commitment.
    would_start_color_commitment = legal_play and not has_numeric_started
    is_numeric_open = would_start_color_commitment and is_numeric
    is_wager_first_open = would_start_color_commitment and is_wager and bool(summary["is_unopened"])
    is_playable_to_existing = legal_play and has_numeric_started
    is_dead_numeric = is_numeric and not legal_play and card.rank <= last_numeric_rank
    is_wager_before_numeric = is_wager and legal_play and not has_numeric_started
    recoverable_score = float(summary["recoverable_score_no_bonus"])
    margin = float(summary["recoverable_margin_no_bonus"])
    is_bad_open_candidate = would_start_color_commitment and recoverable_score < 0.0
    open_risk_score = min(0.0, recoverable_score) if would_start_color_commitment else 0.0
    is_safe_continuation = (not would_start_color_commitment) and is_playable_to_existing

    return (
        recoverable_score / float(layout["max_score_estimate"]),
        margin / float(layout["max_numeric_sum"]),
        float(would_start_color_commitment),
        float(is_numeric_open),
        float(is_wager_first_open),
        float(is_playable_to_existing),
        float(is_dead_numeric),
        float(is_wager_before_numeric),
        float(summary["has_bonus_path"]),
        float(is_bad_open_candidate),
        open_risk_score / float(layout["max_score_estimate"]),
        float(is_safe_continuation),
    )


def _append_slot_aware_playability_features(
    out: np.ndarray,
    idx: int,
    state: GameState,
    player: int,
    layout: dict[str, int | float | bool],
) -> int:
    for card in state.hand_slots(player):
        if card is None:
            idx += SLOT_AWARE_PLAYABILITY_PER_SLOT
            continue
        values = _slot_playability_values(state, player, card, layout)
        if len(values) != SLOT_AWARE_PLAYABILITY_PER_SLOT:
            raise AssertionError(f"slot-aware playability feature count mismatch: {len(values)}")
        for value in values:
            out[idx] = value
            idx += 1
    return idx


def encode_information_state(
    state: GameState,
    player: int,
    encoding: EncodingConfig | None = None,
) -> np.ndarray:
    config = state.config
    layout = _layout_for(config, encoding)
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
    idx += n_colors + 1

    if layout["derived_playability"]:
        idx = _append_derived_playability_features(out, idx, state, player, layout)
    if layout["slot_aware_playability"]:
        idx = _append_slot_aware_playability_features(out, idx, state, player, layout)

    if idx != len(out):
        raise AssertionError(f"encoding dim mismatch: {idx} vs {len(out)}")
    return out


def legal_mask_array(state: GameState) -> np.ndarray:
    return state.unified_legal_mask_np()


def infer_input_dim(config: LostCitiesConfig, encoding: EncodingConfig | None = None) -> int:
    return int(_layout_for(config, encoding)["total"])
