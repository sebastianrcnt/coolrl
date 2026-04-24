from __future__ import annotations

import numpy as np

from coolrl.lost_cities.deep_cfr.encoding import (
    encode_information_state,
    infer_input_dim,
    legal_mask_array,
)
from coolrl.lost_cities.game import Card, GameState, tier_config


def test_encoder_shape_stable_for_tier3() -> None:
    config = tier_config("tier3", seed=7)
    state = GameState.new_game(config)
    encoded = encode_information_state(state, 0)
    assert encoded.dtype == np.float32
    assert encoded.shape == (infer_input_dim(config),)


def test_encoder_does_not_leak_opponent_hand() -> None:
    config = tier_config("tier1", seed=1)
    state_a = GameState.new_game(config)
    state_b = state_a.clone()
    state_b.hands[1] = list(reversed(state_b.hands[1]))
    assert np.array_equal(
        encode_information_state(state_a, 0),
        encode_information_state(state_b, 0),
    )


def test_encoder_perspective_swaps_public_features() -> None:
    config = tier_config("tier1", seed=2)
    state = GameState.empty(config)
    state.deck = []
    state.hands = [[Card(0, 1)], [Card(1, 1)]]
    state.expeditions[0][0].append(Card(0, 2))
    state.expeditions[1][1].append(Card(1, 2))
    state.discards[2].append(Card(2, 1))
    state.current_player = 0
    a = encode_information_state(state, 0)
    b = encode_information_state(state, 1)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_legal_mask_shape_matches_action_size() -> None:
    config = tier_config("tier3", seed=7)
    state = GameState.new_game(config)
    assert legal_mask_array(state).shape == (config.action_size,)
