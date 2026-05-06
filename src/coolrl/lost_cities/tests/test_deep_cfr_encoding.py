from __future__ import annotations

import numpy as np

from coolrl.lost_cities.deep_cfr.encoding import (
    DERIVED_PLAYABILITY_COMMON,
    DERIVED_PLAYABILITY_PER_COLOR,
    encode_information_state,
    infer_input_dim,
    legal_mask_array,
)
from coolrl.lost_cities.deep_cfr.config import EncodingConfig
from coolrl.lost_cities.game import Card, GameState, tier_config


def test_encoder_shape_stable_for_tier3() -> None:
    config = tier_config("tier3", seed=7)
    state = GameState.new_game(config)
    encoded = encode_information_state(state, 0)
    assert encoded.dtype == np.float32
    assert encoded.shape == (infer_input_dim(config),)


def test_derived_playability_encoder_extends_shape_only_when_enabled() -> None:
    config = tier_config("tier3", seed=7)
    state = GameState.new_game(config)
    base_dim = infer_input_dim(config)
    derived_encoding = EncodingConfig(derived_playability=True)
    derived_dim = infer_input_dim(config, derived_encoding)
    assert derived_dim == (
        base_dim + config.n_colors * DERIVED_PLAYABILITY_PER_COLOR + DERIVED_PLAYABILITY_COMMON
    )
    assert encode_information_state(state, 0).shape == (base_dim,)
    assert encode_information_state(state, 0, derived_encoding).shape == (derived_dim,)


def test_encoder_does_not_leak_opponent_hand() -> None:
    config = tier_config("tier1", seed=1)
    state_a = GameState.new_game(config)
    state_b = state_a.clone()
    state_b.hands[1] = list(reversed(state_b.hands[1]))
    assert np.array_equal(
        encode_information_state(state_a, 0),
        encode_information_state(state_b, 0),
    )


def test_derived_playability_encoder_does_not_leak_opponent_hand() -> None:
    config = tier_config("tier3", seed=1)
    encoding = EncodingConfig(derived_playability=True)
    state_a = GameState.new_game(config)
    state_b = state_a.clone()
    state_b.hands[1] = list(reversed(state_b.hands[1]))
    assert np.array_equal(
        encode_information_state(state_a, 0, encoding),
        encode_information_state(state_b, 0, encoding),
    )


def test_derived_playability_color_features_match_known_state() -> None:
    config = tier_config("tier3", seed=3)
    encoding = EncodingConfig(derived_playability=True)
    state = GameState.empty(config)
    state.deck = [Card(0, 8), Card(0, 9), Card(1, 8)]
    state.hands = [
        [Card(0, 1), Card(0, 5), Card(0, 6), Card(0, 0)],
        [Card(2, 3), Card(3, 4)],
    ]
    state.expeditions[0][0] = [Card(0, 0), Card(0, 4)]
    state.expeditions[1][0] = [Card(0, 0)]
    state.discards[0] = [Card(0, 7)]
    state.current_player = 0
    state.phase = "card"

    encoded = encode_information_state(state, 0, encoding)
    base_dim = infer_input_dim(config)
    color0 = encoded[base_dim : base_dim + DERIVED_PLAYABILITY_PER_COLOR]

    max_numeric_sum = 54.0
    max_cards_per_color = 12.0
    max_wagers = 3.0
    max_score_estimate = 136.0
    expected = np.asarray(
        [
            0.0,  # is_unopened
            0.0,  # has_only_wagers_opened
            5.0 / max_numeric_sum,
            1.0 / max_wagers,
            2.0 / max_cards_per_color,
            4.0 / max_numeric_sum,
            4.0 / max_cards_per_color,
            1.0 / max_wagers,
            13.0 / max_numeric_sum,
            2.0 / max_cards_per_color,
            1.0 / max_cards_per_color,
            2.0 / max_numeric_sum,
            -2.0 / max_numeric_sum,
            -4.0 / max_score_estimate,
            2.0 / max_numeric_sum,
            1.0,
            8.0 / max_numeric_sum,
            4.0 / max_cards_per_color,
            4.0 / max_cards_per_color,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(color0, expected, rtol=1.0e-6, atol=1.0e-6)


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
