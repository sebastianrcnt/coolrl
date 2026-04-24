from __future__ import annotations

import math

import numpy as np
import torch

from coolrl.lost_cities.deep_cfr.encoding import infer_input_dim
from coolrl.lost_cities.deep_cfr.memory import AdvantageMemory, StrategyMemory
from coolrl.lost_cities.deep_cfr.networks import AdvantageNet, regret_matching
from coolrl.lost_cities.deep_cfr.config import NetworkConfig
from coolrl.lost_cities.deep_cfr.traversal import cfr_traverse
from coolrl.lost_cities.game import Card, GameState, tier_config


def test_regret_matching_respects_legal_mask() -> None:
    probs = regret_matching(
        np.asarray([-1.0, 2.0, 3.0], dtype=np.float32),
        np.asarray([True, False, True]),
    )
    assert np.isclose(probs.sum(), 1.0)
    assert probs[1] == 0.0
    assert probs[2] == 1.0


def test_cfr_traversal_tier3_completes() -> None:
    config = tier_config("tier3", seed=3)
    input_dim = infer_input_dim(config)
    nets = [
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
    ]
    memories = [AdvantageMemory(100), AdvantageMemory(100)]
    value, stats = cfr_traverse(
        GameState.new_game(config),
        0,
        1,
        nets,
        memories,
        StrategyMemory(100),
        device=torch.device("cpu"),
        max_depth=4,
        rng=np.random.default_rng(5),
    )
    assert math.isfinite(value)
    assert stats.nodes > 0


def test_cfr_traversal_max_depth_cutoff_returns_current_score_diff() -> None:
    config = tier_config("tier1", seed=13)
    input_dim = infer_input_dim(config)
    nets = [
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
    ]
    memories = [AdvantageMemory(100), AdvantageMemory(100)]
    state = GameState.empty(config)
    state.hands = [[Card(0, 1)], [Card(1, 1)]]
    state.expeditions[0][0].append(Card(0, 2))
    state.expeditions[0][0].append(Card(0, 3))
    state.current_player = 0
    state.phase = "card"
    expected_value = float(state.score_diff(0))

    value, stats = cfr_traverse(
        state,
        0,
        1,
        nets,
        memories,
        StrategyMemory(100),
        device=torch.device("cpu"),
        max_depth=0,
        rng=np.random.default_rng(19),
    )

    assert math.isfinite(value)
    assert value == expected_value
    assert stats.cutoffs == 1
    assert stats.nodes == 1


def test_cfr_traversal_node_limit_cutoff_returns_current_score_diff() -> None:
    config = tier_config("tier1", seed=21)
    input_dim = infer_input_dim(config)
    nets = [
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
    ]
    memories = [AdvantageMemory(100), AdvantageMemory(100)]
    state = GameState.new_game(config)
    expected_value = float(state.score_diff(0))

    value, stats = cfr_traverse(
        state,
        0,
        1,
        nets,
        memories,
        StrategyMemory(100),
        device=torch.device("cpu"),
        max_depth=None,
        max_nodes_per_traversal=1,
        rng=np.random.default_rng(23),
    )

    assert math.isfinite(value)
    assert value == expected_value
    assert stats.node_limit_cutoffs == 1
    assert stats.terminals == 0
    assert stats.nodes == 1


def test_cfr_traversal_stores_regrets_for_multiple_traverser_decisions() -> None:
    config = tier_config("tier1", seed=11)
    input_dim = infer_input_dim(config)
    nets = [
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
        AdvantageNet(input_dim, config.action_size, NetworkConfig(hidden_size=16, num_layers=1)),
    ]
    memories = [AdvantageMemory(100), AdvantageMemory(100)]
    state = GameState.new_game(config)
    value, stats = cfr_traverse(
        state,
        0,
        1,
        nets,
        memories,
        StrategyMemory(100),
        device=torch.device("cpu"),
        max_depth=2,
        max_nodes_per_traversal=100,
        rng=np.random.default_rng(17),
    )

    assert math.isfinite(value)
    assert stats.nodes > 0
    assert stats.cutoffs > 0
    assert len(memories[0]) > 1

    saw_card_phase = False
    saw_draw_phase = False
    for sample in memories[0].samples:
        legal_mask = sample.legal_mask
        has_card_action = bool(np.any(legal_mask[: config.card_action_size]))
        has_draw_action = bool(np.any(legal_mask[config.card_action_size :]))
        if has_card_action and not has_draw_action:
            saw_card_phase = True
        if has_draw_action and not has_card_action:
            saw_draw_phase = True

    assert saw_card_phase
    assert saw_draw_phase
