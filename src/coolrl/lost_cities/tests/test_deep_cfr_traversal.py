from __future__ import annotations

import math

import numpy as np
import torch

from coolrl.lost_cities.deep_cfr.encoding import infer_input_dim
from coolrl.lost_cities.deep_cfr.memory import AdvantageMemory, StrategyMemory
from coolrl.lost_cities.deep_cfr.networks import AdvantageNet, regret_matching
from coolrl.lost_cities.deep_cfr.config import NetworkConfig
from coolrl.lost_cities.deep_cfr.traversal import cfr_traverse
from coolrl.lost_cities.game import GameState, tier_config


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
