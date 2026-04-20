from __future__ import annotations

import numpy as np
import pytest
import torch

from coolrl.omok.board import GameState
from coolrl.omok.config import NetworkConfig, config_from_dict
from coolrl.omok.features import states_to_feature_planes
from coolrl.omok.mcts import MCTS
from coolrl.omok.torch_network import PolicyValueNet


class UniformEvaluator:
    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        action_size = states[0].action_size
        priors = np.zeros((len(states), action_size), dtype=np.float32)
        values = np.zeros((len(states),), dtype=np.float32)
        for index, state in enumerate(states):
            legal = state.legal_moves()
            priors[index, legal] = 1.0
            priors[index] /= np.clip(priors[index].sum(), 1.0e-8, None)
        return priors, values


@pytest.mark.parametrize("board_size", (9, 13, 15))
def test_game_state_features_and_network_shapes(board_size: int) -> None:
    state = GameState(board_size=board_size)
    state.apply_action((board_size // 2) * board_size + board_size // 2)

    assert state.legal_moves().shape == (board_size * board_size,)
    assert state.feature_planes().shape == (4, board_size, board_size)
    assert states_to_feature_planes([state]).shape == (1, 4, board_size, board_size)

    model = PolicyValueNet(board_size, NetworkConfig(channels=8, blocks=1, value_hidden=16))
    model.eval()
    with torch.inference_mode():
        policy, value = model(torch.zeros(2, 4, board_size, board_size))

    assert policy.shape == (2, board_size * board_size)
    assert value.shape == (2,)


@pytest.mark.parametrize("board_size", (9, 13, 15))
def test_python_mcts_policy_shape_tracks_board_size(board_size: int) -> None:
    search = MCTS(
        c_puct=1.25,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        evaluator=UniformEvaluator(),
    )
    result = search.search_batch(
        [GameState(board_size=board_size)],
        num_simulations=2,
        temperature=[0.0],
        add_noise=False,
    )[0]

    assert 0 <= result.action < board_size * board_size
    assert result.visit_policy.shape == (board_size * board_size,)


def test_config_accepts_non_nine_board_size() -> None:
    config = config_from_dict({"rules": {"board_size": 15}})

    assert config.rules.board_size == 15
