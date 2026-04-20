from __future__ import annotations

import warnings

import numpy as np
import pytest

from coolrl.omok.board import GameState
from coolrl.omok.mcts_backend import resolve_mcts_backend


class DeterministicEvaluator:
    """Evaluator usable by both Python and C MCTS backends."""

    def __init__(self, preferred_action: int = 40) -> None:
        self.preferred_action = int(preferred_action)

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        priors = np.zeros((len(states), 81), dtype=np.float32)
        values = np.full((len(states),), 0.25, dtype=np.float32)
        for idx, state in enumerate(states):
            legal = state.legal_moves()
            priors[idx, legal] = 1.0e-3
            if legal[self.preferred_action]:
                priors[idx, self.preferred_action] = 1.0
        priors /= np.clip(priors.sum(axis=1, keepdims=True), 1.0e-8, None)
        return priors, values

    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if features.ndim != 4 or features.shape[1:] != (4, 9, 9):
            raise ValueError(f"unexpected feature shape: {features.shape}")

        occupied = (features[:, 0] + features[:, 1]) > 0.5
        legal = (~occupied).reshape(features.shape[0], -1)

        priors = np.zeros((features.shape[0], 81), dtype=np.float32)
        priors[legal] = 1.0e-3
        for idx in range(features.shape[0]):
            if legal[idx, self.preferred_action]:
                priors[idx, self.preferred_action] = 1.0

        priors /= np.clip(priors.sum(axis=1, keepdims=True), 1.0e-8, None)
        values = np.full((features.shape[0],), 0.25, dtype=np.float32)
        return priors, values


def _run_backend(name: str):
    backend = resolve_mcts_backend(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mcts = backend.MCTS(
            c_puct=1.25,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=DeterministicEvaluator(preferred_action=40),
            search_threads=1,
            virtual_loss=1.0,
        )

    state = GameState()
    # 비대칭 상태를 만들어 tie-break 차이를 줄인다.
    for action in (0, 1, 10, 11):
        state.apply_action(action)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mcts.search_batch(
            [state],
            num_simulations=48,
            temperature=[0.0],
            add_noise=False,
            leaves_per_batch=4,
        )[0]
    return result


def test_python_and_rust_backends_match():
    py = _run_backend("python")
    rust = _run_backend("rust")

    assert py.action == rust.action
    np.testing.assert_allclose(py.visit_policy, rust.visit_policy, atol=0.0, rtol=0.0)
    assert py.root_value == pytest.approx(rust.root_value, abs=1.0e-7)


def test_c_backend_matches_python_when_available():
    py = _run_backend("python")

    try:
        c = _run_backend("c")
    except RuntimeError as exc:
        pytest.skip(f"C backend not available in this environment: {exc}")

    assert py.action == c.action
    np.testing.assert_allclose(py.visit_policy, c.visit_policy, atol=1.0e-6, rtol=1.0e-6)
    assert py.root_value == pytest.approx(c.root_value, abs=1.0e-6)
