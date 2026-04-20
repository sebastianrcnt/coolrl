from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from coolrl.omok.board import GameState
from coolrl.omok.mcts_backend import resolve_mcts_backend


ROOT = Path(__file__).resolve().parents[2]
RUST_MANIFEST = ROOT / "src/coolrl/omok/rmcts/Cargo.toml"


@dataclass(frozen=True)
class ParityCase:
    name: str
    moves: tuple[int, ...]
    preferred_action: int
    simulations: int
    value: float = 0.25


@dataclass(frozen=True)
class NativeSearchResult:
    action: int
    visit_policy: np.ndarray
    root_value: float


PARITY_CASES = (
    ParityCase(
        name="empty_board_center_preferred",
        moves=(),
        preferred_action=40,
        simulations=24,
    ),
    ParityCase(
        name="asymmetric_midgame_center_preferred",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=48,
    ),
    ParityCase(
        name="immediate_win_preferred",
        moves=(0, 9, 1, 10, 2, 11, 3, 12),
        preferred_action=4,
        simulations=32,
    ),
)


class DeterministicEvaluator:
    """Evaluator usable by Python and C MCTS backends."""

    def __init__(self, preferred_action: int, value: float = 0.25) -> None:
        self.preferred_action = int(preferred_action)
        self.value = float(value)

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        priors = np.zeros((len(states), 81), dtype=np.float32)
        values = np.full((len(states),), self.value, dtype=np.float32)
        for idx, state in enumerate(states):
            legal = state.legal_moves()
            priors[idx, legal] = 1.0e-3
            if 0 <= self.preferred_action < legal.size and legal[self.preferred_action]:
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
            if 0 <= self.preferred_action < legal.shape[1] and legal[idx, self.preferred_action]:
                priors[idx, self.preferred_action] = 1.0

        priors /= np.clip(priors.sum(axis=1, keepdims=True), 1.0e-8, None)
        values = np.full((features.shape[0],), self.value, dtype=np.float32)
        return priors, values


def _state_from_moves(moves: tuple[int, ...]) -> GameState:
    state = GameState()
    for action in moves:
        state.apply_action(action)
    return state


def _run_backend(name: str, case: ParityCase, *, leaves_per_batch: int) -> NativeSearchResult:
    backend = resolve_mcts_backend(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mcts = backend.MCTS(
            c_puct=1.25,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=DeterministicEvaluator(
                preferred_action=case.preferred_action,
                value=case.value,
            ),
            search_threads=1,
            virtual_loss=1.0,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mcts.search_batch(
            [_state_from_moves(case.moves)],
            num_simulations=case.simulations,
            temperature=[0.0],
            add_noise=False,
            leaves_per_batch=leaves_per_batch,
        )[0]
    return NativeSearchResult(
        action=result.action,
        visit_policy=result.visit_policy,
        root_value=result.root_value,
    )


def _run_native_rust(case: ParityCase) -> NativeSearchResult:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is not available")

    command = [
        cargo,
        "run",
        "--quiet",
        "--locked",
        "--manifest-path",
        str(RUST_MANIFEST),
        "--bin",
        "parity_probe",
        "--",
        "--moves",
        ",".join(str(move) for move in case.moves),
        "--preferred-action",
        str(case.preferred_action),
        "--simulations",
        str(case.simulations),
        "--c-puct",
        "1.25",
        "--temperature",
        "0.0",
        "--value",
        str(case.value),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "native Rust parity probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = json.loads(completed.stdout)
    return NativeSearchResult(
        action=int(payload["action"]),
        visit_policy=np.asarray(payload["visit_policy"], dtype=np.float32),
        root_value=float(payload["root_value"]),
    )


def _assert_search_results_match(
    expected: NativeSearchResult,
    actual: NativeSearchResult,
    *,
    policy_atol: float,
    value_atol: float,
) -> None:
    assert actual.action == expected.action
    np.testing.assert_allclose(
        actual.visit_policy,
        expected.visit_policy,
        atol=policy_atol,
        rtol=policy_atol,
    )
    assert actual.root_value == pytest.approx(expected.root_value, abs=value_atol)


def test_configured_rust_backend_is_currently_python_shim() -> None:
    from coolrl.omok import mcts as python_mcts

    backend = resolve_mcts_backend("rust")

    assert issubclass(backend.MCTS, python_mcts.MCTS)


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.name)
def test_native_rust_mcts_matches_python_for_sequential_search(case: ParityCase) -> None:
    py = _run_backend("python", case, leaves_per_batch=1)
    rust = _run_native_rust(case)

    _assert_search_results_match(py, rust, policy_atol=1.0e-6, value_atol=1.0e-6)


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("leaves_per_batch", (1, 4), ids=("sequential", "batched"))
def test_c_backend_matches_python_when_available(case: ParityCase, leaves_per_batch: int) -> None:
    py = _run_backend("python", case, leaves_per_batch=leaves_per_batch)

    try:
        c = _run_backend("c", case, leaves_per_batch=leaves_per_batch)
    except RuntimeError as exc:
        pytest.skip(f"C backend not available in this environment: {exc}")

    _assert_search_results_match(py, c, policy_atol=1.0e-6, value_atol=1.0e-6)
