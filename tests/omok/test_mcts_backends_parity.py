from __future__ import annotations

import shutil
import subprocess
import warnings
from dataclasses import dataclass
from functools import cache
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
    exactly_five: bool = False
    c_puct: float = 1.25
    virtual_loss: float = 1.0


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


EDGE_PARITY_CASES = (
    # White has four-in-a-row on row 2; black to play must block at action 22.
    # Preferred_action=40 deliberately steers priors to the wrong cell so parity
    # is only preserved if both backends propagate terminal-loss values identically
    # through the search tree.
    ParityCase(
        name="must_block_open_four_threat",
        moves=(0, 18, 80, 19, 79, 20, 78, 21),
        preferred_action=40,
        simulations=96,
    ),
    # Evaluator's preferred_action is already occupied on the board; the masked
    # priors fallback must match across backends.
    ParityCase(
        name="preferred_action_is_occupied",
        moves=(40, 0, 41, 1, 42, 2),
        preferred_action=40,
        simulations=32,
    ),
    # Evaluator prefers a literally out-of-range action (-1) so all priors come
    # from the uniform fallback branch; exercises the "priors sum to zero" path.
    ParityCase(
        name="uniform_priors_fallback",
        moves=(40,),
        preferred_action=-1,
        simulations=32,
    ),
    # Only white has played; to_play == -1 at the root. Ensures sign conventions
    # of features (own/opp planes, color channel) and backup match between
    # backends when the root player is not the default.
    ParityCase(
        name="white_to_play_opening",
        moves=(40,),
        preferred_action=20,
        simulations=24,
    ),
    # Black has a main-diagonal open four; white to move. Exercises the
    # diagonal winning-line branch which the three baseline cases miss
    # (they only set up horizontal threats).
    ParityCase(
        name="main_diagonal_four_white_to_move",
        moves=(0, 80, 10, 79, 20, 78, 30, 77),
        preferred_action=40,
        simulations=96,
    ),
    # Anti-diagonal threat at (0,4)-(3,1); covers the dr=1, dc=-1 branch in
    # the win detector that has no dedicated case above.
    ParityCase(
        name="anti_diagonal_four_white_to_move",
        moves=(4, 80, 12, 79, 20, 78, 28, 77),
        preferred_action=36,
        simulations=96,
    ),
    # Evaluator returns a negative leaf value; the Q-value flips sign on every
    # backup step, so a sign error would diverge quickly.
    ParityCase(
        name="negative_evaluator_value",
        moves=(40,),
        preferred_action=20,
        simulations=48,
        value=-0.75,
    ),
    # Extreme positive leaf value - exercises saturation of the Q/U sum.
    ParityCase(
        name="max_evaluator_value",
        moves=(),
        preferred_action=40,
        simulations=24,
        value=1.0,
    ),
    # Exactly-five rule variant; the flag flows through GameState into both
    # backends and must not change mid-game search behaviour below five stones.
    ParityCase(
        name="exactly_five_rule_midgame",
        moves=(0, 18, 1, 19, 2, 20),
        preferred_action=40,
        simulations=32,
        exactly_five=True,
    ),
    # Very small c_puct: selection is dominated by Q (visit value),
    # so this stresses backup-order equivalence rather than prior weighting.
    ParityCase(
        name="low_c_puct_q_dominated",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=48,
        c_puct=0.25,
    ),
    # Large c_puct: selection is dominated by U (prior * sqrt(N) / (1+n)),
    # which amplifies any virtual-loss or prior rounding difference.
    ParityCase(
        name="high_c_puct_u_dominated",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=48,
        c_puct=6.0,
    ),
    # Heavier virtual loss; with batched leaves the same path should be
    # discouraged more strongly before the evaluation round completes.
    ParityCase(
        name="heavy_virtual_loss",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=64,
        virtual_loss=3.0,
    ),
    # Single simulation: the sim loop executes exactly one round; any
    # off-by-one in the `while sims_done < num_simulations` schedule shows up.
    ParityCase(
        name="single_simulation",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=1,
    ),
    # Simulation budget smaller than one batch; `leaves_this_round` gets
    # clamped down and the implementations must behave identically.
    ParityCase(
        name="simulations_less_than_batch",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=3,
    ),
    # Budget not divisible by the batch size; tests the last-round clamp
    # (7 sims with batch=4 -> rounds of 4 then 3).
    ParityCase(
        name="non_divisible_simulation_schedule",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=7,
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


def _state_from_moves(moves: tuple[int, ...], *, exactly_five: bool = False) -> GameState:
    state = GameState(exactly_five=exactly_five)
    for action in moves:
        state.apply_action(action)
    return state


@cache
def _ensure_native_rust_backend_built() -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is not available")

    completed = subprocess.run(
        [
            cargo,
            "build",
            "--locked",
            "--manifest-path",
            str(RUST_MANIFEST),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "native Rust backend build failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _run_backend(
    name: str,
    case: ParityCase,
    *,
    leaves_per_batch: int,
    search_threads: int = 1,
) -> NativeSearchResult:
    if name == "rust":
        _ensure_native_rust_backend_built()

    backend = resolve_mcts_backend(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mcts = backend.MCTS(
            c_puct=case.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            evaluator=DeterministicEvaluator(
                preferred_action=case.preferred_action,
                value=case.value,
            ),
            search_threads=search_threads,
            virtual_loss=case.virtual_loss,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mcts.search_batch(
            [_state_from_moves(case.moves, exactly_five=case.exactly_five)],
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


def _run_backend_two_step(
    name: str,
    case: ParityCase,
    *,
    leaves_per_batch: int,
    search_threads: int = 1,
) -> NativeSearchResult:
    if name == "rust":
        _ensure_native_rust_backend_built()

    backend = resolve_mcts_backend(name)
    mcts = backend.MCTS(
        c_puct=case.c_puct,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        evaluator=DeterministicEvaluator(
            preferred_action=case.preferred_action,
            value=case.value,
        ),
        search_threads=search_threads,
        virtual_loss=case.virtual_loss,
    )
    state = _state_from_moves(case.moves, exactly_five=case.exactly_five)
    first = mcts.search_batch(
        [state],
        num_simulations=case.simulations,
        temperature=[0.0],
        add_noise=False,
        roots=[None],
        leaves_per_batch=leaves_per_batch,
    )[0]
    assert first.next_root is not None
    state.apply_action(first.action)

    second = mcts.search_batch(
        [state],
        num_simulations=case.simulations,
        temperature=[0.0],
        add_noise=False,
        roots=[first.next_root],
        leaves_per_batch=leaves_per_batch,
    )[0]
    assert second.next_root is not None
    return NativeSearchResult(
        action=second.action,
        visit_policy=second.visit_policy,
        root_value=second.root_value,
    )


def _run_backend_many(
    name: str,
    moves_by_state: tuple[tuple[int, ...], ...],
    *,
    preferred_action: int,
    simulations: int,
    leaves_per_batch: int,
    search_threads: int = 1,
) -> list[NativeSearchResult]:
    if name == "rust":
        _ensure_native_rust_backend_built()

    backend = resolve_mcts_backend(name)
    mcts = backend.MCTS(
        c_puct=1.25,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        evaluator=DeterministicEvaluator(preferred_action=preferred_action),
        search_threads=search_threads,
        virtual_loss=1.0,
    )
    results = mcts.search_batch(
        [_state_from_moves(moves) for moves in moves_by_state],
        num_simulations=simulations,
        temperature=[0.0] * len(moves_by_state),
        add_noise=False,
        roots=[None] * len(moves_by_state),
        leaves_per_batch=leaves_per_batch,
    )
    return [
        NativeSearchResult(
            action=result.action,
            visit_policy=result.visit_policy,
            root_value=result.root_value,
        )
        for result in results
    ]


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


def test_configured_rust_backend_is_native_wrapper() -> None:
    from coolrl.omok import mcts as python_mcts

    _ensure_native_rust_backend_built()
    backend = resolve_mcts_backend("rust")

    assert not issubclass(backend.MCTS, python_mcts.MCTS)


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("leaves_per_batch", (1, 4), ids=("sequential", "batched"))
def test_native_rust_mcts_matches_python(case: ParityCase, leaves_per_batch: int) -> None:
    py = _run_backend("python", case, leaves_per_batch=leaves_per_batch)
    rust = _run_backend("rust", case, leaves_per_batch=leaves_per_batch)

    _assert_search_results_match(py, rust, policy_atol=1.0e-6, value_atol=1.0e-6)


@pytest.mark.parametrize("leaves_per_batch", (1, 4), ids=("sequential", "batched"))
def test_native_rust_reused_root_matches_python(leaves_per_batch: int) -> None:
    case = PARITY_CASES[1]
    py = _run_backend_two_step("python", case, leaves_per_batch=leaves_per_batch)
    rust = _run_backend_two_step("rust", case, leaves_per_batch=leaves_per_batch)

    _assert_search_results_match(py, rust, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_native_rust_threaded_collect_matches_python() -> None:
    moves_by_state = (
        (),
        (0, 1, 10, 11),
        (2, 3, 12, 13, 22, 23),
    )
    py_results = _run_backend_many(
        "python",
        moves_by_state,
        preferred_action=40,
        simulations=48,
        leaves_per_batch=4,
    )
    rust_results = _run_backend_many(
        "rust",
        moves_by_state,
        preferred_action=40,
        simulations=48,
        leaves_per_batch=4,
        search_threads=2,
    )

    for py, rust in zip(py_results, rust_results, strict=True):
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


# ----------------------------------------------------------------------------
# Edge-case parity suite
#
# The baseline PARITY_CASES cover only three "happy path" positions with the
# default c_puct / virtual_loss / evaluator configuration, horizontal win
# detection, and simulation budgets that divide cleanly by the batch size.
# The cases and tests below fill the gaps most likely to hide a backend
# divergence: forced blocks, diagonal win detection, occupied/illegal
# preferred actions, non-default player-to-move, negative/extreme evaluator
# values, the exactly-five rule flag, extreme c_puct and virtual_loss, odd
# simulation schedules, multi-step tree reuse, mixed terminal batches, and
# higher thread counts.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("case", EDGE_PARITY_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("leaves_per_batch", (1, 4, 8), ids=("sequential", "batched", "wide_batch"))
def test_native_rust_edge_case_matches_python(case: ParityCase, leaves_per_batch: int) -> None:
    py = _run_backend("python", case, leaves_per_batch=leaves_per_batch)
    rust = _run_backend("rust", case, leaves_per_batch=leaves_per_batch)

    _assert_search_results_match(py, rust, policy_atol=1.0e-6, value_atol=1.0e-6)


@pytest.mark.parametrize("case", EDGE_PARITY_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("leaves_per_batch", (1, 4, 8), ids=("sequential", "batched", "wide_batch"))
def test_c_backend_edge_case_matches_python(case: ParityCase, leaves_per_batch: int) -> None:
    py = _run_backend("python", case, leaves_per_batch=leaves_per_batch)

    try:
        c = _run_backend("c", case, leaves_per_batch=leaves_per_batch)
    except RuntimeError as exc:
        pytest.skip(f"C backend not available in this environment: {exc}")

    _assert_search_results_match(py, c, policy_atol=1.0e-6, value_atol=1.0e-6)


@pytest.mark.parametrize("case", EDGE_PARITY_CASES, ids=lambda case: case.name)
def test_native_rust_edge_case_reused_root_matches_python(case: ParityCase) -> None:
    # Only one-shot cases become no-ops after tree advance; skip any case whose
    # starting state has no legal follow-up or whose simulation budget is too
    # small to produce a meaningful child subtree.
    if case.simulations < 2:
        pytest.skip("two-step reuse requires at least two simulations")
    py = _run_backend_two_step("python", case, leaves_per_batch=4)
    rust = _run_backend_two_step("rust", case, leaves_per_batch=4)

    _assert_search_results_match(py, rust, policy_atol=1.0e-6, value_atol=1.0e-6)


def _run_backend_multi_step(
    name: str,
    case: ParityCase,
    *,
    steps: int,
    leaves_per_batch: int,
    search_threads: int = 1,
) -> list[NativeSearchResult]:
    """Run `steps` consecutive searches, advancing the root between each."""

    if name == "rust":
        _ensure_native_rust_backend_built()

    backend = resolve_mcts_backend(name)
    mcts = backend.MCTS(
        c_puct=case.c_puct,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        evaluator=DeterministicEvaluator(
            preferred_action=case.preferred_action,
            value=case.value,
        ),
        search_threads=search_threads,
        virtual_loss=case.virtual_loss,
    )
    state = _state_from_moves(case.moves, exactly_five=case.exactly_five)
    root: object | None = None
    collected: list[NativeSearchResult] = []
    for _ in range(steps):
        result = mcts.search_batch(
            [state],
            num_simulations=case.simulations,
            temperature=[0.0],
            add_noise=False,
            roots=[root],
            leaves_per_batch=leaves_per_batch,
        )[0]
        collected.append(
            NativeSearchResult(
                action=result.action,
                visit_policy=result.visit_policy,
                root_value=result.root_value,
            )
        )
        if state.terminal:
            break
        state.apply_action(result.action)
        root = result.next_root
    return collected


def test_native_rust_deep_tree_reuse_matches_python() -> None:
    """Advance the root over four consecutive searches and keep parity."""

    case = ParityCase(
        name="deep_reuse",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=32,
    )
    py = _run_backend_multi_step("python", case, steps=4, leaves_per_batch=4)
    rust = _run_backend_multi_step("rust", case, steps=4, leaves_per_batch=4)

    assert len(py) == len(rust)
    for step_py, step_rust in zip(py, rust, strict=True):
        _assert_search_results_match(step_py, step_rust, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_c_backend_deep_tree_reuse_matches_python() -> None:
    case = ParityCase(
        name="deep_reuse",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=32,
    )
    py = _run_backend_multi_step("python", case, steps=4, leaves_per_batch=4)

    try:
        c = _run_backend_multi_step("c", case, steps=4, leaves_per_batch=4)
    except RuntimeError as exc:
        pytest.skip(f"C backend not available in this environment: {exc}")

    for step_py, step_c in zip(py, c, strict=True):
        _assert_search_results_match(step_py, step_c, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_native_rust_large_batch_matches_python() -> None:
    """Many concurrent trees in one batch - exercises per-tree bookkeeping."""

    moves_by_state = (
        (),
        (40,),
        (0, 1),
        (0, 1, 10, 11),
        (2, 3, 12, 13, 22, 23),
        (0, 9, 1, 10, 2, 11, 3, 12),
        (40, 0, 41, 1, 42, 2),
        (4, 80, 12, 79, 20, 78, 28, 77),
    )
    py = _run_backend_many(
        "python",
        moves_by_state,
        preferred_action=40,
        simulations=32,
        leaves_per_batch=4,
    )
    rust = _run_backend_many(
        "rust",
        moves_by_state,
        preferred_action=40,
        simulations=32,
        leaves_per_batch=4,
    )

    for py_result, rust_result in zip(py, rust, strict=True):
        _assert_search_results_match(py_result, rust_result, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_native_rust_higher_threads_matches_python() -> None:
    """Same batch as the 2-thread test, but with search_threads=4."""

    moves_by_state = (
        (),
        (0, 1, 10, 11),
        (2, 3, 12, 13, 22, 23),
        (40, 0, 41, 1, 42, 2),
    )
    py = _run_backend_many(
        "python",
        moves_by_state,
        preferred_action=40,
        simulations=48,
        leaves_per_batch=4,
    )
    rust = _run_backend_many(
        "rust",
        moves_by_state,
        preferred_action=40,
        simulations=48,
        leaves_per_batch=4,
        search_threads=4,
    )

    for py_result, rust_result in zip(py, rust, strict=True):
        _assert_search_results_match(py_result, rust_result, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_native_rust_terminal_in_batch_matches_python() -> None:
    """A batch where some entries are already terminal wins/draws.

    The baseline threaded-collect test always passes non-terminal states;
    this one forces the terminal short-circuit path in both backends.
    """

    # A terminal position where black (player 1) has won via row 0.
    terminal_win = (0, 9, 1, 10, 2, 11, 3, 12, 4)  # black plays action 4 to win
    moves_by_state = (
        terminal_win,
        (),
        (0, 1, 10, 11),
    )
    py = _run_backend_many(
        "python",
        moves_by_state,
        preferred_action=40,
        simulations=24,
        leaves_per_batch=2,
    )
    rust = _run_backend_many(
        "rust",
        moves_by_state,
        preferred_action=40,
        simulations=24,
        leaves_per_batch=2,
    )

    for py_result, rust_result in zip(py, rust, strict=True):
        _assert_search_results_match(py_result, rust_result, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_c_backend_terminal_in_batch_matches_python() -> None:
    terminal_win = (0, 9, 1, 10, 2, 11, 3, 12, 4)
    moves_by_state = (
        terminal_win,
        (),
        (0, 1, 10, 11),
    )
    py = _run_backend_many(
        "python",
        moves_by_state,
        preferred_action=40,
        simulations=24,
        leaves_per_batch=2,
    )

    try:
        c = _run_backend_many(
            "c",
            moves_by_state,
            preferred_action=40,
            simulations=24,
            leaves_per_batch=2,
        )
    except RuntimeError as exc:
        pytest.skip(f"C backend not available in this environment: {exc}")

    for py_result, c_result in zip(py, c, strict=True):
        _assert_search_results_match(py_result, c_result, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_native_rust_singleton_batch_with_threads_matches_python() -> None:
    """search_threads>1 with a batch of size 1 - the threaded collect path
    must not depend on having multiple trees to divide work between."""

    moves_by_state = ((0, 1, 10, 11),)
    py = _run_backend_many(
        "python",
        moves_by_state,
        preferred_action=40,
        simulations=64,
        leaves_per_batch=4,
    )
    rust = _run_backend_many(
        "rust",
        moves_by_state,
        preferred_action=40,
        simulations=64,
        leaves_per_batch=4,
        search_threads=4,
    )

    for py_result, rust_result in zip(py, rust, strict=True):
        _assert_search_results_match(py_result, rust_result, policy_atol=1.0e-6, value_atol=1.0e-6)


@pytest.mark.parametrize("leaves_per_batch", (1, 2, 4, 16), ids=lambda n: f"batch_{n}")
def test_native_rust_batch_size_sweep_matches_python(leaves_per_batch: int) -> None:
    """Sweep `leaves_per_batch` across a wider range than the baseline tests.

    Virtual loss, priors, and backup are all applied per-round; a bug that
    only surfaces with a very wide or very narrow batch would be missed by
    the existing (1, 4) parametrisation.
    """

    case = ParityCase(
        name="batch_sweep",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=64,
    )
    py = _run_backend("python", case, leaves_per_batch=leaves_per_batch)
    rust = _run_backend("rust", case, leaves_per_batch=leaves_per_batch)

    _assert_search_results_match(py, rust, policy_atol=1.0e-6, value_atol=1.0e-6)


def test_native_rust_zero_value_evaluator_matches_python() -> None:
    """All leaf values are exactly zero; only priors and tree structure
    differentiate children. Isolates the prior/selection path from value
    backup so that any divergence points at PUCT selection or virtual loss."""

    case = ParityCase(
        name="zero_value",
        moves=(0, 1, 10, 11),
        preferred_action=40,
        simulations=48,
        value=0.0,
    )
    py = _run_backend("python", case, leaves_per_batch=4)
    rust = _run_backend("rust", case, leaves_per_batch=4)

    _assert_search_results_match(py, rust, policy_atol=1.0e-6, value_atol=1.0e-6)
