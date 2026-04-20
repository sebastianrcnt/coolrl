from __future__ import annotations

import ctypes
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .board import GameState
from .evaluator import Evaluator
from .mcts_types import SearchResult


ACTION_SIZE = 81
FEATURE_STRIDE = 4 * ACTION_SIZE
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT8_P = ctypes.POINTER(ctypes.c_int8)
_SIZE_P = ctypes.POINTER(ctypes.c_size_t)
_EVAL_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
    ctypes.c_void_p,
)


def _library_names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libomok_rmcts.dylib",)
    if sys.platform == "win32":
        return ("omok_rmcts.dll",)
    return ("libomok_rmcts.so",)


def _load_library() -> ctypes.CDLL:
    package_dir = Path(__file__).resolve().parent
    crate_dir = package_dir / "rmcts"
    candidates: list[Path] = []
    for profile in ("release", "debug"):
        for name in _library_names():
            candidates.append(crate_dir / "target" / profile / name)

    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    raise RuntimeError(
        "Rust MCTS backend is not built. Run "
        "`cargo build --locked --manifest-path src/coolrl/omok/rmcts/Cargo.toml` "
        "or use `selfplay.mcts_backend: python`."
    )


_LIB = _load_library()
_LIB.omok_rmcts_search.argtypes = [
    _INT8_P,
    ctypes.c_int8,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_int8,
    ctypes.c_uint8,
    ctypes.c_uint8,
    ctypes.c_float,
    ctypes.c_size_t,
    ctypes.c_float,
    _FLOAT_P,
    ctypes.c_float,
    _EVAL_CALLBACK,
    ctypes.c_void_p,
    _SIZE_P,
    _FLOAT_P,
    _FLOAT_P,
]
_LIB.omok_rmcts_search.restype = ctypes.c_int


class MCTS:
    """Native Rust MCTS backend using a ctypes callback for neural evaluation."""

    _warned_no_root_reuse = False
    _warned_no_leaf_batch = False

    def __init__(
        self,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        evaluator: Evaluator,
        search_threads: int = 1,
        virtual_loss: float = 1.0,
    ) -> None:
        if not hasattr(evaluator, "evaluate_features"):
            raise TypeError("Rust MCTS backend requires an evaluator with evaluate_features()")
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.evaluator = evaluator
        self.search_threads = max(1, int(search_threads))
        self.virtual_loss = float(virtual_loss)
        self._callback_error: BaseException | None = None
        self._callback = _EVAL_CALLBACK(self._evaluate_callback)

    def search_batch(
        self,
        states: list[GameState],
        num_simulations: int,
        temperature: list[float],
        add_noise: bool,
        roots: list[Any | None] | None = None,
        leaves_per_batch: int = 1,
    ) -> list[SearchResult]:
        if len(temperature) != len(states):
            raise ValueError("temperature and states must have the same length")
        if roots is None:
            roots = [None] * len(states)
        if len(roots) != len(states):
            raise ValueError("roots and states must have the same length")
        if any(root is not None for root in roots) and not MCTS._warned_no_root_reuse:
            warnings.warn(
                "Rust MCTS backend does not reuse search roots yet; running fresh searches.",
                RuntimeWarning,
                stacklevel=2,
            )
            MCTS._warned_no_root_reuse = True
        if leaves_per_batch != 1 and not MCTS._warned_no_leaf_batch:
            warnings.warn(
                "Rust MCTS backend currently runs sequential native searches; leaves_per_batch is ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
            MCTS._warned_no_leaf_batch = True

        results: list[SearchResult] = []
        for state, temp in zip(states, temperature, strict=True):
            results.append(
                self._search_one(
                    state,
                    num_simulations=max(0, int(num_simulations)),
                    temperature=float(temp),
                    add_noise=add_noise,
                )
            )
        return results

    def _search_one(
        self,
        state: GameState,
        *,
        num_simulations: int,
        temperature: float,
        add_noise: bool,
    ) -> SearchResult:
        board = np.ascontiguousarray(state.board.reshape(-1), dtype=np.int8)
        policy = np.empty(ACTION_SIZE, dtype=np.float32)
        action = ctypes.c_size_t(0)
        root_value = ctypes.c_float(0.0)
        root_noise = self._root_noise(state) if add_noise else None
        root_noise_ptr = (
            root_noise.ctypes.data_as(_FLOAT_P)
            if root_noise is not None
            else ctypes.cast(None, _FLOAT_P)
        )
        root_noise_epsilon = self.dirichlet_epsilon if root_noise is not None else 0.0

        self._callback_error = None
        status = _LIB.omok_rmcts_search(
            board.ctypes.data_as(_INT8_P),
            ctypes.c_int8(int(state.to_play)),
            -1 if state.last_action is None else int(state.last_action),
            ctypes.c_size_t(int(state.move_count)),
            ctypes.c_int8(int(state.winner)),
            ctypes.c_uint8(1 if state.terminal else 0),
            ctypes.c_uint8(1 if state.exactly_five else 0),
            ctypes.c_float(self.c_puct),
            ctypes.c_size_t(num_simulations),
            ctypes.c_float(temperature),
            root_noise_ptr,
            ctypes.c_float(root_noise_epsilon),
            self._callback,
            None,
            ctypes.byref(action),
            policy.ctypes.data_as(_FLOAT_P),
            ctypes.byref(root_value),
        )
        if self._callback_error is not None:
            raise RuntimeError("Rust MCTS evaluator callback failed") from self._callback_error
        if status != 0:
            raise RuntimeError(f"Rust MCTS search failed with status {status}")

        return SearchResult(
            action=int(action.value),
            visit_policy=policy,
            root_value=float(root_value.value),
            next_root=None,
        )

    def _root_noise(self, state: GameState) -> np.ndarray | None:
        if self.dirichlet_alpha <= 0.0 or self.dirichlet_epsilon <= 0.0 or state.terminal:
            return None
        legal = state.legal_moves()
        count = int(legal.sum())
        if count <= 0:
            return None
        noise = np.zeros(ACTION_SIZE, dtype=np.float32)
        noise[legal] = np.random.dirichlet([self.dirichlet_alpha] * count).astype(np.float32)
        return noise

    def _evaluate_callback(
        self,
        features_ptr: _FLOAT_P,
        priors_out: _FLOAT_P,
        value_out: _FLOAT_P,
        _user_data: ctypes.c_void_p,
    ) -> int:
        try:
            features = np.ctypeslib.as_array(features_ptr, shape=(FEATURE_STRIDE,))
            batch = features.reshape(1, 4, 9, 9)
            priors, values = self.evaluator.evaluate_features(batch)
            priors_arr = np.asarray(priors[0], dtype=np.float32)
            if priors_arr.shape != (ACTION_SIZE,):
                raise ValueError(f"unexpected prior shape from evaluator: {priors_arr.shape}")
            out_priors = np.ctypeslib.as_array(priors_out, shape=(ACTION_SIZE,))
            out_priors[:] = priors_arr
            value_out[0] = float(np.asarray(values, dtype=np.float32).reshape(-1)[0])
            return 0
        except BaseException as exc:  # ctypes callbacks must not raise through C.
            self._callback_error = exc
            return 1
