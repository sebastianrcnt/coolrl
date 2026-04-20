from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

import numpy as np

from .board import GameState
from .evaluator import Evaluator
from .mcts_types import SearchResult


ACTION_SIZE = 81
FEATURE_STRIDE = 4 * ACTION_SIZE


def _load_library() -> ctypes.CDLL:
    package_dir = Path(__file__).resolve().parent
    candidates = sorted(package_dir.glob("_cmcts_c*.so"))
    spec = importlib.util.find_spec("coolrl.omok._cmcts_c")
    if spec and spec.origin:
        candidates.insert(0, Path(spec.origin))
    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate))
    raise RuntimeError(
        "C MCTS backend is not built. Run `uv run python setup.py build_ext --inplace` "
        "or use `selfplay.mcts_backend: python`."
    )


_LIB = _load_library()
_TREE_P = ctypes.c_void_p
_TREE_ARRAY = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C_CONTIGUOUS")
_FLOAT_ARRAY = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_INT8_ARRAY = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
_INT32_ARRAY = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

_LIB.mcts_tree_new.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int]
_LIB.mcts_tree_new.restype = _TREE_P
_LIB.mcts_tree_free.argtypes = [_TREE_P]
_LIB.mcts_tree_free.restype = None
_LIB.mcts_tree_set_initial.argtypes = [
    _TREE_P,
    _INT8_ARRAY,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_LIB.mcts_tree_set_initial.restype = None
_LIB.mcts_tree_advance.argtypes = [_TREE_P, ctypes.c_int]
_LIB.mcts_tree_advance.restype = ctypes.c_int
_LIB.mcts_batch_prepare_roots.argtypes = [_TREE_ARRAY, ctypes.c_int, _FLOAT_ARRAY, ctypes.c_int]
_LIB.mcts_batch_prepare_roots.restype = ctypes.c_int
_LIB.mcts_batch_feed_roots.argtypes = [_TREE_ARRAY, ctypes.c_int, _FLOAT_ARRAY, _FLOAT_ARRAY]
_LIB.mcts_batch_feed_roots.restype = None
_LIB.mcts_batch_apply_root_noise.argtypes = [
    _TREE_ARRAY,
    ctypes.c_int,
    _FLOAT_ARRAY,
    _INT32_ARRAY,
    ctypes.c_float,
]
_LIB.mcts_batch_apply_root_noise.restype = None
_LIB.mcts_batch_root_num_legal.argtypes = [_TREE_ARRAY, ctypes.c_int, _INT32_ARRAY]
_LIB.mcts_batch_root_num_legal.restype = None
_LIB.mcts_batch_get_root_values.argtypes = [_TREE_ARRAY, ctypes.c_int, _FLOAT_ARRAY]
_LIB.mcts_batch_get_root_values.restype = None
_LIB.mcts_batch_collect_leaves.argtypes = [
    _TREE_ARRAY,
    ctypes.c_int,
    ctypes.c_int,
    _FLOAT_ARRAY,
    ctypes.c_int,
]
_LIB.mcts_batch_collect_leaves.restype = ctypes.c_int
_HAS_THREADED_COLLECT = hasattr(_LIB, "mcts_batch_collect_leaves_threaded")
if _HAS_THREADED_COLLECT:
    _LIB.mcts_batch_collect_leaves_threaded.argtypes = [
        _TREE_ARRAY,
        ctypes.c_int,
        ctypes.c_int,
        _FLOAT_ARRAY,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _LIB.mcts_batch_collect_leaves_threaded.restype = ctypes.c_int
_LIB.mcts_batch_feed_leaves.argtypes = [_TREE_ARRAY, ctypes.c_int, _FLOAT_ARRAY, _FLOAT_ARRAY]
_LIB.mcts_batch_feed_leaves.restype = None
_LIB.mcts_batch_extract_visit_counts.argtypes = [_TREE_ARRAY, ctypes.c_int, _FLOAT_ARRAY]
_LIB.mcts_batch_extract_visit_counts.restype = None


class _ChildrenProxy:
    def __init__(self, root: "TreeNode") -> None:
        self.root = root

    def get(self, action: int) -> "TreeNode":
        self.root.advance(action)
        return self.root


class TreeNode:
    def __init__(self, ptr: int, *, owns_ptr: bool = True) -> None:
        if not ptr:
            raise RuntimeError("failed to allocate C MCTS tree")
        self.ptr = int(ptr)
        self._owns_ptr = owns_ptr
        self.children = _ChildrenProxy(self)

    @classmethod
    def from_state(cls, state: GameState, c_puct: float, virtual_loss: float) -> "TreeNode":
        node = cls(
            _LIB.mcts_tree_new(
                ctypes.c_float(c_puct),
                ctypes.c_float(virtual_loss),
                int(state.exactly_five),
            )
        )
        node.reset(state)
        return node

    def reset(self, state: GameState) -> None:
        board = np.ascontiguousarray(state.board.reshape(-1), dtype=np.int8)
        _LIB.mcts_tree_set_initial(
            self.ptr,
            board,
            int(state.to_play),
            -1 if state.last_action is None else int(state.last_action),
            int(state.move_count),
            int(state.winner),
            int(state.terminal),
        )

    def advance(self, action: int) -> None:
        if not _LIB.mcts_tree_advance(self.ptr, int(action)):
            raise ValueError(f"illegal C MCTS action: {action}")

    def close(self) -> None:
        if self._owns_ptr and self.ptr:
            _LIB.mcts_tree_free(self.ptr)
            self.ptr = 0

    def __del__(self) -> None:
        self.close()


class MCTS:
    def __init__(
        self,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        evaluator: Evaluator,
        search_threads: int = 1,
        virtual_loss: float = 1.0,
    ) -> None:
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.evaluator = evaluator
        self.search_threads = max(1, int(search_threads))
        self.virtual_loss = float(virtual_loss)

    def search_batch(
        self,
        states: list[GameState],
        num_simulations: int,
        temperature: list[float],
        add_noise: bool,
        roots: list[TreeNode | None] | None = None,
        leaves_per_batch: int = 1,
    ) -> list[SearchResult]:
        if roots is None:
            roots = [None] * len(states)
        if len(roots) != len(states):
            raise ValueError("roots and states must have the same length")
        if not hasattr(self.evaluator, "evaluate_features"):
            raise TypeError("C MCTS backend requires an evaluator with evaluate_features()")

        active_roots = [
            root if root is not None else TreeNode.from_state(state, self.c_puct, self.virtual_loss)
            for state, root in zip(states, roots, strict=True)
        ]
        tree_ptrs = np.ascontiguousarray([root.ptr for root in active_roots], dtype=np.uintp)

        root_features = np.empty((len(states), 4, 9, 9), dtype=np.float32)
        root_count = _LIB.mcts_batch_prepare_roots(
            tree_ptrs,
            len(active_roots),
            root_features.reshape(-1),
            len(active_roots),
        )
        if root_count:
            priors, values = self.evaluator.evaluate_features(root_features[:root_count])
            _LIB.mcts_batch_feed_roots(
                tree_ptrs,
                len(active_roots),
                np.ascontiguousarray(priors, dtype=np.float32),
                np.ascontiguousarray(values, dtype=np.float32),
            )

        root_values = np.empty(len(active_roots), dtype=np.float32)
        _LIB.mcts_batch_get_root_values(tree_ptrs, len(active_roots), root_values)

        if add_noise and self.dirichlet_alpha > 0.0 and self.dirichlet_epsilon > 0.0:
            self._apply_root_noise(tree_ptrs)

        leaves_per_batch = max(1, int(leaves_per_batch))
        sims_done = 0
        while sims_done < num_simulations:
            leaves_this_round = min(leaves_per_batch, num_simulations - sims_done)
            max_leaves = len(active_roots) * leaves_this_round
            leaf_features = np.empty((max_leaves, 4, 9, 9), dtype=np.float32)
            if _HAS_THREADED_COLLECT and self.search_threads > 1:
                leaf_count = _LIB.mcts_batch_collect_leaves_threaded(
                    tree_ptrs,
                    len(active_roots),
                    leaves_this_round,
                    leaf_features.reshape(-1),
                    max_leaves,
                    self.search_threads,
                )
            else:
                leaf_count = _LIB.mcts_batch_collect_leaves(
                    tree_ptrs,
                    len(active_roots),
                    leaves_this_round,
                    leaf_features.reshape(-1),
                    max_leaves,
                )
            sims_done += leaves_this_round
            if not leaf_count:
                continue
            priors, values = self.evaluator.evaluate_features(leaf_features[:leaf_count])
            _LIB.mcts_batch_feed_leaves(
                tree_ptrs,
                len(active_roots),
                np.ascontiguousarray(priors, dtype=np.float32),
                np.ascontiguousarray(values, dtype=np.float32),
            )

        counts = np.empty((len(active_roots), ACTION_SIZE), dtype=np.float32)
        _LIB.mcts_batch_extract_visit_counts(tree_ptrs, len(active_roots), counts)
        results: list[SearchResult] = []
        for idx, (state, root, temp) in enumerate(zip(states, active_roots, temperature, strict=True)):
            policy = counts[idx].copy()
            total = float(policy.sum())
            if total == 0.0:
                legal = state.legal_moves().astype(np.float32)
                legal_total = float(legal.sum())
                policy = legal / legal_total if legal_total > 0.0 else legal
            else:
                policy /= total
            action = sample_action_from_policy(policy, temp)
            next_root = None if state.terminal else root
            if next_root is not None:
                next_root.advance(action)
            results.append(
                SearchResult(
                    action=action,
                    visit_policy=policy,
                    root_value=float(root_values[idx]),
                    next_root=next_root,
                )
            )
        return results

    def _apply_root_noise(self, tree_ptrs: np.ndarray) -> None:
        counts = np.empty(len(tree_ptrs), dtype=np.int32)
        _LIB.mcts_batch_root_num_legal(tree_ptrs, len(tree_ptrs), counts)
        offsets = np.zeros(len(tree_ptrs) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts, dtype=np.int32)
        total = int(offsets[-1])
        if total == 0:
            return
        noise = np.empty(total, dtype=np.float32)
        for idx, count in enumerate(counts):
            if count > 0:
                start = int(offsets[idx])
                stop = int(offsets[idx + 1])
                noise[start:stop] = np.random.dirichlet([self.dirichlet_alpha] * int(count))
        _LIB.mcts_batch_apply_root_noise(
            tree_ptrs,
            len(tree_ptrs),
            noise,
            offsets,
            ctypes.c_float(self.dirichlet_epsilon),
        )


def sample_action_from_policy(policy: np.ndarray, temperature: float) -> int:
    if temperature <= 1.0e-6:
        return int(np.argmax(policy))
    adjusted = np.power(np.maximum(policy, 1.0e-8), 1.0 / temperature)
    adjusted /= adjusted.sum()
    return int(np.random.choice(np.arange(policy.size), p=adjusted))
