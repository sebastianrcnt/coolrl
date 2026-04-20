from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from .board import GameState
from .features import states_to_feature_planes

if TYPE_CHECKING:
    from .network import PolicyValueNet

try:
    from tinygrad import Tensor
except ModuleNotFoundError:  # pragma: no cover
    Tensor = None


class Evaluator:
    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def effective_batch_size(self, batch_size: int) -> int:
        return batch_size

    def close(self) -> None:
        return None


class ModelEvaluator(Evaluator):
    def __init__(self, model: "PolicyValueNet", device: str | None = None) -> None:
        if Tensor is None:
            raise RuntimeError("tinygrad not installed. Install with `uv sync --extra omok`.")
        self.model = model
        self.device = device
        self._seen_buckets: set[int] = set()

    def effective_batch_size(self, batch_size: int) -> int:
        return 1 << (max(batch_size, 1) - 1).bit_length()

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        features = states_to_feature_planes(states)
        return self.evaluate_features(features)

    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Pad the batch dimension to the next power of two. tinygrad JIT-compiles
        # a separate kernel for every (shape, dtype) combination it sees, and
        # self-play produces a wide spread of active-game counts as games
        # terminate at different times. Rounding to 1/2/4/.../128 collapses that
        # spread to at most ~8 unique shapes and eliminates the per-shape
        # compile stalls.
        n = features.shape[0]
        bucket = 1 << (max(n, 1) - 1).bit_length()
        if bucket not in self._seen_buckets:
            self._seen_buckets.add(bucket)
            logger.info(
                "ModelEvaluator JIT bucket: size={} (n={}, device={}) — first use, tinygrad will compile kernels for this shape",
                bucket,
                n,
                self.device,
            )
        if bucket > n:
            pad_shape = (bucket - n, *features.shape[1:])
            features = np.concatenate(
                [features, np.zeros(pad_shape, dtype=features.dtype)],
                axis=0,
            )

        from .features import states_to_feature_planes  # circular-safe

        tensor = Tensor(np.ascontiguousarray(features), device=self.device)
        with Tensor.train(False):
            logits, values = self.model(tensor)
            priors = logits.softmax(axis=1).realize().numpy()
            value_np = values.realize().numpy()
        if bucket > n:
            priors = priors[:n]
            value_np = value_np[:n]
        return priors.astype(np.float32, copy=False), value_np.astype(np.float32, copy=False)
