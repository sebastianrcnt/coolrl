from __future__ import annotations

import numpy as np

from .board import GameState


class Evaluator:
    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def effective_batch_size(self, batch_size: int) -> int:
        return batch_size

    def close(self) -> None:
        return None
