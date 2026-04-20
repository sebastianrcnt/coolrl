from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .board import GameState


@dataclass(slots=True)
class SearchResult:
    action: int
    visit_policy: np.ndarray
    root_value: float
    next_root: object | None = None


class MCTSBackend(Protocol):
    def search_batch(
        self,
        states: list[GameState],
        num_simulations: int,
        temperature: list[float],
        add_noise: bool,
        roots: list[Any | None] | None = None,
        leaves_per_batch: int = 1,
    ) -> list[SearchResult]: ...
