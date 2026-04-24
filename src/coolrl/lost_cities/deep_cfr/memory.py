from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _Sample:
    info_state: np.ndarray
    target: np.ndarray
    legal_mask: np.ndarray
    player: int
    iteration: int


class _ReservoirMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.samples: list[_Sample] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _add_sample(
        self,
        info_state: np.ndarray,
        target: np.ndarray,
        legal_mask: np.ndarray,
        player: int,
        iteration: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.seen += 1
        sample = _Sample(
            info_state=np.asarray(info_state, dtype=np.float32).copy(),
            target=np.asarray(target, dtype=np.float32).copy(),
            legal_mask=np.asarray(legal_mask, dtype=bool).copy(),
            player=int(player),
            iteration=int(iteration),
        )
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return
        rng = rng or np.random.default_rng()
        index = int(rng.integers(0, self.seen))
        if index < self.capacity:
            self.samples[index] = sample

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if not self.samples:
            raise ValueError("cannot sample from empty memory")
        size = min(int(batch_size), len(self.samples))
        indices = rng.choice(len(self.samples), size=size, replace=len(self.samples) < size)
        batch = [self.samples[int(index)] for index in indices]
        return {
            "info_state": np.stack([sample.info_state for sample in batch]).astype(np.float32),
            "target": np.stack([sample.target for sample in batch]).astype(np.float32),
            "legal_mask": np.stack([sample.legal_mask for sample in batch]).astype(bool),
            "player": np.asarray([sample.player for sample in batch], dtype=np.int64),
            "iteration": np.asarray([sample.iteration for sample in batch], dtype=np.int64),
        }


class AdvantageMemory(_ReservoirMemory):
    def add(
        self,
        info_state: np.ndarray,
        regrets: np.ndarray,
        legal_mask: np.ndarray,
        player: int,
        iteration: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._add_sample(info_state, regrets, legal_mask, player, iteration, rng)


class StrategyMemory(_ReservoirMemory):
    def add(
        self,
        info_state: np.ndarray,
        policy: np.ndarray,
        legal_mask: np.ndarray,
        player: int,
        iteration: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._add_sample(info_state, policy, legal_mask, player, iteration, rng)
