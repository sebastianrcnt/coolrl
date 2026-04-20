from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import numpy as np

from .board import GameState
from .evaluator import Evaluator


T = TypeVar("T")


def _round_seconds(value: float) -> float:
    return round(value, 6)


@dataclass(slots=True)
class EvaluatorMetrics:
    calls: int = 0
    positions: int = 0
    padded_positions: int = 0
    seconds: float = 0.0
    max_batch: int = 0
    max_bucket: int = 0
    bucket_counts: dict[int, int] = field(default_factory=dict)

    def record(self, batch_size: int, effective_batch_size: int, seconds: float) -> None:
        bucket = max(1, effective_batch_size)
        self.calls += 1
        self.positions += batch_size
        self.padded_positions += bucket
        self.seconds += seconds
        self.max_batch = max(self.max_batch, batch_size)
        self.max_bucket = max(self.max_bucket, bucket)
        self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1

    @property
    def avg_batch(self) -> float:
        return 0.0 if self.calls == 0 else self.positions / self.calls

    @property
    def avg_seconds(self) -> float:
        return 0.0 if self.calls == 0 else self.seconds / self.calls

    @property
    def pad_ratio(self) -> float:
        return 0.0 if self.positions == 0 else self.padded_positions / self.positions


@dataclass(slots=True)
class SearchMetrics:
    calls: int = 0
    states: int = 0
    simulations: int = 0
    requested_leaves: int = 0
    seconds: float = 0.0
    max_states: int = 0
    max_simulations: int = 0
    max_leaves_per_batch: int = 0

    def record(self, states: int, simulations: int, leaves_per_batch: int, seconds: float) -> None:
        self.calls += 1
        self.states += states
        self.simulations += simulations
        self.requested_leaves += states * simulations
        self.seconds += seconds
        self.max_states = max(self.max_states, states)
        self.max_simulations = max(self.max_simulations, simulations)
        self.max_leaves_per_batch = max(self.max_leaves_per_batch, leaves_per_batch)

    @property
    def avg_states(self) -> float:
        return 0.0 if self.calls == 0 else self.states / self.calls

    @property
    def avg_seconds(self) -> float:
        return 0.0 if self.calls == 0 else self.seconds / self.calls


@dataclass(slots=True)
class TrainingMetrics:
    updates: int = 0
    samples: int = 0
    sample_seconds: float = 0.0
    forward_seconds: float = 0.0
    loss_seconds: float = 0.0
    backward_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    sync_seconds: float = 0.0

    def record(
        self,
        *,
        batch_size: int,
        sample_seconds: float,
        forward_seconds: float,
        loss_seconds: float,
        backward_seconds: float,
        optimizer_seconds: float,
        sync_seconds: float,
    ) -> None:
        self.updates += 1
        self.samples += batch_size
        self.sample_seconds += sample_seconds
        self.forward_seconds += forward_seconds
        self.loss_seconds += loss_seconds
        self.backward_seconds += backward_seconds
        self.optimizer_seconds += optimizer_seconds
        self.sync_seconds += sync_seconds

    @property
    def total_measured_seconds(self) -> float:
        return (
            self.sample_seconds
            + self.forward_seconds
            + self.loss_seconds
            + self.backward_seconds
            + self.optimizer_seconds
            + self.sync_seconds
        )


class IterationMetrics:
    def __init__(self) -> None:
        self.evaluator: dict[str, EvaluatorMetrics] = {}
        self.search: dict[str, SearchMetrics] = {}
        self.training = TrainingMetrics()

    def timed_evaluator(self, phase: str, evaluator: Evaluator) -> "TimedEvaluator":
        return TimedEvaluator(evaluator=evaluator, metrics=self, phase=phase)

    def record_evaluator(self, phase: str, batch_size: int, effective_batch_size: int, seconds: float) -> None:
        self.evaluator.setdefault(phase, EvaluatorMetrics()).record(batch_size, effective_batch_size, seconds)

    def record_search(self, phase: str, states: int, simulations: int, leaves_per_batch: int, seconds: float) -> None:
        self.search.setdefault(phase, SearchMetrics()).record(states, simulations, leaves_per_batch, seconds)

    def time_search(
        self,
        phase: str,
        states: int,
        simulations: int,
        leaves_per_batch: int,
        callback: Callable[[], T],
    ) -> T:
        started = time.perf_counter()
        try:
            return callback()
        finally:
            self.record_search(phase, states, simulations, leaves_per_batch, time.perf_counter() - started)

    def to_log_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {}
        for phase, stats in sorted(self.search.items()):
            prefix = f"search_{phase}"
            fields[f"{prefix}_calls"] = stats.calls
            fields[f"{prefix}_seconds"] = _round_seconds(stats.seconds)
            fields[f"{prefix}_avg_seconds"] = _round_seconds(stats.avg_seconds)
            fields[f"{prefix}_states"] = stats.states
            fields[f"{prefix}_avg_states"] = round(stats.avg_states, 2)
            fields[f"{prefix}_requested_leaves"] = stats.requested_leaves
            fields[f"{prefix}_max_states"] = stats.max_states
            fields[f"{prefix}_max_simulations"] = stats.max_simulations
            fields[f"{prefix}_max_leaves_per_batch"] = stats.max_leaves_per_batch

        for phase, stats in sorted(self.evaluator.items()):
            prefix = f"eval_{phase}"
            fields[f"{prefix}_calls"] = stats.calls
            fields[f"{prefix}_seconds"] = _round_seconds(stats.seconds)
            fields[f"{prefix}_avg_seconds"] = _round_seconds(stats.avg_seconds)
            fields[f"{prefix}_positions"] = stats.positions
            fields[f"{prefix}_padded_positions"] = stats.padded_positions
            fields[f"{prefix}_pad_ratio"] = round(stats.pad_ratio, 4)
            fields[f"{prefix}_avg_batch"] = round(stats.avg_batch, 2)
            fields[f"{prefix}_max_batch"] = stats.max_batch
            fields[f"{prefix}_max_bucket"] = stats.max_bucket
            fields[f"{prefix}_bucket_counts"] = dict(sorted(stats.bucket_counts.items()))

        train = self.training
        fields.update(
            {
                "train_metric_updates": train.updates,
                "train_metric_samples": train.samples,
                "train_sample_seconds": _round_seconds(train.sample_seconds),
                "train_forward_seconds": _round_seconds(train.forward_seconds),
                "train_loss_seconds": _round_seconds(train.loss_seconds),
                "train_backward_seconds": _round_seconds(train.backward_seconds),
                "train_optimizer_seconds": _round_seconds(train.optimizer_seconds),
                "train_sync_seconds": _round_seconds(train.sync_seconds),
                "train_measured_seconds": _round_seconds(train.total_measured_seconds),
            }
        )
        return fields


class TimedEvaluator(Evaluator):
    def __init__(self, evaluator: Evaluator, metrics: IterationMetrics, phase: str) -> None:
        self.evaluator = evaluator
        self.metrics = metrics
        self.phase = phase

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter()
        try:
            return self.evaluator.evaluate(states)
        finally:
            batch_size = len(states)
            self.metrics.record_evaluator(
                self.phase,
                batch_size,
                self.evaluator.effective_batch_size(batch_size),
                time.perf_counter() - started,
            )

    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter()
        try:
            return self.evaluator.evaluate_features(features)  # type: ignore[attr-defined]
        finally:
            batch_size = int(features.shape[0])
            self.metrics.record_evaluator(
                self.phase,
                batch_size,
                self.evaluator.effective_batch_size(batch_size),
                time.perf_counter() - started,
            )

    def close(self) -> None:
        self.evaluator.close()
