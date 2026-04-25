from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import RunConfig, config_from_dict
from .trainer import DeepCFRTrainer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TraversalBenchmarkResult:
    num_workers: int
    requested_workers: int
    effective_workers: int
    num_batches: int
    traversal_seconds: float
    total_nodes: int
    traversals: int
    nodes_per_second: float
    avg_nodes_per_traversal: float
    cutoffs: int
    cutoff_rate: float
    node_limit_cutoffs: int
    node_limit_cutoff_rate: float
    cutoff_rollouts: int
    cutoff_rollout_steps: int
    cutoff_rollout_max_step_timeouts: int


def _benchmark_config_variant(config: RunConfig, *, num_workers: int, checkpoint_dir: Path) -> RunConfig:
    payload = config.to_dict()
    # Force CPU traversal in both modes so the comparison measures multiprocessing
    # speedup rather than CUDA transfer behavior or mixed CPU/GPU traversal paths.
    payload["device"] = "cpu"
    payload["max_iterations"] = 0
    payload.setdefault("evaluation", {})
    payload["evaluation"]["eval_every"] = 0
    payload.setdefault("checkpoint", {})
    payload["checkpoint"]["directory"] = str(checkpoint_dir)
    payload["checkpoint"]["save_latest_only"] = True
    payload.setdefault("traversal", {})
    payload["traversal"]["num_workers"] = int(num_workers)
    payload["traversal"]["progress_every_traversals"] = 0
    payload["traversal"]["profile_hotspots"] = False
    return config_from_dict(payload)


def _run_traversal_benchmark_once(config: RunConfig, *, iteration: int) -> TraversalBenchmarkResult:
    trainer = DeepCFRTrainer(config)
    requested_workers = trainer.num_workers
    effective_workers = requested_workers
    num_batches = 0
    if requested_workers > 1:
        num_batches = trainer._estimated_traversal_batch_count()
        effective_workers = min(requested_workers, num_batches)

    started = time.monotonic()
    if requested_workers <= 1:
        total_stats, traversals, _ = trainer._run_traversals_single_process(iteration)
    else:
        total_stats, traversals, _ = trainer._run_traversals_parallel(iteration)
    traversal_seconds = time.monotonic() - started
    return TraversalBenchmarkResult(
        num_workers=requested_workers,
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        num_batches=num_batches,
        traversal_seconds=traversal_seconds,
        total_nodes=total_stats.nodes,
        traversals=traversals,
        nodes_per_second=total_stats.nodes / max(1.0e-9, traversal_seconds),
        avg_nodes_per_traversal=total_stats.nodes / max(1, traversals),
        cutoffs=total_stats.cutoffs,
        cutoff_rate=total_stats.cutoffs / max(1, total_stats.nodes),
        node_limit_cutoffs=total_stats.node_limit_cutoffs,
        node_limit_cutoff_rate=total_stats.node_limit_cutoffs / max(1, total_stats.nodes),
        cutoff_rollouts=total_stats.cutoff_rollouts,
        cutoff_rollout_steps=total_stats.cutoff_rollout_steps,
        cutoff_rollout_max_step_timeouts=total_stats.cutoff_rollout_max_step_timeouts,
    )


def _result_to_dict(r: TraversalBenchmarkResult) -> dict[str, Any]:
    return {
        "num_workers": r.num_workers,
        "requested_workers": r.requested_workers,
        "effective_workers": r.effective_workers,
        "num_batches": r.num_batches,
        "traversal_seconds": r.traversal_seconds,
        "total_nodes": r.total_nodes,
        "traversals": r.traversals,
        "nodes_per_second": r.nodes_per_second,
        "avg_nodes_per_traversal": r.avg_nodes_per_traversal,
        "cutoffs": r.cutoffs,
        "cutoff_rate": r.cutoff_rate,
        "node_limit_cutoffs": r.node_limit_cutoffs,
        "node_limit_cutoff_rate": r.node_limit_cutoff_rate,
        "cutoff_rollouts": r.cutoff_rollouts,
        "cutoff_rollout_steps": r.cutoff_rollout_steps,
        "cutoff_rollout_max_step_timeouts": r.cutoff_rollout_max_step_timeouts,
    }


def benchmark_traversal_modes(
    config: RunConfig,
    *,
    mp_workers: int = 2,
    iteration: int = 1,
    mode: Literal["compare", "single", "mp"] = "compare",
) -> dict[str, Any]:
    if mode in ("compare", "mp") and mp_workers <= 1:
        raise ValueError(f"mp_workers must be > 1 for mode={mode!r}, got {mp_workers}")

    with tempfile.TemporaryDirectory(prefix="lost_cities_deep_cfr_benchmark_") as temp_dir:
        base_dir = Path(temp_dir)

        single_result: TraversalBenchmarkResult | None = None
        mp_result: TraversalBenchmarkResult | None = None

        if mode in ("compare", "single"):
            logger.info("Running single-process traversal benchmark...")
            single_result = _run_traversal_benchmark_once(
                _benchmark_config_variant(config, num_workers=0, checkpoint_dir=base_dir / "single"),
                iteration=iteration,
            )

        if mode in ("compare", "mp"):
            logger.info("Running multiprocessing traversal benchmark with requested_workers=%s...", mp_workers)
            mp_result = _run_traversal_benchmark_once(
                _benchmark_config_variant(config, num_workers=mp_workers, checkpoint_dir=base_dir / "multi"),
                iteration=iteration,
            )

    out: dict[str, Any] = {"device_used": "cpu", "iteration": iteration}

    if single_result is not None:
        out["single_process"] = _result_to_dict(single_result)
    if mp_result is not None:
        out["multiprocessing"] = _result_to_dict(mp_result)

    if single_result is not None and mp_result is not None:
        out["speedup_vs_single_process"] = single_result.traversal_seconds / max(1.0e-9, mp_result.traversal_seconds)
    else:
        out["speedup_vs_single_process"] = "n/a"

    return out
