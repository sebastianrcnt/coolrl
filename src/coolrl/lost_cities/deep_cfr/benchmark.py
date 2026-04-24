from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig, config_from_dict
from .trainer import DeepCFRTrainer


@dataclass(slots=True)
class TraversalBenchmarkResult:
    num_workers: int
    traversal_seconds: float
    total_nodes: int
    traversals: int
    nodes_per_second: float
    avg_nodes_per_traversal: float


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
    started = time.monotonic()
    if trainer.num_workers <= 1:
        total_stats, traversals, _ = trainer._run_traversals_single_process(iteration)
    else:
        total_stats, traversals, _ = trainer._run_traversals_parallel(iteration)
    traversal_seconds = time.monotonic() - started
    return TraversalBenchmarkResult(
        num_workers=trainer.num_workers,
        traversal_seconds=traversal_seconds,
        total_nodes=total_stats.nodes,
        traversals=traversals,
        nodes_per_second=total_stats.nodes / max(1.0e-9, traversal_seconds),
        avg_nodes_per_traversal=total_stats.nodes / max(1, traversals),
    )


def benchmark_traversal_modes(
    config: RunConfig,
    *,
    mp_workers: int,
    iteration: int = 1,
) -> dict[str, Any]:
    if mp_workers <= 1:
        raise ValueError(f"mp_workers must be > 1 for comparison benchmarking, got {mp_workers}")

    with tempfile.TemporaryDirectory(prefix="lost_cities_deep_cfr_benchmark_") as temp_dir:
        base_dir = Path(temp_dir)
        single = _run_traversal_benchmark_once(
            _benchmark_config_variant(config, num_workers=0, checkpoint_dir=base_dir / "single"),
            iteration=iteration,
        )
        multiprocessing = _run_traversal_benchmark_once(
            _benchmark_config_variant(config, num_workers=mp_workers, checkpoint_dir=base_dir / "multi"),
            iteration=iteration,
        )

    return {
        "device_used": "cpu",
        "iteration": iteration,
        "single_process": {
            "num_workers": single.num_workers,
            "traversal_seconds": single.traversal_seconds,
            "total_nodes": single.total_nodes,
            "traversals": single.traversals,
            "nodes_per_second": single.nodes_per_second,
            "avg_nodes_per_traversal": single.avg_nodes_per_traversal,
        },
        "multiprocessing": {
            "num_workers": multiprocessing.num_workers,
            "traversal_seconds": multiprocessing.traversal_seconds,
            "total_nodes": multiprocessing.total_nodes,
            "traversals": multiprocessing.traversals,
            "nodes_per_second": multiprocessing.nodes_per_second,
            "avg_nodes_per_traversal": multiprocessing.avg_nodes_per_traversal,
        },
        "speedup_vs_single_process": single.traversal_seconds / max(1.0e-9, multiprocessing.traversal_seconds),
    }
