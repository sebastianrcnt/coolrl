from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..game import GameState, LostCitiesConfig
from .config import NetworkConfig
from .memory import _Sample, AdvantageMemory, StrategyMemory
from .networks import AdvantageNet
from .traversal import DeepCFRTraverser, TraversalStats, TraversalTimingStats

_UNBOUNDED_WORKER_MEMORY_CAPACITY = 2_147_483_647
_TORCH_THREADS_CONFIGURED = False


@dataclass(slots=True)
class TraversalWorkerBatch:
    batch_index: int
    lc_config_snapshot: dict[str, Any]
    input_dim: int
    action_size: int
    network_config: NetworkConfig
    advantage_net_state_dicts: list[dict[str, Any]]
    traverser: int
    iteration: int
    seeds: list[int]
    max_depth: int | None
    max_nodes_per_traversal: int | None
    strategy_sample_interval: int
    store_strategy_on_opponent_nodes: bool
    store_strategy_on_traverser_nodes: bool
    profile_hotspots: bool
    regret_matching_epsilon: float
    worker_seed: int


@dataclass(slots=True)
class TraversalWorkerBatchResult:
    batch_index: int
    traverser: int
    seeds_completed: int
    stats: TraversalStats
    advantage_samples: list[_Sample]
    strategy_samples: list[_Sample]
    timing_stats: TraversalTimingStats | None = None


def _worker_memory_capacity(batch: TraversalWorkerBatch) -> int:
    if batch.max_nodes_per_traversal is not None and batch.max_nodes_per_traversal > 0:
        return max(1, int(batch.max_nodes_per_traversal) * max(1, len(batch.seeds)))
    return _UNBOUNDED_WORKER_MEMORY_CAPACITY


def _configure_worker_torch_threads() -> None:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    _TORCH_THREADS_CONFIGURED = True


def _run_traversal_worker_batch(batch: TraversalWorkerBatch) -> TraversalWorkerBatchResult:
    _configure_worker_torch_threads()

    lc_config = LostCitiesConfig(**batch.lc_config_snapshot)
    lc_config.validate()

    # Traversal is node-by-node inference. Workers always reconstruct CPU models,
    # even when the trainer process is using CUDA for network training.
    device = torch.device("cpu")
    advantage_nets: list[AdvantageNet] = []
    for state_dict in batch.advantage_net_state_dicts:
        net = AdvantageNet(batch.input_dim, batch.action_size, batch.network_config).to(device)
        cpu_state_dict = {
            name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in state_dict.items()
        }
        net.load_state_dict(cpu_state_dict)
        net.eval()
        advantage_nets.append(net)

    memory_capacity = _worker_memory_capacity(batch)
    advantage_memories = [
        AdvantageMemory(memory_capacity),
        AdvantageMemory(memory_capacity),
    ]
    strategy_memory = StrategyMemory(memory_capacity)
    rng = np.random.default_rng(batch.worker_seed)
    timing_stats = TraversalTimingStats() if batch.profile_hotspots else None
    traverser = DeepCFRTraverser(
        advantage_nets,
        advantage_memories,
        strategy_memory,
        device=device,
        epsilon=batch.regret_matching_epsilon,
        strategy_sample_interval=batch.strategy_sample_interval,
        store_strategy_on_opponent_nodes=batch.store_strategy_on_opponent_nodes,
        store_strategy_on_traverser_nodes=batch.store_strategy_on_traverser_nodes,
        max_depth=batch.max_depth,
        max_nodes_per_traversal=batch.max_nodes_per_traversal,
        rng=rng,
        timing_stats=timing_stats,
    )

    total_stats = TraversalStats()
    for seed in batch.seeds:
        state = GameState.new_game(lc_config, seed=seed)
        _, stats = traverser.traverse(state, batch.traverser, batch.iteration)
        total_stats.nodes += stats.nodes
        total_stats.terminals += stats.terminals
        total_stats.cutoffs += stats.cutoffs
        total_stats.node_limit_cutoffs += stats.node_limit_cutoffs
        total_stats.max_depth_reached = max(total_stats.max_depth_reached, stats.max_depth_reached)

    return TraversalWorkerBatchResult(
        batch_index=batch.batch_index,
        traverser=batch.traverser,
        seeds_completed=len(batch.seeds),
        stats=total_stats,
        advantage_samples=list(advantage_memories[batch.traverser].samples),
        strategy_samples=list(strategy_memory.samples),
        timing_stats=timing_stats,
    )
