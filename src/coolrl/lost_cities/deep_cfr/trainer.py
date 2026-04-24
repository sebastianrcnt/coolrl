from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from ..game import GameState
from .config import RunConfig
from .encoding import infer_input_dim
from .evaluate import evaluate_against_bot, make_opponent
from .memory import AdvantageMemory, StrategyMemory
from .networks import AdvantageNet, StrategyNet
from .traversal import DeepCFRTraverser, TraversalStats, TraversalTimingStats
from .traversal_worker import (
    TraversalWorkerBatch,
    TraversalWorkerBatchResult,
    _run_traversal_worker_batch,
)


def _torch_device(name: str) -> torch.device:
    token = name.strip().lower()
    if token == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is unavailable")
        return torch.device("cuda")
    if token == "cpu":
        return torch.device("cpu")
    if token == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device: {name!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepCFRTrainer:
    def __init__(self, config: RunConfig, resume_path: str | None = None) -> None:
        if config.traversal.backend != "python":
            raise ValueError("Lost Cities Deep CFR MVP only supports traversal.backend=python")
        self.config = config
        self.device = _torch_device(config.device)
        set_seed(config.seed)
        self.lc_config = config.rules.to_lost_cities_config(seed=config.seed)
        self.input_dim = infer_input_dim(self.lc_config)
        self.action_size = self.lc_config.action_size
        self.advantage_nets = [
            AdvantageNet(self.input_dim, self.action_size, config.network).to(self.device),
            AdvantageNet(self.input_dim, self.action_size, config.network).to(self.device),
        ]
        self.strategy_net = StrategyNet(self.input_dim, self.action_size, config.network).to(self.device)
        self.advantage_optimizers = [
            torch.optim.AdamW(net.parameters(), lr=config.optimization.learning_rate, weight_decay=config.optimization.weight_decay)
            for net in self.advantage_nets
        ]
        self.strategy_optimizer = torch.optim.AdamW(
            self.strategy_net.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        self.advantage_memories = [
            AdvantageMemory(config.memory.advantage_capacity),
            AdvantageMemory(config.memory.advantage_capacity),
        ]
        self.strategy_memory = StrategyMemory(config.memory.strategy_capacity)
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.checkpoint_dir / "metrics.jsonl"
        self.progress_path = self.checkpoint_dir / "runtime_progress.json"
        self.num_workers, _ = config.traversal.resolved_num_workers()
        self.traversal_worker_chunk_size = max(1, int(config.traversal.traversal_worker_chunk_size))
        self.profile_hotspots = bool(config.traversal.profile_hotspots)
        if resume_path is None and self.metrics_path.exists():
            archive = self.metrics_path.with_name(f"metrics.{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
            shutil.move(self.metrics_path, archive)
        self.rng = np.random.default_rng(config.seed + 811)
        self.iteration = 0
        self.start_time = time.monotonic()
        if resume_path:
            self.load_checkpoint(resume_path)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def run(self) -> None:
        logger.info(
            "Starting Lost Cities Deep CFR: experiment={} tier={} device={} input_dim={} actions={}",
            self.config.experiment_name,
            self.config.rules.tier,
            self.device,
            self.input_dim,
            self.action_size,
        )
        while self._should_continue():
            self.iteration += 1
            metrics = self.run_iteration(self.iteration)
            self._append_metrics(metrics)
            self._write_progress(metrics)
            if self._should_save(self.iteration):
                if not self.config.checkpoint.save_latest_only:
                    self.save_checkpoint(self.checkpoint_dir / f"iteration_{self.iteration:05d}.pt", metrics)
                self.save_checkpoint(self.checkpoint_dir / "latest.pt", metrics)

    def _should_continue(self) -> bool:
        if self.config.max_iterations is not None and self.iteration >= self.config.max_iterations:
            return False
        if self.config.max_hours is not None and self.elapsed_seconds >= self.config.max_hours * 3600.0:
            return False
        return True

    def _should_save(self, iteration: int) -> bool:
        if self.config.checkpoint.save_every_iteration:
            return True
        interval = max(1, self.config.checkpoint.save_iteration_interval)
        return iteration % interval == 0

    def run_iteration(self, iteration: int) -> dict[str, Any]:
        traversal_started = time.monotonic()
        if self.num_workers <= 1:
            total_stats, traversals, hotspot_stats = self._run_traversals_single_process(iteration)
        else:
            total_stats, traversals, hotspot_stats = self._run_traversals_parallel(iteration)
        traversal_seconds = time.monotonic() - traversal_started

        adv_started = time.monotonic()
        advantage_losses = [
            self._train_advantage(player, self.config.optimization.advantage_updates_per_iteration)
            for player in (0, 1)
        ]
        train_advantage_seconds = time.monotonic() - adv_started

        strategy_started = time.monotonic()
        strategy_loss = self._train_strategy(self.config.optimization.strategy_updates_per_iteration)
        train_strategy_seconds = time.monotonic() - strategy_started

        eval_started = time.monotonic()
        eval_metrics: dict[str, Any] = {}
        if self.config.evaluation.eval_every > 0 and iteration % self.config.evaluation.eval_every == 0:
            for opponent_name in self.config.evaluation.opponents:
                opponent = make_opponent(opponent_name, seed=self.config.seed + iteration)
                result = evaluate_against_bot(
                    self.strategy_net,
                    opponent,
                    self.lc_config,
                    self.config.evaluation.games,
                    self.config.seed + 50_000 + iteration * 100,
                    device=self.device,
                )
                eval_metrics[f"eval_{opponent_name}_win_rate"] = result["win_rate"]
                eval_metrics[f"eval_{opponent_name}_avg_diff"] = result["avg_diff"]
        eval_seconds = time.monotonic() - eval_started

        nodes_per_second = total_stats.nodes / max(1.0e-9, traversal_seconds)
        cutoff_rate = total_stats.cutoffs / max(1, total_stats.nodes)
        node_limit_cutoff_rate = total_stats.node_limit_cutoffs / max(1, total_stats.nodes)
        metrics: dict[str, Any] = {
            "iteration": iteration,
            "elapsed_seconds": self.elapsed_seconds,
            "traversal_seconds": traversal_seconds,
            "train_advantage_seconds": train_advantage_seconds,
            "train_strategy_seconds": train_strategy_seconds,
            "eval_seconds": eval_seconds,
            "total_nodes": total_stats.nodes,
            "total_cutoffs": total_stats.cutoffs,
            "cutoff_rate": cutoff_rate,
            "total_node_limit_cutoffs": total_stats.node_limit_cutoffs,
            "node_limit_cutoff_rate": node_limit_cutoff_rate,
            "nodes_per_second": nodes_per_second,
            "traversals_per_second": traversals / max(1.0e-9, traversal_seconds),
            "avg_nodes_per_traversal": total_stats.nodes / max(1, traversals),
            "advantage_memory_size_p0": len(self.advantage_memories[0]),
            "advantage_memory_size_p1": len(self.advantage_memories[1]),
            "strategy_memory_size": len(self.strategy_memory),
            "advantage_loss_p0": advantage_losses[0],
            "advantage_loss_p1": advantage_losses[1],
            "strategy_loss": strategy_loss,
            **eval_metrics,
        }
        if hotspot_stats is not None:
            metrics.update(self._hotspot_metrics(hotspot_stats))
        logger.info(
            "Iteration {}: nodes={} cutoffs={} node_limit_cutoffs={} cutoff_rate={:.4f} node_limit_cutoff_rate={:.4f} nps={:.1f} adv_loss=({:.4f},{:.4f}) strategy_loss={:.4f}",
            iteration,
            total_stats.nodes,
            total_stats.cutoffs,
            total_stats.node_limit_cutoffs,
            cutoff_rate,
            node_limit_cutoff_rate,
            nodes_per_second,
            advantage_losses[0],
            advantage_losses[1],
            strategy_loss,
        )
        if hotspot_stats is not None:
            logger.info(
                "Traversal hotspots: wall={:.3f}s encode={:.3f}s forward={:.3f}s regret={:.3f}s clone_apply={:.3f}s memory_add={:.3f}s policy_calls={} clone_apply_calls={} memory_add_calls={}",
                hotspot_stats.traversal_wall_seconds,
                hotspot_stats.encode_information_state_seconds,
                hotspot_stats.advantage_forward_seconds,
                hotspot_stats.regret_matching_seconds,
                hotspot_stats.clone_apply_seconds,
                hotspot_stats.memory_add_seconds,
                hotspot_stats.policy_calls,
                hotspot_stats.clone_apply_calls,
                hotspot_stats.memory_add_calls,
            )
        return metrics

    def _traversal_seed(self, iteration: int, player: int, index: int) -> int:
        return self.config.seed + iteration * 1_000_003 + player * 100_003 + index

    def _hotspot_metrics(self, hotspot_stats: TraversalTimingStats) -> dict[str, Any]:
        return {
            "traversal_profile_wall_seconds": hotspot_stats.traversal_wall_seconds,
            "traversal_profile_encode_seconds": hotspot_stats.encode_information_state_seconds,
            "traversal_profile_forward_seconds": hotspot_stats.advantage_forward_seconds,
            "traversal_profile_regret_matching_seconds": hotspot_stats.regret_matching_seconds,
            "traversal_profile_clone_apply_seconds": hotspot_stats.clone_apply_seconds,
            "traversal_profile_memory_add_seconds": hotspot_stats.memory_add_seconds,
            "traversal_profile_policy_calls": hotspot_stats.policy_calls,
            "traversal_profile_clone_apply_calls": hotspot_stats.clone_apply_calls,
            "traversal_profile_memory_add_calls": hotspot_stats.memory_add_calls,
        }

    def _run_traversals_single_process(self, iteration: int) -> tuple[TraversalStats, int, TraversalTimingStats | None]:
        total_stats = TraversalStats()
        hotspot_stats = TraversalTimingStats() if self.profile_hotspots else None
        traverser = DeepCFRTraverser(
            self.advantage_nets,
            self.advantage_memories,
            self.strategy_memory,
            device=self.device,
            epsilon=self.config.traversal.regret_matching_epsilon,
            strategy_sample_interval=self.config.traversal.strategy_sample_interval,
            store_strategy_on_opponent_nodes=self.config.traversal.store_strategy_on_opponent_nodes,
            store_strategy_on_traverser_nodes=self.config.traversal.store_strategy_on_traverser_nodes,
            max_depth=self.config.traversal.max_depth,
            max_nodes_per_traversal=self.config.traversal.max_nodes_per_traversal,
            rng=self.rng,
            timing_stats=hotspot_stats,
        )
        traversals = 0
        for player in (0, 1):
            self.advantage_nets[player].eval()
            player_traversal_started = time.monotonic()
            for index in range(self.config.traversal.traversals_per_player):
                seed = self._traversal_seed(iteration, player, index)
                state = GameState.new_game(self.lc_config, seed=seed)
                _, stats = traverser.traverse(state, player, iteration)
                total_stats.nodes += stats.nodes
                total_stats.terminals += stats.terminals
                total_stats.cutoffs += stats.cutoffs
                total_stats.node_limit_cutoffs += stats.node_limit_cutoffs
                total_stats.max_depth_reached = max(total_stats.max_depth_reached, stats.max_depth_reached)
                traversals += 1
                progress_every = self.config.traversal.progress_every_traversals
                if progress_every > 0 and (index + 1) % progress_every == 0:
                    elapsed = time.monotonic() - player_traversal_started
                    avg_nodes = total_stats.nodes / max(1, traversals)
                    logger.info(
                        "Traversal progress: player={} completed={} elapsed_seconds={:.2f} total_nodes={} avg_nodes_per_traversal={:.1f}",
                        player,
                        index + 1,
                        elapsed,
                        total_stats.nodes,
                        avg_nodes,
                    )
        return total_stats, traversals, hotspot_stats

    def _frozen_advantage_state_dicts(self) -> list[dict[str, Any]]:
        frozen_state_dicts: list[dict[str, Any]] = []
        for net in self.advantage_nets:
            net.eval()
            frozen_state_dicts.append(
                {
                    name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                    for name, value in net.state_dict().items()
                }
            )
        return frozen_state_dicts

    def _build_traversal_worker_batches(
        self,
        iteration: int,
        advantage_net_state_dicts: list[dict[str, Any]],
    ) -> list[TraversalWorkerBatch]:
        batches: list[TraversalWorkerBatch] = []
        batch_index = 0
        for player in (0, 1):
            seeds = [
                self._traversal_seed(iteration, player, index)
                for index in range(self.config.traversal.traversals_per_player)
            ]
            for start in range(0, len(seeds), self.traversal_worker_chunk_size):
                chunk = seeds[start : start + self.traversal_worker_chunk_size]
                batches.append(
                    TraversalWorkerBatch(
                        batch_index=batch_index,
                        lc_config_snapshot=self.lc_config.to_snapshot(),
                        input_dim=self.input_dim,
                        action_size=self.action_size,
                        network_config=self.config.network,
                        advantage_net_state_dicts=advantage_net_state_dicts,
                        traverser=player,
                        iteration=iteration,
                        seeds=chunk,
                        max_depth=self.config.traversal.max_depth,
                        max_nodes_per_traversal=self.config.traversal.max_nodes_per_traversal,
                        strategy_sample_interval=self.config.traversal.strategy_sample_interval,
                        store_strategy_on_opponent_nodes=self.config.traversal.store_strategy_on_opponent_nodes,
                        store_strategy_on_traverser_nodes=self.config.traversal.store_strategy_on_traverser_nodes,
                        profile_hotspots=self.profile_hotspots,
                        regret_matching_epsilon=self.config.traversal.regret_matching_epsilon,
                        worker_seed=self.config.seed + iteration * 10_000_019 + player * 1_000_003 + batch_index,
                    )
                )
                batch_index += 1
        return batches

    def _merge_advantage_samples(self, player: int, samples: list[Any]) -> None:
        if samples:
            self.advantage_memories[player].extend_samples(samples, self.rng)

    def _merge_strategy_samples(self, samples: list[Any]) -> None:
        if samples:
            self.strategy_memory.extend_samples(samples, self.rng)

    def _accumulate_parallel_traversal_result(
        self,
        total_stats: TraversalStats,
        result: TraversalWorkerBatchResult,
        hotspot_stats: TraversalTimingStats | None = None,
    ) -> int:
        total_stats.nodes += result.stats.nodes
        total_stats.terminals += result.stats.terminals
        total_stats.cutoffs += result.stats.cutoffs
        total_stats.node_limit_cutoffs += result.stats.node_limit_cutoffs
        total_stats.max_depth_reached = max(total_stats.max_depth_reached, result.stats.max_depth_reached)
        self._merge_advantage_samples(result.traverser, result.advantage_samples)
        self._merge_strategy_samples(result.strategy_samples)
        if hotspot_stats is not None:
            hotspot_stats.accumulate(result.timing_stats)
        return result.seeds_completed

    def _merge_parallel_traversal_results(
        self,
        results: list[TraversalWorkerBatchResult],
    ) -> tuple[TraversalStats, int]:
        total_stats = TraversalStats()
        traversals = 0
        for result in sorted(results, key=lambda item: item.batch_index):
            traversals += self._accumulate_parallel_traversal_result(total_stats, result)
        return total_stats, traversals

    def _run_traversals_parallel(self, iteration: int) -> tuple[TraversalStats, int, TraversalTimingStats | None]:
        advantage_net_state_dicts = self._frozen_advantage_state_dicts()
        batches = self._build_traversal_worker_batches(iteration, advantage_net_state_dicts)
        if not batches:
            return TraversalStats(), 0, TraversalTimingStats() if self.profile_hotspots else None

        max_workers = min(self.num_workers, len(batches))
        logger.info(
            "Traversal multiprocessing enabled: num_workers={} batches={} chunk_size={}",
            max_workers,
            len(batches),
            self.traversal_worker_chunk_size,
        )

        total_stats = TraversalStats()
        traversals = 0
        hotspot_stats = TraversalTimingStats() if self.profile_hotspots else None
        pending_results: dict[int, TraversalWorkerBatchResult] = {}
        next_batch_index = 0
        progress_nodes = 0
        progress_traversals = 0
        progress_every = self.config.traversal.progress_every_traversals
        next_progress_at = progress_every if progress_every > 0 else None
        progress_started = time.monotonic()
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(_run_traversal_worker_batch, batch) for batch in batches]
            total_batches = len(futures)
            for completed_batches, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                pending_results[result.batch_index] = result
                progress_nodes += result.stats.nodes
                progress_traversals += result.seeds_completed
                elapsed = time.monotonic() - progress_started
                if next_progress_at is not None and progress_traversals >= next_progress_at:
                    avg_nodes = progress_nodes / max(1, progress_traversals)
                    nodes_per_second = progress_nodes / max(1.0e-9, elapsed)
                    logger.info(
                        "Traversal multiprocessing progress: num_workers={} completed_batches={}/{} elapsed_seconds={:.2f} total_nodes={} avg_nodes_per_traversal={:.1f} nodes_per_second={:.1f}",
                        max_workers,
                        completed_batches,
                        total_batches,
                        elapsed,
                        progress_nodes,
                        avg_nodes,
                        nodes_per_second,
                    )
                    while next_progress_at is not None and next_progress_at <= progress_traversals:
                        next_progress_at += progress_every
                while next_batch_index in pending_results:
                    traversals += self._accumulate_parallel_traversal_result(
                        total_stats,
                        pending_results.pop(next_batch_index),
                        hotspot_stats,
                    )
                    next_batch_index += 1
        return total_stats, traversals, hotspot_stats

    def _train_advantage(self, player: int, updates: int) -> float:
        memory = self.advantage_memories[player]
        if len(memory) == 0 or updates <= 0:
            return 0.0
        net = self.advantage_nets[player]
        optimizer = self.advantage_optimizers[player]
        net.train()
        losses: list[float] = []
        for _ in range(updates):
            batch = memory.sample(self.config.optimization.advantage_batch_size, self.rng)
            x = torch.as_tensor(batch["info_state"], dtype=torch.float32, device=self.device)
            target = torch.as_tensor(batch["target"], dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(batch["legal_mask"], dtype=torch.bool, device=self.device)
            pred = net(x)
            diff = (pred - target).masked_fill(~mask, 0.0)
            loss = (diff.square().sum() / mask.sum().clamp_min(1)).to(torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.optimization.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.config.optimization.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def _train_strategy(self, updates: int) -> float:
        if len(self.strategy_memory) == 0 or updates <= 0:
            return 0.0
        self.strategy_net.train()
        losses: list[float] = []
        for _ in range(updates):
            batch = self.strategy_memory.sample(self.config.optimization.strategy_batch_size, self.rng)
            x = torch.as_tensor(batch["info_state"], dtype=torch.float32, device=self.device)
            target = torch.as_tensor(batch["target"], dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(batch["legal_mask"], dtype=torch.bool, device=self.device)
            logits = self.strategy_net(x).masked_fill(~mask, torch.finfo(torch.float32).min)
            log_probs = F.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
            loss = -(target * log_probs).sum(dim=-1).mean()
            self.strategy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.optimization.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), self.config.optimization.grad_clip)
            self.strategy_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def _append_metrics(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    def _write_progress(self, metrics: dict[str, Any]) -> None:
        self.progress_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    def save_checkpoint(self, path: Path, metrics: dict[str, Any] | None = None) -> None:
        payload = {
            "config": self.config.to_dict(),
            "lost_cities_config": self.lc_config.to_snapshot(),
            "resume_semantics": "networks_optimizers_iteration_only",
            "iteration": self.iteration,
            "input_dim": self.input_dim,
            "action_size": self.action_size,
            "advantage_nets": [net.state_dict() for net in self.advantage_nets],
            "strategy_net": self.strategy_net.state_dict(),
            "advantage_optimizers": [optimizer.state_dict() for optimizer in self.advantage_optimizers],
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "metrics": metrics or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.iteration = int(payload.get("iteration", 0))
        logger.warning(
            "Resuming from {} restores networks, optimizers, and iteration only; reservoir memories and RNG state are not restored",
            path,
        )
        for net, state_dict in zip(self.advantage_nets, payload["advantage_nets"], strict=True):
            net.load_state_dict(state_dict)
        self.strategy_net.load_state_dict(payload["strategy_net"])
        if "advantage_optimizers" in payload:
            for optimizer, state_dict in zip(self.advantage_optimizers, payload["advantage_optimizers"], strict=True):
                optimizer.load_state_dict(state_dict)
        if "strategy_optimizer" in payload:
            self.strategy_optimizer.load_state_dict(payload["strategy_optimizer"])
