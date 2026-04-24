from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from coolrl.lost_cities.deep_cfr.benchmark import benchmark_traversal_modes
from coolrl.lost_cities.deep_cfr.config import config_from_dict
from coolrl.lost_cities.deep_cfr.encoding import infer_input_dim
from coolrl.lost_cities.deep_cfr.evaluate import (
    StrategyNetBot,
    evaluate_against_bot,
    load_strategy_bot_from_checkpoint,
)
from coolrl.lost_cities.deep_cfr.memory import _Sample
from coolrl.lost_cities.deep_cfr.trainer import DeepCFRTrainer
from coolrl.lost_cities.deep_cfr.traversal import TraversalStats
from coolrl.lost_cities.deep_cfr.traversal_worker import TraversalWorkerBatchResult
from coolrl.lost_cities.game import GameState


def test_tiny_training_run_completes(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 1,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {"traversals_per_player": 2, "max_depth": 4},
            "optimization": {
                "advantage_batch_size": 8,
                "strategy_batch_size": 8,
                "advantage_updates_per_iteration": 1,
                "strategy_updates_per_iteration": 1,
            },
            "memory": {"advantage_capacity": 100, "strategy_capacity": 100},
            "evaluation": {"eval_every": 0, "games": 2},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.run()
    checkpoint_path = tmp_path / "latest.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["resume_semantics"] == "networks_optimizers_iteration_only"


def test_tiny_training_run_completes_with_parallel_traversal(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 1,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 2,
                "max_depth": 2,
                "max_nodes_per_traversal": 32,
                "num_workers": 2,
                "traversal_worker_chunk_size": 4,
            },
            "optimization": {
                "advantage_batch_size": 8,
                "strategy_batch_size": 8,
                "advantage_updates_per_iteration": 1,
                "strategy_updates_per_iteration": 1,
            },
            "memory": {"advantage_capacity": 100, "strategy_capacity": 100},
            "evaluation": {"eval_every": 0, "games": 2},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.run()
    checkpoint_path = tmp_path / "latest.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["resume_semantics"] == "networks_optimizers_iteration_only"
    assert len(trainer.advantage_memories[0]) > 0 or len(trainer.advantage_memories[1]) > 0
    assert len(trainer.strategy_memory) > 0


def test_tiny_training_run_profiles_hotspots_and_can_save_latest_only(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 1,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 2,
                "max_depth": 2,
                "progress_every_traversals": 0,
                "profile_hotspots": True,
            },
            "optimization": {
                "advantage_batch_size": 8,
                "strategy_batch_size": 8,
                "advantage_updates_per_iteration": 1,
                "strategy_updates_per_iteration": 1,
            },
            "memory": {"advantage_capacity": 100, "strategy_capacity": 100},
            "evaluation": {"eval_every": 0, "games": 2},
            "checkpoint": {"directory": str(tmp_path), "save_latest_only": True},
        }
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.run()

    assert (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "iteration_00001.pt").exists()

    progress = json.loads((tmp_path / "runtime_progress.json").read_text(encoding="utf-8"))
    assert progress["traversal_profile_wall_seconds"] >= 0.0
    assert progress["traversal_profile_encode_seconds"] >= 0.0
    assert progress["traversal_profile_forward_seconds"] >= 0.0
    assert progress["traversal_profile_regret_matching_seconds"] >= 0.0
    assert progress["traversal_profile_clone_apply_seconds"] >= 0.0
    assert progress["traversal_profile_memory_add_seconds"] >= 0.0


def test_traversal_benchmark_compares_single_and_multiprocessing(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 0,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 1,
                "max_depth": 2,
                "max_nodes_per_traversal": 32,
                "num_workers": 2,
                "traversal_worker_chunk_size": 4,
            },
            "optimization": {
                "advantage_updates_per_iteration": 0,
                "strategy_updates_per_iteration": 0,
            },
            "evaluation": {"eval_every": 0, "games": 1},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )

    result = benchmark_traversal_modes(cfg, mp_workers=2, iteration=1)

    assert result["device_used"] == "cpu"
    assert result["single_process"]["num_workers"] == 0
    assert result["multiprocessing"]["num_workers"] == 2
    assert result["single_process"]["traversal_seconds"] >= 0.0
    assert result["multiprocessing"]["traversal_seconds"] >= 0.0
    assert result["speedup_vs_single_process"] >= 0.0


def test_parallel_traversal_result_merge_aggregates_stats(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 0,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "memory": {"advantage_capacity": 10, "strategy_capacity": 10},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)
    input_dim = infer_input_dim(trainer.lc_config)
    legal_mask = np.zeros(trainer.action_size, dtype=bool)
    legal_mask[0] = True

    def make_sample(player: int, iteration: int, value: float) -> _Sample:
        target = np.zeros(trainer.action_size, dtype=np.float32)
        target[0] = value
        return _Sample(
            info_state=np.full(input_dim, value, dtype=np.float32),
            target=target,
            legal_mask=legal_mask.copy(),
            player=player,
            iteration=iteration,
        )

    first_result = TraversalWorkerBatchResult(
        batch_index=1,
        traverser=0,
        seeds_completed=1,
        stats=TraversalStats(nodes=5, terminals=1, cutoffs=2, node_limit_cutoffs=0, max_depth_reached=3),
        advantage_samples=[make_sample(0, 1, 1.0)],
        strategy_samples=[make_sample(0, 1, 2.0)],
    )
    second_result = TraversalWorkerBatchResult(
        batch_index=0,
        traverser=1,
        seeds_completed=2,
        stats=TraversalStats(nodes=7, terminals=0, cutoffs=0, node_limit_cutoffs=4, max_depth_reached=2),
        advantage_samples=[make_sample(1, 1, 3.0)],
        strategy_samples=[make_sample(1, 1, 4.0), make_sample(0, 1, 5.0)],
    )

    total_stats, traversals = trainer._merge_parallel_traversal_results([first_result, second_result])

    assert traversals == 3
    assert total_stats.nodes == 12
    assert total_stats.terminals == 1
    assert total_stats.cutoffs == 2
    assert total_stats.node_limit_cutoffs == 4
    assert total_stats.max_depth_reached == 3
    assert len(trainer.advantage_memories[0]) == 1
    assert len(trainer.advantage_memories[1]) == 1
    assert len(trainer.strategy_memory) == 3

    second_result.advantage_samples[0].target[0] = 99.0
    assert trainer.advantage_memories[1].samples[0].target[0] == 3.0


def test_strategy_net_bot_returns_legal_phase_local_action(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 0,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)
    state = GameState.new_game(trainer.lc_config)
    bot = StrategyNetBot(trainer.strategy_net, trainer.lc_config, device="cpu")
    action = bot.act(state)
    assert state.legal_mask()[action]


def test_load_strategy_bot_from_checkpoint_returns_legal_action(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 1,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {"traversals_per_player": 2, "max_depth": 2},
            "optimization": {
                "advantage_batch_size": 8,
                "strategy_batch_size": 8,
                "advantage_updates_per_iteration": 1,
                "strategy_updates_per_iteration": 1,
            },
            "memory": {"advantage_capacity": 100, "strategy_capacity": 100},
            "evaluation": {"eval_every": 0, "games": 1},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)
    trainer.run()

    bot, lc_config = load_strategy_bot_from_checkpoint(tmp_path / "latest.pt", device="cpu", sample=False, seed=7)
    state = GameState.new_game(lc_config, seed=11)
    action = bot.act(state)

    assert state.legal_mask()[action]


def test_evaluate_against_bot_handles_max_steps_timeout() -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 0,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "evaluation": {"max_steps": 1, "on_max_steps": "draw"},
        }
    )
    trainer = DeepCFRTrainer(cfg)

    result = evaluate_against_bot(
        trainer.strategy_net,
        StrategyNetBot(trainer.strategy_net, trainer.lc_config, device="cpu"),
        trainer.lc_config,
        games=2,
        seed=123,
        device="cpu",
        max_steps=1,
        on_max_steps="draw",
    )

    assert result["games"] == 2
    assert result["max_step_timeouts"] == 2
    assert result["draws"] == 2
    assert result["avg_diff"] == 0.0


def test_trainer_evaluation_timeout_does_not_crash_run(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 1,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {"traversals_per_player": 2, "max_depth": 2},
            "optimization": {
                "advantage_batch_size": 8,
                "strategy_batch_size": 8,
                "advantage_updates_per_iteration": 1,
                "strategy_updates_per_iteration": 1,
            },
            "memory": {"advantage_capacity": 100, "strategy_capacity": 100},
            "evaluation": {
                "eval_every": 1,
                "games": 1,
                "opponents": ["random"],
                "max_steps": 1,
                "on_max_steps": "draw",
            },
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)

    trainer.run()

    assert (tmp_path / "latest.pt").exists()
    progress = json.loads((tmp_path / "runtime_progress.json").read_text(encoding="utf-8"))
    assert progress["eval_random_max_step_timeouts"] == 1
    assert progress["eval_random_win_rate"] == 0.0
    assert progress["eval_random_avg_diff"] == 0.0
