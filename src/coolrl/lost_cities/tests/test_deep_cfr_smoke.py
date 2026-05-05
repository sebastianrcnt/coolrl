from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

import coolrl.lost_cities.deep_cfr.benchmark as benchmark_module
import coolrl.lost_cities.deep_cfr.trainer as trainer_module
from coolrl.lost_cities.deep_cfr.benchmark import benchmark_traversal_modes
from coolrl.lost_cities.deep_cfr.config import config_from_dict
from coolrl.lost_cities.deep_cfr.encoding import infer_input_dim
from coolrl.lost_cities.deep_cfr.evaluate import (
    StrategyNetBot,
    evaluate_against_bot,
    load_strategy_bot_from_checkpoint,
    make_opponent,
)
from coolrl.lost_cities.deep_cfr.imitation import pretrain_safe_heuristic_checkpoint
from coolrl.lost_cities.deep_cfr.memory import _Sample
from coolrl.lost_cities.deep_cfr.cli import _RESUME_LATEST, _resolve_resume_path, build_parser
from coolrl.lost_cities.deep_cfr.trainer import DeepCFRTrainer
from coolrl.lost_cities.deep_cfr.traversal import TraversalStats
from coolrl.lost_cities.deep_cfr.traversal_worker import TraversalWorkerBatchResult
from coolrl.lost_cities.game import GameState
from coolrl.lost_cities.bots import PassiveDiscardBot


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


def test_train_parser_resume_without_value_uses_latest_sentinel() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "configs/lost_cities_deep_cfr_tier3.yaml", "--resume"])

    assert args.resume == _RESUME_LATEST


def test_train_parser_resume_with_path_preserves_path() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--config",
            "configs/lost_cities_deep_cfr_tier3.yaml",
            "--resume",
            "checkpoints/lost_cities_deep_cfr_tier3/iteration_00005.pt",
        ]
    )

    assert args.resume == "checkpoints/lost_cities_deep_cfr_tier3/iteration_00005.pt"


def test_train_parser_accepts_runtime_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--config",
            "configs/lost_cities_deep_cfr_pure_self_play_a.yaml",
            "--init-checkpoint",
            "checkpoints/init.pt",
            "--max-hours",
            "2",
            "--max-iterations",
            "10",
            "--checkpoint-dir",
            "checkpoints/example",
        ]
    )

    assert str(args.init_checkpoint) == "checkpoints/init.pt"
    assert args.max_hours == 2.0
    assert args.max_iterations == 10
    assert str(args.checkpoint_dir) == "checkpoints/example"


def test_pretrain_heuristic_parser_accepts_training_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "pretrain-heuristic",
            "--config",
            "configs/lost_cities_deep_cfr_safe_rollout300_safe_opponent.yaml",
            "--output",
            "checkpoints/example/pretrain.pt",
            "--games",
            "10",
            "--epochs",
            "2",
            "--batch-size",
            "32",
            "--max-steps",
            "100",
            "--seed",
            "123",
            "--learning-rate",
            "1e-5",
            "--dataset-mode",
            "successful_policy_vs_safe",
            "--base-checkpoint",
            "checkpoints/example/base.pt",
            "--init-checkpoint",
            "checkpoints/example/init.pt",
            "--policy-sample",
            "--improvement-rollouts",
            "2",
            "--improvement-rollout-max-steps",
            "50",
            "--improvement-max-examples",
            "100",
            "--improvement-top-k",
            "4",
            "--improvement-progress-every",
            "10",
        ]
    )

    assert str(args.config) == "configs/lost_cities_deep_cfr_safe_rollout300_safe_opponent.yaml"
    assert str(args.output) == "checkpoints/example/pretrain.pt"
    assert args.games == 10
    assert args.epochs == 2
    assert args.batch_size == 32
    assert args.max_steps == 100
    assert args.seed == 123
    assert args.learning_rate == 1.0e-5
    assert args.dataset_mode == "successful_policy_vs_safe"
    assert str(args.base_checkpoint) == "checkpoints/example/base.pt"
    assert str(args.init_checkpoint) == "checkpoints/example/init.pt"
    assert args.policy_sample is True
    assert args.improvement_rollouts == 2
    assert args.improvement_rollout_max_steps == 50
    assert args.improvement_max_examples == 100
    assert args.improvement_top_k == 4
    assert args.improvement_progress_every == 10


def test_fine_tune_policy_parser_accepts_training_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "fine-tune-policy",
            "--config",
            "configs/lost_cities_deep_cfr_safe_dagger_256.yaml",
            "--checkpoint",
            "checkpoints/base.pt",
            "--output",
            "checkpoints/tuned.pt",
            "--games",
            "10",
            "--opponent",
            "safe_heuristic",
            "--max-steps",
            "100",
            "--learning-rate",
            "1e-6",
            "--reward-scale",
            "50",
            "--reward-clip",
            "1.5",
            "--kl-coef",
            "0.1",
            "--entropy-coef",
            "0.002",
            "--grad-clip",
            "0.25",
            "--batch-games",
            "16",
            "--normalize-advantages",
            "--baseline-decay",
            "0.95",
            "--seed",
            "123",
        ]
    )

    assert str(args.config) == "configs/lost_cities_deep_cfr_safe_dagger_256.yaml"
    assert str(args.checkpoint) == "checkpoints/base.pt"
    assert str(args.output) == "checkpoints/tuned.pt"
    assert args.games == 10
    assert args.opponent == "safe_heuristic"
    assert args.max_steps == 100
    assert args.learning_rate == 1.0e-6
    assert args.reward_scale == 50
    assert args.reward_clip == 1.5
    assert args.kl_coef == 0.1
    assert args.entropy_coef == 0.002
    assert args.grad_clip == 0.25
    assert args.batch_games == 16
    assert args.normalize_advantages is True
    assert args.baseline_decay == 0.95
    assert args.seed == 123


def test_resolve_resume_without_path_uses_config_latest(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    latest = checkpoint_dir / "latest.pt"
    latest.write_bytes(b"placeholder")
    cfg = config_from_dict({"checkpoint": {"directory": str(checkpoint_dir)}})

    assert _resolve_resume_path(cfg, _RESUME_LATEST) == str(latest)


def test_resolve_resume_without_path_requires_existing_latest(tmp_path: Path) -> None:
    cfg = config_from_dict({"checkpoint": {"directory": str(tmp_path)}})

    try:
        _resolve_resume_path(cfg, _RESUME_LATEST)
    except FileNotFoundError as exc:
        assert "latest checkpoint does not exist" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_trainer_init_checkpoint_loads_weights_without_iteration_or_optimizers(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "optimization": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
            "checkpoint": {"directory": str(tmp_path / "source")},
        }
    )
    source = DeepCFRTrainer(cfg)
    for parameter in source.strategy_net.parameters():
        parameter.data.fill_(0.25)
    source.iteration = 7
    source.save_checkpoint(tmp_path / "init.pt")

    target_cfg = config_from_dict(
        {
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "optimization": {"learning_rate": 2.0e-4, "weight_decay": 0.0},
            "checkpoint": {"directory": str(tmp_path / "target")},
        }
    )
    target = DeepCFRTrainer(target_cfg, init_checkpoint_path=str(tmp_path / "init.pt"))

    assert target.iteration == 0
    assert target.strategy_optimizer.param_groups[0]["lr"] == 2.0e-4
    for parameter in target.strategy_net.parameters():
        assert torch.all(parameter == torch.full_like(parameter, 0.25))


def test_safe_heuristic_pretrain_checkpoint_is_resume_compatible(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "optimization": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    output_path = tmp_path / "pretrain.pt"

    result = pretrain_safe_heuristic_checkpoint(
        cfg,
        output_path=output_path,
        games=1,
        epochs=1,
        batch_size=16,
        max_steps=200,
        seed=123,
    )

    assert result.output_path == output_path
    assert result.games == 1
    assert result.states > 0
    checkpoint = torch.load(output_path, map_location="cpu")
    assert checkpoint["resume_semantics"] == "networks_optimizers_iteration_only"
    assert checkpoint["iteration"] == 0
    assert checkpoint["metrics"]["pretrain_states"] == result.states
    trainer = DeepCFRTrainer(cfg, resume_path=output_path)
    assert trainer.iteration == 0


def test_aggregated_safe_heuristic_pretrain_checkpoint_uses_base_policy_states(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "optimization": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    base_path = tmp_path / "base.pt"
    output_path = tmp_path / "aggregated.pt"

    pretrain_safe_heuristic_checkpoint(
        cfg,
        output_path=base_path,
        games=1,
        epochs=1,
        batch_size=16,
        max_steps=80,
        seed=123,
    )
    result = pretrain_safe_heuristic_checkpoint(
        cfg,
        output_path=output_path,
        games=3,
        epochs=1,
        batch_size=16,
        max_steps=80,
        seed=456,
        dataset_mode="aggregated",
        base_checkpoint=base_path,
        init_checkpoint=base_path,
    )

    assert result.dataset_mode == "aggregated"
    assert result.states > 0
    checkpoint = torch.load(output_path, map_location="cpu")
    assert checkpoint["metrics"]["pretrain_dataset_mode"] == "aggregated"
    assert checkpoint["metrics"]["pretrain_base_checkpoint"] == str(base_path)
    assert checkpoint["metrics"]["pretrain_init_checkpoint"] == str(base_path)


def test_successful_policy_vs_safe_pretrain_checkpoint_uses_winning_policy_actions(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "optimization": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    base_path = tmp_path / "base.pt"
    output_path = tmp_path / "successful.pt"

    pretrain_safe_heuristic_checkpoint(
        cfg,
        output_path=base_path,
        games=1,
        epochs=1,
        batch_size=16,
        max_steps=80,
        seed=123,
    )
    result = pretrain_safe_heuristic_checkpoint(
        cfg,
        output_path=output_path,
        games=20,
        epochs=1,
        batch_size=16,
        max_steps=80,
        seed=456,
        dataset_mode="successful_policy_vs_safe",
        base_checkpoint=base_path,
        init_checkpoint=base_path,
    )

    assert result.dataset_mode == "successful_policy_vs_safe"
    assert result.states > 0
    checkpoint = torch.load(output_path, map_location="cpu")
    assert checkpoint["metrics"]["pretrain_dataset_mode"] == "successful_policy_vs_safe"


def test_safe_action_rollout_pretrain_checkpoint_uses_rollout_improvement_labels(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "optimization": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    base_path = tmp_path / "base.pt"
    output_path = tmp_path / "safe_action_rollout.pt"

    pretrain_safe_heuristic_checkpoint(
        cfg,
        output_path=base_path,
        games=1,
        epochs=1,
        batch_size=16,
        max_steps=80,
        seed=123,
    )
    result = pretrain_safe_heuristic_checkpoint(
        cfg,
        output_path=output_path,
        games=1,
        epochs=1,
        batch_size=16,
        max_steps=40,
        seed=456,
        dataset_mode="safe_action_rollout",
        base_checkpoint=base_path,
        init_checkpoint=base_path,
        improvement_rollouts=1,
        improvement_rollout_max_steps=20,
        improvement_max_examples=4,
        improvement_top_k=3,
        improvement_progress_every=2,
        learning_rate=1.0e-5,
    )

    assert result.dataset_mode == "safe_action_rollout"
    assert result.states > 0
    checkpoint = torch.load(output_path, map_location="cpu")
    assert checkpoint["metrics"]["pretrain_dataset_mode"] == "safe_action_rollout"
    assert checkpoint["metrics"]["pretrain_improvement_rollouts"] == 1
    assert checkpoint["metrics"]["pretrain_improvement_max_examples"] == 4
    assert checkpoint["metrics"]["pretrain_improvement_top_k"] == 3
    assert checkpoint["metrics"]["pretrain_learning_rate"] == 1.0e-5
    assert "pretrain_safe_action_rollout_disagreement_rate" in checkpoint["metrics"]


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


def test_tiny_parallel_training_run_completes_with_cutoff_rollouts(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 1,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 1,
                "max_depth": 1,
                "max_nodes_per_traversal": 32,
                "cutoff_value_mode": "random_rollout",
                "cutoff_rollouts": 1,
                "cutoff_rollout_max_steps": 50,
                "num_workers": 2,
                "traversal_worker_chunk_size": 1,
                "progress_every_traversals": 0,
            },
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

    progress = json.loads((tmp_path / "runtime_progress.json").read_text(encoding="utf-8"))
    assert progress["cutoff_rollouts"] > 0
    assert progress["cutoff_rollout_steps"] > 0
    assert "cutoff_rollout_max_step_timeouts" in progress


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


def _minimal_benchmark_cfg(tmp_path: Path):
    return config_from_dict(
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


def test_traversal_benchmark_compares_single_and_multiprocessing(tmp_path: Path) -> None:
    cfg = _minimal_benchmark_cfg(tmp_path)
    result = benchmark_traversal_modes(cfg, mp_workers=2, iteration=1, mode="compare")

    assert result["device_used"] == "cpu"
    assert result["single_process"]["requested_workers"] == 0
    assert result["multiprocessing"]["requested_workers"] == 2
    assert result["single_process"]["traversal_seconds"] >= 0.0
    assert result["multiprocessing"]["traversal_seconds"] >= 0.0
    assert isinstance(result["speedup_vs_single_process"], float)
    assert result["speedup_vs_single_process"] >= 0.0


def test_traversal_benchmark_single_mode(tmp_path: Path) -> None:
    cfg = _minimal_benchmark_cfg(tmp_path)
    result = benchmark_traversal_modes(cfg, iteration=1, mode="single")

    assert result["device_used"] == "cpu"
    assert "single_process" in result
    assert "multiprocessing" not in result
    assert result["single_process"]["requested_workers"] == 0
    assert result["single_process"]["traversal_seconds"] >= 0.0
    assert result["speedup_vs_single_process"] == "n/a"


def test_traversal_benchmark_mp_mode(tmp_path: Path) -> None:
    cfg = _minimal_benchmark_cfg(tmp_path)
    result = benchmark_traversal_modes(cfg, mp_workers=2, iteration=1, mode="mp")

    assert result["device_used"] == "cpu"
    assert "multiprocessing" in result
    assert "single_process" not in result
    assert result["multiprocessing"]["requested_workers"] == 2
    assert result["multiprocessing"]["traversal_seconds"] >= 0.0
    assert result["speedup_vs_single_process"] == "n/a"


def test_traversal_benchmark_mp_mode_reports_requested_and_effective_workers(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 0,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 20,
                "max_depth": 2,
                "max_nodes_per_traversal": 32,
                "num_workers": 12,
                "traversal_worker_chunk_size": 4,
                "progress_every_traversals": 0,
            },
            "optimization": {
                "advantage_updates_per_iteration": 0,
                "strategy_updates_per_iteration": 0,
            },
            "evaluation": {"eval_every": 0, "games": 1},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    result = benchmark_traversal_modes(cfg, mp_workers=12, iteration=1, mode="mp")

    assert result["multiprocessing"]["requested_workers"] == 12
    assert result["multiprocessing"]["effective_workers"] == 10
    assert result["multiprocessing"]["num_batches"] == 10


def test_traversal_benchmark_mp_mode_does_not_emit_logging_error(
    tmp_path: Path,
    monkeypatch,
    caplog,
    capsys,
) -> None:
    cfg = _minimal_benchmark_cfg(tmp_path)

    def _fake_run_traversal_benchmark_once(*_args, **_kwargs) -> benchmark_module.TraversalBenchmarkResult:
        return benchmark_module.TraversalBenchmarkResult(
            requested_workers=2,
            effective_workers=2,
            num_batches=2,
            traversal_seconds=0.01,
            total_nodes=10,
            traversals=2,
            nodes_per_second=1000.0,
            avg_nodes_per_traversal=5.0,
            cutoffs=0,
            cutoff_rate=0.0,
            node_limit_cutoffs=0,
            node_limit_cutoff_rate=0.0,
            cutoff_rollouts=0,
            cutoff_rollout_steps=0,
            cutoff_rollout_max_step_timeouts=0,
        )

    monkeypatch.setattr(
        benchmark_module,
        "_run_traversal_benchmark_once",
        _fake_run_traversal_benchmark_once,
    )

    with caplog.at_level(logging.INFO, logger=benchmark_module.__name__):
        benchmark_traversal_modes(cfg, mp_workers=2, iteration=1, mode="mp")

    assert "requested_workers=2" in caplog.text
    assert "--- Logging error ---" not in capsys.readouterr().err


def test_traversal_benchmark_mp_mode_uses_lightweight_batch_count(monkeypatch) -> None:
    class _FakeTrainer:
        def __init__(self, _config) -> None:
            self.num_workers = 12

        def _estimated_traversal_batch_count(self) -> int:
            return 10

        def _frozen_advantage_state_dicts(self):
            raise AssertionError("_frozen_advantage_state_dicts should not be called in benchmark metadata path")

        def _build_traversal_worker_batches(self, _iteration, _advantage_net_state_dicts):
            raise AssertionError("_build_traversal_worker_batches should not be called in benchmark metadata path")

        def _run_traversals_parallel(self, _iteration):
            return TraversalStats(nodes=20), 4, None

        def _run_traversals_single_process(self, _iteration):
            raise AssertionError("single-process path should not be used")

    monkeypatch.setattr(benchmark_module, "DeepCFRTrainer", _FakeTrainer)
    cfg = config_from_dict({"device": "cpu"})
    result = benchmark_module._run_traversal_benchmark_once(cfg, iteration=1)

    assert result.requested_workers == 12
    assert result.effective_workers == 10
    assert result.num_batches == 10
    assert result.traversals == 4


def test_parallel_traversal_logs_warning_when_requested_workers_exceed_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 0,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 1,
                "max_depth": 2,
                "num_workers": 12,
                "traversal_worker_chunk_size": 4,
                "progress_every_traversals": 0,
            },
            "optimization": {
                "advantage_updates_per_iteration": 0,
                "strategy_updates_per_iteration": 0,
            },
            "evaluation": {"eval_every": 0, "games": 1},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)

    warnings: list[str] = []

    def _capture_warning(message: str, *args) -> None:
        warnings.append(message.format(*args))

    class _FakeFuture:
        def __init__(self, result: TraversalWorkerBatchResult) -> None:
            self._result = result

        def result(self) -> TraversalWorkerBatchResult:
            return self._result

    class _FakeExecutor:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, _func, batch):
            return _FakeFuture(
                TraversalWorkerBatchResult(
                    batch_index=batch.batch_index,
                    traverser=batch.traverser,
                    seeds_completed=len(batch.seeds),
                    stats=TraversalStats(nodes=1),
                    advantage_samples=[],
                    strategy_samples=[],
                )
            )

    monkeypatch.setattr(trainer_module.logger, "warning", _capture_warning)
    monkeypatch.setattr(trainer_module, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(trainer_module, "as_completed", lambda futures: futures)

    _, traversals, _ = trainer._run_traversals_parallel(iteration=1)

    assert traversals == 2
    assert any(
        "Requested 12 traversal workers but only 2 batches are available; using 2 workers." in warning
        for warning in warnings
    )


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
        stats=TraversalStats(
            nodes=5,
            terminals=1,
            cutoffs=2,
            node_limit_cutoffs=0,
            max_depth_reached=3,
            cutoff_rollouts=2,
            cutoff_rollout_steps=10,
            cutoff_rollout_max_step_timeouts=1,
        ),
        advantage_samples=[make_sample(0, 1, 1.0)],
        strategy_samples=[make_sample(0, 1, 2.0)],
    )
    second_result = TraversalWorkerBatchResult(
        batch_index=0,
        traverser=1,
        seeds_completed=2,
        stats=TraversalStats(
            nodes=7,
            terminals=0,
            cutoffs=0,
            node_limit_cutoffs=4,
            max_depth_reached=2,
            cutoff_rollouts=4,
            cutoff_rollout_steps=20,
            cutoff_rollout_max_step_timeouts=2,
        ),
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
    assert total_stats.cutoff_rollouts == 6
    assert total_stats.cutoff_rollout_steps == 30
    assert total_stats.cutoff_rollout_max_step_timeouts == 3
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


def test_passive_discard_opponent_acts_legally_and_prefers_deck_draw() -> None:
    bot = make_opponent("passive_discard", seed=7)
    assert isinstance(bot, PassiveDiscardBot)

    state = GameState.new_game()
    card_action = bot.act(state)
    assert state.legal_mask()[card_action]
    assert card_action % 2 == 1
    state.apply_action(card_action)

    draw_action = bot.act(state)
    assert state.legal_mask()[draw_action]
    assert draw_action == 0


def test_variant_safe_opponents_act_legally() -> None:
    for name in ("safe_heuristic_loose", "safe_heuristic_strict", "noisy_safe"):
        bot = make_opponent(name, seed=7)
        state = GameState.new_game()
        action = bot.act(state)

        assert state.legal_mask()[action]


def test_evaluate_against_bot_reports_passive_no_expedition_diagnostics(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 0,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "rules": {"tier": "tier0"},
            "checkpoint": {"directory": str(tmp_path)},
        }
    )
    trainer = DeepCFRTrainer(cfg)
    for parameter in trainer.strategy_net.parameters():
        parameter.data.zero_()
    final_layer = trainer.strategy_net.net[-1]
    final_layer.bias.data.fill_(-10.0)
    for action in range(1, trainer.lc_config.card_action_size, 2):
        final_layer.bias.data[action] = 10.0
    final_layer.bias.data[trainer.lc_config.card_action_size] = 10.0

    result = evaluate_against_bot(
        trainer.strategy_net,
        PassiveDiscardBot(),
        trainer.lc_config,
        games=2,
        seed=123,
        device="cpu",
    )

    assert result["avg_opened_colors"] == 0.0
    assert result["play_action_rate"] == 0.0
    assert result["discard_action_rate"] == 1.0
    assert result["avg_final_score"] == 0.0
    assert result["avg_game_length"] > 0.0
    assert result["policy_entropy"] >= 0.0


def test_deep_cfr_eval_cli_accepts_passive_discard_opponent() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "--checkpoint",
            "checkpoint.pt",
            "--opponent",
            "passive_discard",
        ]
    )

    assert args.opponent == "passive_discard"


def test_deep_cfr_eval_cli_accepts_sampling_and_max_steps_overrides() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "--checkpoint",
            "checkpoint.pt",
            "--sample",
            "--max-steps",
            "123",
        ]
    )

    assert args.sample is True
    assert args.max_steps == 123


def test_deep_cfr_eval_cli_accepts_checkpoint_opponent() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "--checkpoint",
            "checkpoint.pt",
            "--opponent-checkpoint",
            "old.pt",
            "--opponent-sample",
        ]
    )

    assert str(args.opponent_checkpoint) == "old.pt"
    assert args.opponent_sample is True


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
    assert progress["eval_random_avg_final_score"] == 0.0
    assert progress["eval_random_avg_opened_colors"] >= 0.0
    assert progress["eval_random_avg_game_length"] == 1.0
    assert progress["eval_random_policy_entropy"] >= 0.0
    assert progress["eval_random_play_action_rate"] >= 0.0
    assert progress["eval_random_discard_action_rate"] >= 0.0


def test_tiny_self_play_league_training_run_completes(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "max_iterations": 2,
            "device": "cpu",
            "network": {"hidden_size": 16, "num_layers": 1},
            "traversal": {
                "traversals_per_player": 2,
                "max_depth": 3,
                "opponent_policy": "self_play_league",
                "store_strategy_on_opponent_nodes": False,
                "num_workers": 2,
                "traversal_worker_chunk_size": 1,
                "self_play_league": {
                    "current_weight": 0.5,
                    "recent_weight": 0.3,
                    "older_weight": 0.2,
                    "recent_window": 1,
                    "max_snapshots": 2,
                },
            },
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

    progress = json.loads((tmp_path / "runtime_progress.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(tmp_path / "latest.pt", map_location="cpu")
    assert progress["self_play_league_snapshots"] == 2
    assert len(checkpoint["self_play_league_snapshots"]) == 2
