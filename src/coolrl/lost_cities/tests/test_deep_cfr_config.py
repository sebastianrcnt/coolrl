from __future__ import annotations

import sys
import types
from pathlib import Path

from coolrl.lost_cities.deep_cfr.config import RulesConfig, config_from_dict, load_config


def test_load_default_config_is_tier3() -> None:
    cfg = load_config(None)
    assert cfg.rules.tier == "tier3"
    assert cfg.rules.to_lost_cities_config().n_colors == 5


def test_load_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_tier3.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_tier3"
    assert cfg.traversal.backend == "python"
    assert cfg.traversal.store_strategy_on_opponent_nodes is True
    assert cfg.traversal.store_strategy_on_traverser_nodes is True
    assert cfg.traversal.max_nodes_per_traversal == 10_000
    assert cfg.traversal.max_depth == 16
    assert cfg.traversal.cutoff_value_mode == "score_diff"
    assert cfg.traversal.cutoff_rollouts == 0
    assert cfg.traversal.cutoff_rollout_policy == "random"
    assert cfg.traversal.cutoff_rollout_max_steps == 10_000
    assert cfg.traversal.progress_every_traversals == 10
    assert cfg.traversal.num_workers == 0
    assert cfg.traversal.traversal_worker_chunk_size == 4
    assert cfg.traversal.profile_hotspots is False
    assert cfg.traversal.outcome_sampling_epsilon == 0.05
    assert cfg.evaluation.max_steps == 10_000
    assert cfg.evaluation.on_max_steps == "score_diff"
    assert cfg.checkpoint.save_latest_only is False


def test_load_probe_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_probe.yaml"))
    assert cfg.rules.tier == "tier3"
    assert cfg.traversal.store_strategy_on_opponent_nodes is True
    assert cfg.traversal.store_strategy_on_traverser_nodes is True
    assert cfg.traversal.max_nodes_per_traversal == 5_000
    assert cfg.traversal.progress_every_traversals == 1
    assert cfg.traversal.num_workers == 0
    assert cfg.traversal.traversal_worker_chunk_size == 4
    assert cfg.traversal.profile_hotspots is False
    assert cfg.evaluation.max_steps == 10_000
    assert cfg.evaluation.on_max_steps == "score_diff"
    assert cfg.checkpoint.save_latest_only is False


def test_load_capped_rollout300_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_capped_rollout300.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_capped_rollout300"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollouts == 1
    assert cfg.traversal.cutoff_rollout_policy == "random"
    assert cfg.traversal.cutoff_rollout_max_steps == 300
    assert cfg.traversal.max_depth == 16
    assert cfg.traversal.num_workers == 8
    assert cfg.traversal.traversal_worker_chunk_size == 4
    assert cfg.traversal.outcome_sampling_epsilon == 0.05
    assert cfg.evaluation.eval_every == 10
    assert cfg.evaluation.games == 50
    assert cfg.evaluation.max_steps == 1000
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_capped_rollout300"


def test_load_safe_rollout300_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_rollout300.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_rollout300"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollouts == 1
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.cutoff_rollout_max_steps == 300
    assert cfg.traversal.max_depth == 16
    assert cfg.traversal.num_workers == 8
    assert cfg.traversal.traversal_worker_chunk_size == 4
    assert cfg.evaluation.eval_every == 10
    assert cfg.evaluation.games == 50
    assert cfg.evaluation.max_steps == 1000
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_rollout300"


def test_load_safe_rollout300_eps05_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_rollout300_eps05.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_rollout300_eps05"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.cutoff_rollout_max_steps == 300
    assert cfg.traversal.outcome_sampling_epsilon == 0.5
    assert cfg.traversal.traversals_per_player == 100
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_rollout300_eps05"


def test_load_safe_rollout300_safe_opponent_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_rollout300_safe_opponent.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_rollout300_safe_opponent"
    assert cfg.traversal.opponent_policy == "safe_heuristic"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.outcome_sampling_epsilon == 0.05
    assert cfg.traversal.traversals_per_player == 100
    assert cfg.traversal.outcome_sampling_value_clip is None
    assert cfg.traversal.outcome_unsampled_regret == "negative_node_value"
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_rollout300_safe_opponent"


def test_load_safe_rollout300_t500_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_rollout300_t500.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_rollout300_t500"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.cutoff_rollout_max_steps == 300
    assert cfg.traversal.outcome_sampling_epsilon == 0.05
    assert cfg.traversal.traversals_per_player == 500
    assert cfg.traversal.progress_every_traversals == 50
    assert cfg.memory.advantage_capacity == 400000
    assert cfg.memory.strategy_capacity == 800000
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_rollout300_t500"


def test_load_safe_rollout300_clip500_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_rollout300_clip500.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_rollout300_clip500"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.cutoff_rollout_max_steps == 300
    assert cfg.traversal.outcome_sampling_epsilon == 0.05
    assert cfg.traversal.outcome_sampling_value_clip == 500.0
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_rollout300_clip500"


def test_load_safe_rollout300_eps02_t500_clip500_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_rollout300_eps02_t500_clip500.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_rollout300_eps02_t500_clip500"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.cutoff_rollout_max_steps == 300
    assert cfg.traversal.outcome_sampling_epsilon == 0.2
    assert cfg.traversal.outcome_sampling_value_clip == 500.0
    assert cfg.traversal.traversals_per_player == 500
    assert cfg.traversal.traversal_worker_chunk_size == 8
    assert cfg.optimization.advantage_updates_per_iteration == 512
    assert cfg.optimization.strategy_updates_per_iteration == 512
    assert cfg.memory.advantage_capacity == 2_000_000
    assert cfg.memory.strategy_capacity == 2_000_000
    assert cfg.evaluation.games == 100
    assert cfg.checkpoint.directory == (
        "checkpoints/lost_cities_deep_cfr_safe_rollout300_eps02_t500_clip500"
    )


def test_load_safe_rollout300_zero_unsampled_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_rollout300_zero_unsampled.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_rollout300_zero_unsampled"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.cutoff_rollout_max_steps == 300
    assert cfg.traversal.outcome_sampling_epsilon == 0.2
    assert cfg.traversal.outcome_sampling_value_clip == 500.0
    assert cfg.traversal.outcome_unsampled_regret == "zero"
    assert cfg.traversal.traversals_per_player == 500
    assert cfg.traversal.traversal_worker_chunk_size == 8
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_rollout300_zero_unsampled"


def test_load_safe_br_zero_unsampled_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_br_zero_unsampled.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_br_zero_unsampled"
    assert cfg.traversal.opponent_policy == "safe_heuristic"
    assert cfg.traversal.store_strategy_on_opponent_nodes is False
    assert cfg.traversal.store_strategy_on_traverser_nodes is True
    assert cfg.traversal.cutoff_rollout_policy == "safe_heuristic"
    assert cfg.traversal.outcome_sampling_epsilon == 0.2
    assert cfg.traversal.outcome_sampling_value_clip == 500.0
    assert cfg.traversal.outcome_unsampled_regret == "zero"
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_br_zero_unsampled"


def test_load_safe_br_pretrained_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_br_pretrained.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_br_pretrained"
    assert cfg.traversal.opponent_policy == "safe_heuristic"
    assert cfg.traversal.store_strategy_on_opponent_nodes is False
    assert cfg.traversal.outcome_sampling_epsilon == 0.2
    assert cfg.traversal.outcome_sampling_value_clip == 500.0
    assert cfg.traversal.outcome_unsampled_regret == "zero"
    assert cfg.optimization.learning_rate == 1.0e-4
    assert cfg.evaluation.eval_every == 5
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_br_pretrained"


def test_load_safe_pretrain_512_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_pretrain_512.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_pretrain_512"
    assert cfg.network.hidden_size == 512
    assert cfg.network.num_layers == 4
    assert cfg.traversal.opponent_policy == "safe_heuristic"
    assert cfg.traversal.outcome_unsampled_regret == "zero"
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_pretrain_512"


def test_load_safe_dagger_512_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_dagger_512.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_dagger_512"
    assert cfg.network.hidden_size == 512
    assert cfg.network.num_layers == 4
    assert cfg.optimization.learning_rate == 1.0e-4
    assert cfg.evaluation.opponents == [
        "random",
        "safe_heuristic",
        "safe_heuristic_loose",
        "safe_heuristic_strict",
        "noisy_safe",
        "passive_discard",
    ]
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_dagger_512"


def test_load_safe_dagger_256_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_safe_dagger_256.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_safe_dagger_256"
    assert cfg.network.hidden_size == 256
    assert cfg.network.num_layers == 3
    assert cfg.optimization.learning_rate == 3.0e-5
    assert "safe_heuristic_loose" in cfg.evaluation.opponents
    assert "safe_heuristic_strict" in cfg.evaluation.opponents
    assert "noisy_safe" in cfg.evaluation.opponents
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_safe_dagger_256"


def test_load_pure_self_play_a_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_pure_self_play_a.yaml"))

    assert cfg.experiment_name == "lost_cities_deep_cfr_pure_self_play_a"
    assert cfg.traversal.opponent_policy == "self_play_league"
    assert cfg.traversal.cutoff_value_mode == "score_diff"
    assert cfg.traversal.cutoff_rollouts == 0
    assert cfg.traversal.self_play_league.current_weight == 0.5
    assert cfg.traversal.self_play_league.recent_weight == 0.3
    assert cfg.traversal.self_play_league.older_weight == 0.2
    assert cfg.traversal.self_play_league.recent_window == 5
    assert cfg.traversal.self_play_league.max_snapshots == 20
    assert cfg.traversal.store_strategy_on_opponent_nodes is False
    assert cfg.evaluation.opponents == [
        "random",
        "passive_discard",
        "safe_heuristic",
        "safe_heuristic_loose",
        "safe_heuristic_strict",
        "noisy_safe",
    ]


def test_load_pure_self_play_b_yaml_profile_matches_safe_adv_imitation_shape() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_pure_self_play_b.yaml"))

    assert cfg.experiment_name == "lost_cities_deep_cfr_pure_self_play_b"
    assert cfg.network.hidden_size == 256
    assert cfg.network.num_layers == 3
    assert cfg.traversal.opponent_policy == "self_play_league"
    assert cfg.checkpoint.directory == "checkpoints/lost_cities_deep_cfr_pure_self_play_b"


def test_default_cutoff_config_preserves_score_diff_behavior() -> None:
    cfg = config_from_dict({})
    assert cfg.traversal.cutoff_value_mode == "score_diff"
    assert cfg.traversal.cutoff_rollouts == 0
    assert cfg.traversal.cutoff_rollout_policy == "random"
    assert cfg.traversal.opponent_policy == "network"
    assert cfg.traversal.self_play_league.current_weight == 0.5
    assert cfg.traversal.max_depth == 8
    assert cfg.traversal.outcome_sampling_epsilon == 0.0
    assert cfg.traversal.outcome_sampling_value_clip is None
    assert cfg.traversal.outcome_unsampled_regret == "negative_node_value"


def test_rules_config_default_tier3_shape() -> None:
    lc_cfg = RulesConfig().to_lost_cities_config()
    assert lc_cfg.n_colors == 5
    assert lc_cfg.n_ranks == 9
    assert lc_cfg.n_handshakes == 3
    assert lc_cfg.hand_size == 8


def test_rules_config_overrides() -> None:
    lc_cfg = RulesConfig(bonus_threshold=7, expedition_penalty=-15).to_lost_cities_config()
    assert lc_cfg.bonus_threshold == 7
    assert lc_cfg.expedition_penalty == -15


def test_traversal_num_workers_explicit_values_remain_unchanged() -> None:
    cfg = config_from_dict(
        {
            "traversal": {
                "num_workers": 4,
                "traversals_per_player": 20,
                "traversal_worker_chunk_size": 4,
            }
        }
    )

    resolved, is_auto, cpu_guess, num_batches = cfg.traversal.resolved_num_workers_for_traversal()

    assert resolved == 4
    assert is_auto is False
    assert cpu_guess is None
    assert num_batches is None


def test_traversal_num_workers_auto_is_capped_by_batch_count(monkeypatch) -> None:
    monkeypatch.setattr("coolrl.lost_cities.deep_cfr.config.os.cpu_count", lambda: 32)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda logical=False: 16))
    cfg = config_from_dict(
        {
            "traversal": {
                "num_workers": "auto",
                "traversals_per_player": 3,
                "traversal_worker_chunk_size": 2,
            }
        }
    )

    resolved, is_auto, cpu_guess, num_batches = cfg.traversal.resolved_num_workers_for_traversal()

    assert resolved == 4
    assert is_auto is True
    assert cpu_guess == 16
    assert num_batches == 4


def test_traversal_num_workers_auto_never_returns_below_one(monkeypatch) -> None:
    monkeypatch.setattr("coolrl.lost_cities.deep_cfr.config.os.cpu_count", lambda: 1)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda logical=False: None))
    cfg = config_from_dict(
        {
            "traversal": {
                "num_workers": "auto",
                "traversals_per_player": 0,
                "traversal_worker_chunk_size": 4,
            }
        }
    )

    resolved, is_auto, cpu_guess, num_batches = cfg.traversal.resolved_num_workers_for_traversal()

    assert resolved == 1
    assert is_auto is True
    assert cpu_guess == 1
    assert num_batches == 0


def test_traversal_num_workers_auto_falls_back_without_psutil(monkeypatch) -> None:
    monkeypatch.setattr("coolrl.lost_cities.deep_cfr.config.os.cpu_count", lambda: 12)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    cfg = config_from_dict(
        {
            "traversal": {
                "num_workers": "auto",
                "traversals_per_player": 20,
                "traversal_worker_chunk_size": 4,
            }
        }
    )

    resolved, is_auto, cpu_guess, num_batches = cfg.traversal.resolved_num_workers_for_traversal()

    assert resolved == 6
    assert is_auto is True
    assert cpu_guess == 6
    assert num_batches == 10


def test_traversal_num_workers_auto_prefers_physical_core_count(monkeypatch) -> None:
    monkeypatch.setattr("coolrl.lost_cities.deep_cfr.config.os.cpu_count", lambda: 32)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda logical=False: 14))
    cfg = config_from_dict(
        {
            "traversal": {
                "num_workers": "auto",
                "traversals_per_player": 20,
                "traversal_worker_chunk_size": 4,
            }
        }
    )

    resolved, is_auto, cpu_guess, num_batches = cfg.traversal.resolved_num_workers_for_traversal()

    assert resolved == 10
    assert is_auto is True
    assert cpu_guess == 14
    assert num_batches == 10
