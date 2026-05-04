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
    assert cfg.traversal.max_depth is None
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


def test_load_cutoff_random_rollout_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_cutoff_random_rollout.yaml"))
    assert cfg.experiment_name == "lost_cities_deep_cfr_cutoff_random_rollout"
    assert cfg.traversal.cutoff_value_mode == "random_rollout"
    assert cfg.traversal.cutoff_rollouts == 1
    assert cfg.traversal.cutoff_rollout_policy == "random"
    assert cfg.traversal.cutoff_rollout_max_steps == 10_000
    assert cfg.traversal.max_depth == 12
    assert cfg.traversal.max_nodes_per_traversal == 10_000


def test_default_cutoff_config_preserves_score_diff_behavior() -> None:
    cfg = config_from_dict({})
    assert cfg.traversal.cutoff_value_mode == "score_diff"
    assert cfg.traversal.cutoff_rollouts == 0
    assert cfg.traversal.cutoff_rollout_policy == "random"
    assert cfg.traversal.max_depth is None
    assert cfg.traversal.outcome_sampling_epsilon == 0.0


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
