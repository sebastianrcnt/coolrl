from __future__ import annotations

from pathlib import Path

from coolrl.lost_cities.deep_cfr.config import RulesConfig, load_config


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
    assert cfg.traversal.progress_every_traversals == 10
    assert cfg.traversal.num_workers == 0
    assert cfg.traversal.traversal_worker_chunk_size == 1
    assert cfg.traversal.profile_hotspots is False
    assert cfg.checkpoint.save_latest_only is False


def test_load_probe_yaml_profile() -> None:
    cfg = load_config(Path("configs/lost_cities_deep_cfr_probe.yaml"))
    assert cfg.rules.tier == "tier3"
    assert cfg.traversal.store_strategy_on_opponent_nodes is True
    assert cfg.traversal.store_strategy_on_traverser_nodes is True
    assert cfg.traversal.max_nodes_per_traversal == 5_000
    assert cfg.traversal.progress_every_traversals == 1
    assert cfg.traversal.num_workers == 0
    assert cfg.traversal.traversal_worker_chunk_size == 1
    assert cfg.traversal.profile_hotspots is False
    assert cfg.checkpoint.save_latest_only is False


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
