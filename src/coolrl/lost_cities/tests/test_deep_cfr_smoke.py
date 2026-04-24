from __future__ import annotations

from pathlib import Path

from coolrl.lost_cities.deep_cfr.config import config_from_dict
from coolrl.lost_cities.deep_cfr.evaluate import StrategyNetBot
from coolrl.lost_cities.deep_cfr.trainer import DeepCFRTrainer
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
    assert (tmp_path / "latest.pt").exists()


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
