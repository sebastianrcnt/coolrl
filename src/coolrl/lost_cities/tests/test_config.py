from coolrl.lost_cities.bots import RandomBot, run_series
from coolrl.lost_cities.game import GameState, tier_config


def test_all_tiers_create_valid_games() -> None:
    for tier in ["tier0", "tier1", "tier2", "tier3"]:
        config = tier_config(tier, seed=17)
        state = GameState.new_game(config)
        assert len(state.hands[0]) == config.hand_size
        assert len(state.hands[1]) == config.hand_size
        assert len(state.discards) == config.n_colors


def test_random_vs_random_average_is_reasonably_centered() -> None:
    tolerances = {
        "tier0": 15.0,
        "tier1": 40.0,
        "tier2": 80.0,
        "tier3": 140.0,
    }
    for tier, tolerance in tolerances.items():
        config = tier_config(tier)
        result = run_series(RandomBot(1), RandomBot(2), config, games=100, seed=100)
        assert abs(result["avg_diff"]) <= tolerance
