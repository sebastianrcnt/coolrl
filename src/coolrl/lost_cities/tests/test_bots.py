from coolrl.lost_cities.bots import (
    LostCitiesBot,
    RandomBot,
    SafeHeuristicBot,
    play_game,
)
from coolrl.lost_cities.game import tier_config


def test_builtin_bots_implement_lost_cities_bot() -> None:
    assert isinstance(RandomBot(1), LostCitiesBot)
    assert isinstance(SafeHeuristicBot(), LostCitiesBot)


def test_safe_heuristic_mirror_match_finishes() -> None:
    state = play_game(
        SafeHeuristicBot(),
        SafeHeuristicBot(),
        tier_config("tier1"),
        seed=2000,
        max_steps=200,
    )
    assert state.terminal is True
