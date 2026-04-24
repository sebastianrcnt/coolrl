from coolrl.lost_cities.game import LostCitiesConfig
from coolrl.lost_cities.backends import build_lost_cities_backend
from coolrl.lost_cities.bots import available_bot_names, build_bot


def test_random_gui_bot_selects_unified_legal_action() -> None:
    backend = build_lost_cities_backend("python", LostCitiesConfig(seed=3), seed=3)
    snapshot = backend.snapshot()
    bot = build_bot("random", seed=11)

    action = bot.act(snapshot)

    assert action < len(snapshot.legal_mask)
    assert snapshot.legal_mask[action]


def test_bot_registry_lists_random_bot() -> None:
    assert "random" in available_bot_names()
