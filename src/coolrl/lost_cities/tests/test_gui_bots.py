from coolrl.lost_cities.game import LostCitiesConfig
from coolrl.lost_cities.backends import build_lost_cities_backend
from coolrl.lost_cities.bots import SafeHeuristicBot, available_bot_names, build_bot
from coolrl.lost_cities.game import GameState, tier_config
from coolrl.lost_cities.pygame_pvp import snapshot_to_json, undo_until_player_card_phase


def test_random_gui_bot_selects_unified_legal_action() -> None:
    backend = build_lost_cities_backend("python", LostCitiesConfig(seed=3), seed=3)
    snapshot = backend.snapshot()
    bot = build_bot("random", seed=11)

    action = bot.act(snapshot)

    assert action < len(snapshot.legal_mask)
    assert snapshot.legal_mask[action]


def test_bot_registry_lists_random_bot() -> None:
    assert "random" in available_bot_names()
    assert "safe-heuristic" in available_bot_names()


def test_bot_registry_builds_safe_heuristic_bot() -> None:
    assert isinstance(build_bot("safe-heuristic"), SafeHeuristicBot)


def test_safe_heuristic_game_state_path_differs_from_snapshot_fallback() -> None:
    backend = build_lost_cities_backend("python", tier_config("tier3"), seed=2)
    bot = SafeHeuristicBot()

    while backend.snapshot().current_player != 1:
        human_action = next(
            i for i, legal in enumerate(backend.snapshot().legal_mask) if legal
        )
        backend.apply(human_action)
    snapshot = backend.snapshot()

    fallback_action = bot.act(snapshot)
    state_input = GameState.from_snapshot(snapshot_to_json(snapshot))
    heuristic_action = state_input.to_unified_action(bot.act(state_input))

    assert fallback_action == 0
    assert heuristic_action == 8


def test_pvc_undo_rewinds_human_turn_when_bot_turn_starts() -> None:
    backend = build_lost_cities_backend("python", LostCitiesConfig(seed=3), seed=3)
    initial = backend.snapshot()

    for _ in range(2):
        snapshot = backend.snapshot()
        action = next(i for i, legal in enumerate(snapshot.legal_mask) if legal)
        backend.apply(action)

    assert backend.snapshot().current_player == 1
    assert backend.snapshot().phase == "card"

    undo_count = undo_until_player_card_phase(backend, player=0)

    assert undo_count == 2
    assert backend.snapshot() == initial


def test_pvc_undo_rewinds_full_human_and_bot_cycle() -> None:
    backend = build_lost_cities_backend("python", LostCitiesConfig(seed=3), seed=3)
    initial = backend.snapshot()

    for _ in range(4):
        snapshot = backend.snapshot()
        action = next(i for i, legal in enumerate(snapshot.legal_mask) if legal)
        backend.apply(action)

    assert backend.snapshot().current_player == 0
    assert backend.snapshot().phase == "card"

    undo_count = undo_until_player_card_phase(backend, player=0)

    assert undo_count == 4
    assert backend.snapshot() == initial
