from coolrl.lost_cities.game import LostCitiesConfig
from coolrl.lost_cities.gui_models import available_model_names, build_model
from coolrl.lost_cities.pygame_common import build_lost_cities_backend


def test_random_gui_model_selects_unified_legal_action() -> None:
    backend = build_lost_cities_backend("python", LostCitiesConfig(seed=3), seed=3)
    snapshot = backend.snapshot()
    model = build_model("random", seed=11)

    action = model.act(snapshot)

    assert action < len(snapshot.legal_mask)
    assert snapshot.legal_mask[action]


def test_model_registry_lists_random_model() -> None:
    assert "random" in available_model_names()
