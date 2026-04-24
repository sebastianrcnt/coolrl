from coolrl.lost_cities.bots import LostCitiesBot, RandomBot, SafeHeuristicBot


def test_builtin_bots_implement_lost_cities_bot() -> None:
    assert isinstance(RandomBot(1), LostCitiesBot)
    assert isinstance(SafeHeuristicBot(), LostCitiesBot)
