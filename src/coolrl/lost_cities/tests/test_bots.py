from coolrl.lost_cities.bots import (
    LostCitiesBot,
    RandomBot,
    SafeHeuristicBot,
    play_game,
)
from coolrl.lost_cities.game import Card, GameState, LostCitiesConfig, tier_config
from coolrl.lost_cities.bots.heuristic import draw_from_discard_action


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


def test_safe_heuristic_opponent_value_ignores_hidden_hand() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3)
    bot = SafeHeuristicBot()
    discard_card = Card(color=0, rank=6)

    state_a = GameState.empty(config)
    state_a.expeditions[1][0] = [Card(color=0, rank=0), Card(color=0, rank=4)]
    state_a.discards[0] = [discard_card]
    state_a.hands[1] = [Card(color=0, rank=5)]

    state_b = GameState.empty(config)
    state_b.expeditions[1][0] = [Card(color=0, rank=0), Card(color=0, rank=4)]
    state_b.discards[0] = [discard_card]
    state_b.hands[1] = [Card(color=0, rank=5), Card(color=0, rank=7), Card(color=0, rank=8)]

    value_a = bot._card_value_for_opponent(
        state=state_a,
        opponent=1,
        card=discard_card,
        derived=bot._derived(state_a),
    )
    value_b = bot._card_value_for_opponent(
        state=state_b,
        opponent=1,
        card=discard_card,
        derived=bot._derived(state_b),
    )

    assert value_a == value_b


def test_safe_heuristic_started_expedition_value_ignores_invalid_lower_followup() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3)
    bot = SafeHeuristicBot()
    high_card = Card(color=0, rank=8)

    base_state = GameState.empty(config)
    base_state.expeditions[0][0] = [Card(color=0, rank=4)]
    base_state.hands[0] = [high_card]

    lower_followup_state = GameState.empty(config)
    lower_followup_state.expeditions[0][0] = [Card(color=0, rank=4)]
    lower_followup_state.hands[0] = [Card(color=0, rank=5), high_card]

    base_value = bot._started_expedition_play_value(
        state=base_state,
        player=0,
        card=high_card,
        derived=bot._derived(base_state),
        deck_left=config.deck_size,
    )
    lower_followup_value = bot._started_expedition_play_value(
        state=lower_followup_state,
        player=0,
        card=high_card,
        derived=bot._derived(lower_followup_state),
        deck_left=config.deck_size,
    )

    assert lower_followup_value == base_value


def test_safe_heuristic_draws_playable_discard_instead_of_deck() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=3)
    bot = SafeHeuristicBot()

    state = GameState.empty(config)
    state.current_player = 0
    state.phase = "draw"
    state.expeditions[0][0] = [Card(color=0, rank=4)]
    state.discards[0] = [Card(color=0, rank=6)]
    state.deck = [Card(color=1, rank=8)]

    assert bot._act_draw(state) == draw_from_discard_action(0)


def test_safe_heuristic_can_draw_discard_to_deny_opponent_when_losing() -> None:
    config = LostCitiesConfig(n_colors=2, n_ranks=8, hand_size=4)
    bot = SafeHeuristicBot()

    state = GameState.empty(config)
    state.current_player = 0
    state.phase = "draw"
    state.deck = [Card(color=1, rank=8), Card(color=1, rank=7)]
    state.hands[0] = [Card(color=0, rank=0), Card(color=0, rank=7)]
    state.expeditions[0][1] = [Card(color=1, rank=8)]
    state.expeditions[1][0] = [
        Card(color=0, rank=0),
        Card(color=0, rank=5),
        Card(color=0, rank=6),
        Card(color=0, rank=7),
        Card(color=0, rank=8),
    ]
    state.discards[0] = [Card(color=0, rank=6)]

    assert state.score_diff(0) < 0
    assert bot._act_draw(state) == draw_from_discard_action(0)


def test_safe_heuristic_tier3_self_play_opens_expeditions() -> None:
    state = GameState.new_game(tier_config("tier3"), seed=1)
    bot = SafeHeuristicBot()
    player0_actions: list[int] = []

    for _ in range(60):
        if state.terminal:
            break
        action = bot.act(state)
        unified = state.to_unified_action(action)
        if state.current_player == 0:
            player0_actions.append(unified)
        state.apply_unified_action(unified)

    play_actions = [
        action
        for action in player0_actions
        if action < state.config.card_action_size and action % 2 == 0
    ]

    assert play_actions
    assert any(state.expeditions[0][color] for color in range(state.config.n_colors))
