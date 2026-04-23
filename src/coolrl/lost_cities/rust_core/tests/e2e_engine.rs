use lost_cities_core::proto;
use lost_cities_core::{EngineErrorKind, LostCitiesEngine};

fn small_config(seed: u64) -> proto::GameConfig {
    proto::GameConfig {
        n_colors: 2,
        n_ranks: 2,
        min_rank: 1,
        n_handshakes: 0,
        hand_size: 1,
        expedition_penalty: 0,
        bonus_threshold: 99,
        bonus_amount: 0,
        seed: Some(seed),
    }
}

#[test]
fn apply_action_advances_version_and_phase() {
    let mut engine = LostCitiesEngine::new();
    let observation = engine
        .new_game(proto::NewGameRequest {
            session_id: "phase-flow".to_string(),
            config: Some(small_config(11)),
        })
        .expect("session should start");
    assert_eq!(observation.state_version, 0);
    assert_eq!(observation.current_player, 0);
    assert_eq!(observation.phase, proto::Phase::Card as i32);

    let card_action = observation
        .legal_actions
        .as_ref()
        .and_then(|actions| actions.actions.first())
        .map(|action| action.id)
        .expect("card phase must expose a legal action");
    let draw_phase = engine
        .apply_action(proto::ApplyActionRequest {
            session_id: "phase-flow".to_string(),
            action_id: card_action,
            expected_state_version: observation.state_version,
            observer_player: None,
        })
        .expect("card action should apply")
        .observation
        .expect("observation should be returned");
    assert_eq!(draw_phase.state_version, 1);
    assert_eq!(draw_phase.current_player, 0);
    assert_eq!(draw_phase.phase, proto::Phase::Draw as i32);

    let draw_action = draw_phase
        .legal_actions
        .as_ref()
        .and_then(|actions| actions.actions.first())
        .map(|action| action.id)
        .expect("draw phase must expose a legal action");
    let next_turn = engine
        .apply_action(proto::ApplyActionRequest {
            session_id: "phase-flow".to_string(),
            action_id: draw_action,
            expected_state_version: draw_phase.state_version,
            observer_player: None,
        })
        .expect("draw action should apply")
        .observation
        .expect("observation should be returned");
    assert_eq!(next_turn.state_version, 2);
    if !next_turn.terminal {
        assert_eq!(next_turn.current_player, 1);
        assert_eq!(next_turn.phase, proto::Phase::Card as i32);
    }
}

#[test]
fn stale_state_version_is_rejected() {
    let mut engine = LostCitiesEngine::new();
    let observation = engine
        .new_game(proto::NewGameRequest {
            session_id: "stale".to_string(),
            config: Some(small_config(3)),
        })
        .expect("session should start");
    let action_id = observation
        .legal_actions
        .as_ref()
        .and_then(|actions| actions.actions.first())
        .map(|action| action.id)
        .expect("must have a legal action");

    engine
        .apply_action(proto::ApplyActionRequest {
            session_id: "stale".to_string(),
            action_id,
            expected_state_version: observation.state_version,
            observer_player: None,
        })
        .expect("first action should succeed");

    let err = engine
        .apply_action(proto::ApplyActionRequest {
            session_id: "stale".to_string(),
            action_id,
            expected_state_version: observation.state_version,
            observer_player: None,
        })
        .expect_err("stale version must fail");
    assert_eq!(err.kind(), EngineErrorKind::FailedPrecondition);
}

#[test]
fn end_session_is_idempotent() {
    let mut engine = LostCitiesEngine::new();
    engine
        .new_game(proto::NewGameRequest {
            session_id: "cleanup".to_string(),
            config: Some(small_config(5)),
        })
        .expect("session should start");
    assert_eq!(engine.session_count(), 1);

    engine
        .end_session(proto::SessionRef {
            session_id: "cleanup".to_string(),
            observer_player: None,
        })
        .expect("first end_session should succeed");
    engine
        .end_session(proto::SessionRef {
            session_id: "cleanup".to_string(),
            observer_player: None,
        })
        .expect("second end_session should also succeed");
    assert_eq!(engine.session_count(), 0);
}

#[test]
fn off_turn_observation_hides_actions() {
    let mut engine = LostCitiesEngine::new();
    let config = small_config(9);
    engine
        .new_game(proto::NewGameRequest {
            session_id: "hidden".to_string(),
            config: Some(config.clone()),
        })
        .expect("session should start");

    let hidden = engine
        .get_observation(proto::SessionRef {
            session_id: "hidden".to_string(),
            observer_player: Some(1),
        })
        .expect("off-turn observation should succeed");
    assert_eq!(hidden.observer_player, 1);
    assert_eq!(hidden.hand.len(), config.hand_size as usize);
    assert_eq!(hidden.opponent_hand_size, config.hand_size);
    let legal = hidden.legal_actions.expect("legal action set should exist");
    assert!(legal.actions.is_empty());
    assert!(legal.mask.iter().all(|value| !value));
}

#[test]
fn full_session_loop_returns_terminal_reward_and_scores() {
    let mut engine = LostCitiesEngine::new();
    let mut observation = engine
        .new_game(proto::NewGameRequest {
            session_id: "loop".to_string(),
            config: Some(small_config(17)),
        })
        .expect("session should start");

    loop {
        if observation.terminal {
            panic!("game should not start terminal");
        }

        let action_id = observation
            .legal_actions
            .as_ref()
            .and_then(|actions| actions.actions.first())
            .map(|action| action.id)
            .expect("non-terminal observation must have a legal action");
        let expected_state_version = observation.state_version;
        let step = engine
            .apply_action(proto::ApplyActionRequest {
                session_id: "loop".to_string(),
                action_id,
                expected_state_version,
                observer_player: None,
            })
            .expect("legal action should apply");
        let next_observation = step.observation.expect("observation should be returned");

        if step.terminal {
            let observer = next_observation.observer_player;
            let other = 1 - observer;
            assert_eq!(step.reward as i32, next_observation.score_diff);
            assert_eq!(step.final_scores.len(), 2);
            assert_eq!(
                step.final_scores.get(&observer).copied(),
                Some(next_observation.my_score)
            );
            assert_eq!(
                step.final_scores.get(&other).copied(),
                Some(next_observation.opponent_score)
            );
            break;
        }

        observation = next_observation;
    }
}
