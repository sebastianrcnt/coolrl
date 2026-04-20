use std::env;
use std::process;

use omok_rmcts::{Evaluator, GameState, Mcts, ACTION_SIZE};

struct DeterministicEvaluator {
    preferred_action: usize,
    value: f32,
}

impl Evaluator for DeterministicEvaluator {
    fn evaluate(&self, state: &GameState) -> ([f32; ACTION_SIZE], f32) {
        let legal = state.legal_moves();
        let mut priors = [0.0_f32; ACTION_SIZE];
        for action in 0..ACTION_SIZE {
            if legal[action] {
                priors[action] = 1.0e-3;
            }
        }
        if self.preferred_action < ACTION_SIZE && legal[self.preferred_action] {
            priors[self.preferred_action] = 1.0;
        }
        (priors, self.value)
    }
}

#[derive(Debug)]
struct Config {
    moves: Vec<usize>,
    preferred_action: usize,
    simulations: usize,
    c_puct: f32,
    temperature: f32,
    value: f32,
    exactly_five: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            moves: Vec::new(),
            preferred_action: 40,
            simulations: 48,
            c_puct: 1.25,
            temperature: 0.0,
            value: 0.25,
            exactly_five: false,
        }
    }
}

fn parse_moves(raw: &str) -> Result<Vec<usize>, String> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .map_err(|err| format!("invalid move {item:?}: {err}"))
        })
        .collect()
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--moves" => {
                let value = args.next().ok_or("--moves requires a value")?;
                config.moves = parse_moves(&value)?;
            }
            "--preferred-action" => {
                let value = args.next().ok_or("--preferred-action requires a value")?;
                config.preferred_action = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid preferred action: {err}"))?;
            }
            "--simulations" => {
                let value = args.next().ok_or("--simulations requires a value")?;
                config.simulations = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid simulations: {err}"))?;
            }
            "--c-puct" => {
                let value = args.next().ok_or("--c-puct requires a value")?;
                config.c_puct = value
                    .parse::<f32>()
                    .map_err(|err| format!("invalid c_puct: {err}"))?;
            }
            "--temperature" => {
                let value = args.next().ok_or("--temperature requires a value")?;
                config.temperature = value
                    .parse::<f32>()
                    .map_err(|err| format!("invalid temperature: {err}"))?;
            }
            "--value" => {
                let value = args.next().ok_or("--value requires a value")?;
                config.value = value
                    .parse::<f32>()
                    .map_err(|err| format!("invalid value: {err}"))?;
            }
            "--exactly-five" => {
                config.exactly_five = true;
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    Ok(config)
}

fn run(config: Config) -> Result<(), String> {
    let mut state = GameState::with_exactly_five(config.exactly_five);
    for action in config.moves {
        if !state.apply_action(action) {
            return Err(format!("illegal move in input sequence: {action}"));
        }
    }

    let evaluator = DeterministicEvaluator {
        preferred_action: config.preferred_action,
        value: config.value,
    };
    let mcts = Mcts::new(config.c_puct, evaluator);
    let result = mcts.search(&state, config.simulations, config.temperature);
    let policy = result
        .visit_policy
        .iter()
        .map(|value| format!("{value:.9}"))
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "{{\"action\":{},\"root_value\":{:.9},\"visit_policy\":[{}]}}",
        result.action, result.root_value, policy
    );
    Ok(())
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(err) => {
            eprintln!("{err}");
            process::exit(2);
        }
    }
}
