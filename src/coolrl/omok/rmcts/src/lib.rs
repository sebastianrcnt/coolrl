use rand::distributions::WeightedIndex;
use rand::prelude::*;

pub const BOARD_SIZE: usize = 9;
pub const ACTION_SIZE: usize = BOARD_SIZE * BOARD_SIZE;

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub action: usize,
    pub visit_policy: Vec<f32>,
    pub root_value: f32,
}

pub trait Evaluator {
    fn evaluate(&self, state: &GameState) -> ([f32; ACTION_SIZE], f32);
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub board: [i8; ACTION_SIZE],
    pub to_play: i8,
    pub terminal: bool,
    pub winner: i8,
}

impl GameState {
    pub fn new() -> Self {
        Self {
            board: [0; ACTION_SIZE],
            to_play: 1,
            terminal: false,
            winner: 0,
        }
    }

    pub fn legal_moves(&self) -> [bool; ACTION_SIZE] {
        let mut legal = [false; ACTION_SIZE];
        for i in 0..ACTION_SIZE {
            legal[i] = !self.terminal && self.board[i] == 0;
        }
        legal
    }

    pub fn apply_action(&mut self, action: usize) -> bool {
        if self.terminal || action >= ACTION_SIZE || self.board[action] != 0 {
            return false;
        }
        self.board[action] = self.to_play;
        self.to_play = -self.to_play;
        true
    }

    pub fn outcome_for_player(&self, player: i8) -> f32 {
        if !self.terminal || self.winner == 0 {
            return 0.0;
        }
        if self.winner == player {
            1.0
        } else {
            -1.0
        }
    }
}

#[derive(Clone, Debug)]
pub struct TreeNode {
    pub to_play: i8,
    pub prior: f32,
    pub visit_count: i32,
    pub value_sum: f32,
    pub children: Vec<Option<Box<TreeNode>>>,
    pub expanded: bool,
}

impl TreeNode {
    pub fn new(to_play: i8, prior: f32) -> Self {
        Self {
            to_play,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children: vec![None; ACTION_SIZE],
            expanded: false,
        }
    }

    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

pub struct Mcts<E: Evaluator> {
    pub c_puct: f32,
    pub evaluator: E,
}

impl<E: Evaluator> Mcts<E> {
    pub fn new(c_puct: f32, evaluator: E) -> Self {
        Self { c_puct, evaluator }
    }

    pub fn search(&self, state: &GameState, num_simulations: usize, temperature: f32) -> SearchResult {
        let mut root = TreeNode::new(state.to_play, 0.0);
        let (priors, root_value) = self.evaluator.evaluate(state);
        Self::expand(&mut root, state, &priors);

        for _ in 0..num_simulations {
            let mut sim_state = state.clone();
            let mut path = Vec::new();
            Self::simulate(&mut root, &mut sim_state, &mut path, self.c_puct, &self.evaluator);
        }

        let mut counts = vec![0.0_f32; ACTION_SIZE];
        for (action, child) in root.children.iter().enumerate() {
            if let Some(child) = child {
                counts[action] = child.visit_count as f32;
            }
        }

        let total: f32 = counts.iter().sum();
        if total > 0.0 {
            for p in &mut counts {
                *p /= total;
            }
        }

        let action = sample_action_from_policy(&counts, temperature);
        SearchResult {
            action,
            visit_policy: counts,
            root_value,
        }
    }

    fn simulate(node: &mut TreeNode, state: &mut GameState, path: &mut Vec<*mut TreeNode>, c_puct: f32, evaluator: &E) -> f32 {
        path.push(node as *mut TreeNode);

        if state.terminal {
            return state.outcome_for_player(state.to_play);
        }

        if !node.expanded {
            let (priors, value) = evaluator.evaluate(state);
            Self::expand(node, state, &priors);
            Self::backup(path, value);
            return value;
        }

        let action = Self::select_child_action(node, c_puct);
        let child = node.children[action].as_mut().expect("selected action must exist");
        let _ = state.apply_action(action);
        let value = -Self::simulate(child, state, path, c_puct, evaluator);
        Self::backup(path, value);
        value
    }

    fn select_child_action(node: &TreeNode, c_puct: f32) -> usize {
        let sqrt_visits = (node.visit_count.max(1) as f32).sqrt();
        let mut best_action = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (action, child) in node.children.iter().enumerate() {
            if let Some(child) = child {
                let q = -child.value();
                let u = c_puct * child.prior * sqrt_visits / (1 + child.visit_count) as f32;
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_action = action;
                }
            }
        }
        best_action
    }

    fn expand(node: &mut TreeNode, state: &GameState, priors: &[f32; ACTION_SIZE]) {
        let legal = state.legal_moves();
        let mut masked = [0.0_f32; ACTION_SIZE];
        let mut total = 0.0_f32;
        let mut legal_count = 0usize;

        for i in 0..ACTION_SIZE {
            if legal[i] {
                masked[i] = priors[i].max(0.0);
                total += masked[i];
                legal_count += 1;
            }
        }

        if legal_count == 0 {
            node.expanded = true;
            return;
        }

        if total <= 0.0 {
            let p = 1.0 / legal_count as f32;
            for i in 0..ACTION_SIZE {
                if legal[i] {
                    masked[i] = p;
                }
            }
        } else {
            for i in 0..ACTION_SIZE {
                masked[i] /= total;
            }
        }

        for i in 0..ACTION_SIZE {
            if legal[i] {
                node.children[i] = Some(Box::new(TreeNode::new(-state.to_play, masked[i])));
            }
        }
        node.expanded = true;
    }

    fn backup(path: &[*mut TreeNode], mut value: f32) {
        for node_ptr in path.iter().rev() {
            // SAFETY: pointers in `path` originate from a unique mutable traversal within a single simulation.
            let node = unsafe { &mut **node_ptr };
            node.visit_count += 1;
            node.value_sum += value;
            value = -value;
        }
    }
}

pub fn sample_action_from_policy(policy: &[f32], temperature: f32) -> usize {
    if policy.is_empty() {
        return 0;
    }
    if temperature <= 1.0e-6 {
        return policy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    let adjusted: Vec<f32> = policy
        .iter()
        .map(|p| p.max(1.0e-8).powf(1.0 / temperature))
        .collect();
    let sum: f32 = adjusted.iter().sum();
    if sum <= 0.0 {
        return 0;
    }

    let dist = WeightedIndex::new(adjusted).ok();
    let mut rng = thread_rng();
    dist.map(|d| d.sample(&mut rng)).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UniformEvaluator;

    impl Evaluator for UniformEvaluator {
        fn evaluate(&self, state: &GameState) -> ([f32; ACTION_SIZE], f32) {
            let legal = state.legal_moves();
            let mut priors = [0.0_f32; ACTION_SIZE];
            for i in 0..ACTION_SIZE {
                priors[i] = if legal[i] { 1.0 } else { 0.0 };
            }
            (priors, 0.0)
        }
    }

    #[test]
    fn returns_valid_action_and_policy() {
        let state = GameState::new();
        let mcts = Mcts::new(1.25, UniformEvaluator);
        let out = mcts.search(&state, 32, 0.0);

        assert!(out.action < ACTION_SIZE);
        assert!(out.visit_policy.iter().all(|p| *p >= 0.0));
        let sum: f32 = out.visit_policy.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-3);
    }
}
