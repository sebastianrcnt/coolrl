use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cell::Cell;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::thread;

pub const DEFAULT_BOARD_SIZE: usize = 9;
pub const MIN_BOARD_SIZE: usize = 5;
pub const MAX_BOARD_SIZE: usize = 19;
pub const FEATURE_PLANES: usize = 4;

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub action: usize,
    pub visit_policy: Vec<f32>,
    pub root_value: f32,
}

pub trait Evaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32);
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub board_size: usize,
    pub board: Vec<i8>,
    pub to_play: i8,
    pub terminal: bool,
    pub winner: i8,
    pub last_action: Option<usize>,
    pub move_count: usize,
    pub exactly_five: bool,
}

impl GameState {
    pub fn new(board_size: usize) -> Self {
        assert!(
            (MIN_BOARD_SIZE..=MAX_BOARD_SIZE).contains(&board_size),
            "unsupported board size"
        );
        Self {
            board_size,
            board: vec![0; board_size * board_size],
            to_play: 1,
            terminal: false,
            winner: 0,
            last_action: None,
            move_count: 0,
            exactly_five: false,
        }
    }

    pub fn with_exactly_five(board_size: usize, exactly_five: bool) -> Self {
        Self {
            exactly_five,
            ..Self::new(board_size)
        }
    }

    pub fn action_size(&self) -> usize {
        self.board_size * self.board_size
    }

    pub fn feature_stride(&self) -> usize {
        FEATURE_PLANES * self.action_size()
    }

    pub fn legal_moves(&self) -> Vec<bool> {
        self.board
            .iter()
            .map(|stone| !self.terminal && *stone == 0)
            .collect()
    }

    pub fn apply_action(&mut self, action: usize) -> bool {
        if self.terminal || action >= self.action_size() || self.board[action] != 0 {
            return false;
        }

        let player = self.to_play;
        let row = action / self.board_size;
        let col = action % self.board_size;
        self.board[action] = player;
        self.last_action = Some(action);
        self.move_count += 1;
        self.to_play = -player;

        if self.is_winning_move(row, col, player) {
            self.winner = player;
            self.terminal = true;
        } else if self.move_count == self.action_size() {
            self.winner = 0;
            self.terminal = true;
        }
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

    pub fn write_features(&self, out: &mut [f32]) -> bool {
        let action_size = self.action_size();
        if out.len() < self.feature_stride() {
            return false;
        }
        let color = if self.to_play == 1 { 1.0 } else { 0.0 };
        for action in 0..action_size {
            out[action] = if self.board[action] == self.to_play {
                1.0
            } else {
                0.0
            };
            out[action_size + action] = if self.board[action] == -self.to_play {
                1.0
            } else {
                0.0
            };
            out[2 * action_size + action] = if self.last_action == Some(action) {
                1.0
            } else {
                0.0
            };
            out[3 * action_size + action] = color;
        }
        true
    }

    fn is_winning_move(&self, row: usize, col: usize, player: i8) -> bool {
        for (dr, dc) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
            let count = 1
                + self.count_dir(row, col, dr, dc, player)
                + self.count_dir(row, col, -dr, -dc, player);
            if self.exactly_five {
                if count == 5 {
                    return true;
                }
            } else if count >= 5 {
                return true;
            }
        }
        false
    }

    fn count_dir(&self, row: usize, col: usize, dr: isize, dc: isize, player: i8) -> usize {
        let mut total = 0;
        let mut r = row as isize + dr;
        let mut c = col as isize + dc;
        while r >= 0 && r < self.board_size as isize && c >= 0 && c < self.board_size as isize {
            let idx = r as usize * self.board_size + c as usize;
            if self.board[idx] != player {
                break;
            }
            total += 1;
            r += dr;
            c += dc;
        }
        total
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new(DEFAULT_BOARD_SIZE)
    }
}

#[derive(Clone, Debug)]
pub struct TreeNode {
    pub action_size: usize,
    pub to_play: i8,
    pub prior: f32,
    pub visit_count: i32,
    pub value_sum: f32,
    pub children: Vec<Option<Box<TreeNode>>>,
    pub expanded: bool,
}

impl TreeNode {
    pub fn new(action_size: usize, to_play: i8, prior: f32) -> Self {
        Self {
            action_size,
            to_play,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children: Vec::new(),
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

    fn has_children(&self) -> bool {
        self.children.iter().any(Option::is_some)
    }

    fn ensure_children(&mut self) {
        if self.children.is_empty() {
            self.children.resize_with(self.action_size, || None);
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

    pub fn search(
        &self,
        state: &GameState,
        num_simulations: usize,
        temperature: f32,
    ) -> SearchResult {
        self.search_with_root_noise(state, num_simulations, temperature, None, 0.0)
    }

    pub fn search_with_root_noise(
        &self,
        state: &GameState,
        num_simulations: usize,
        temperature: f32,
        root_noise: Option<&[f32]>,
        root_noise_epsilon: f32,
    ) -> SearchResult {
        let action_size = state.action_size();
        let mut root = TreeNode::new(action_size, state.to_play, 0.0);
        let mut root_value = 0.0;
        if !state.terminal {
            let (priors, value) = self.evaluator.evaluate(state);
            Self::expand(&mut root, state, &priors);
            if let Some(noise) = root_noise {
                Self::apply_root_noise(&mut root, noise, root_noise_epsilon);
            }
            root_value = value;
        }

        for _ in 0..num_simulations {
            Self::simulate_once(&mut root, state, self.c_puct, &self.evaluator);
        }

        let mut counts = vec![0.0_f32; action_size];
        for (action, child) in root.children.iter().enumerate() {
            if let Some(child) = child {
                counts[action] = child.visit_count as f32;
            }
        }

        normalize_counts_or_legal(&mut counts, state);
        let action = sample_action_from_policy(&counts, temperature);
        SearchResult {
            action,
            visit_policy: counts,
            root_value,
        }
    }

    fn apply_root_noise(root: &mut TreeNode, noise: &[f32], epsilon: f32) {
        if noise.len() < root.action_size || epsilon <= 0.0 {
            return;
        }
        let keep = 1.0 - epsilon;
        for (action, child) in root.children.iter_mut().enumerate() {
            if let Some(child) = child {
                child.prior = keep * child.prior + epsilon * noise[action].max(0.0);
            }
        }
    }

    fn simulate_once(root: &mut TreeNode, root_state: &GameState, c_puct: f32, evaluator: &E) {
        if root_state.terminal {
            return;
        }

        let mut state = root_state.clone();
        let mut path: Vec<*mut TreeNode> = Vec::new();
        let mut node_ptr: *mut TreeNode = root;
        path.push(node_ptr);

        loop {
            let node = unsafe { &mut *node_ptr };
            if !(node.expanded && node.has_children() && !state.terminal) {
                break;
            }
            let action =
                Self::select_child_action(node, c_puct).expect("expanded node must have a child");
            if !state.apply_action(action) {
                return;
            }
            let child = node
                .children
                .get_mut(action)
                .and_then(Option::as_mut)
                .expect("selected child must exist");
            node_ptr = child.as_mut() as *mut TreeNode;
            path.push(node_ptr);
        }

        if state.terminal {
            Self::backup(&path, state.outcome_for_player(state.to_play));
            return;
        }

        let node = unsafe { &mut *node_ptr };
        let (priors, value) = evaluator.evaluate(&state);
        Self::expand(node, &state, &priors);
        Self::backup(&path, value);
    }

    fn select_child_action(node: &TreeNode, c_puct: f32) -> Option<usize> {
        let sqrt_visits = (node.visit_count.max(1) as f32).sqrt();
        let mut best_action = None;
        let mut best_score = f32::NEG_INFINITY;
        for (action, child) in node.children.iter().enumerate() {
            if let Some(child) = child {
                let q = -child.value();
                let u = c_puct * child.prior * sqrt_visits / (1 + child.visit_count) as f32;
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_action = Some(action);
                }
            }
        }
        best_action
    }

    fn expand(node: &mut TreeNode, state: &GameState, priors: &[f32]) {
        if node.expanded {
            return;
        }
        let legal = state.legal_moves();
        let mut masked = vec![0.0_f32; state.action_size()];
        let mut total = 0.0_f32;
        let mut legal_count = 0usize;

        for i in 0..state.action_size() {
            if legal[i] {
                masked[i] = priors.get(i).copied().unwrap_or(0.0).max(0.0);
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
            for i in 0..state.action_size() {
                if legal[i] {
                    masked[i] = p;
                }
            }
        } else {
            for value in &mut masked {
                *value /= total;
            }
        }

        node.ensure_children();
        for i in 0..state.action_size() {
            if legal[i] {
                node.children[i] = Some(Box::new(TreeNode::new(
                    state.action_size(),
                    -state.to_play,
                    masked[i],
                )));
            }
        }
        node.expanded = true;
    }

    fn backup(path: &[*mut TreeNode], mut value: f32) {
        for node_ptr in path.iter().rev() {
            let node = unsafe { &mut **node_ptr };
            node.visit_count += 1;
            node.value_sum += value;
            value = -value;
        }
    }
}

fn normalize_counts_or_legal(counts: &mut [f32], state: &GameState) {
    let total: f32 = counts.iter().sum();
    if total > 0.0 {
        for p in counts {
            *p /= total;
        }
        return;
    }

    let legal = state.legal_moves();
    let legal_count = legal.iter().filter(|is_legal| **is_legal).count();
    if legal_count > 0 {
        let p = 1.0 / legal_count as f32;
        for (action, value) in counts.iter_mut().enumerate() {
            if legal[action] {
                *value = p;
            }
        }
    }
}

pub fn sample_action_from_policy(policy: &[f32], temperature: f32) -> usize {
    if policy.is_empty() {
        return 0;
    }
    if temperature <= 1.0e-6 {
        let mut best_action = 0;
        let mut best_value = f32::NEG_INFINITY;
        for (action, value) in policy.iter().enumerate() {
            if *value > best_value {
                best_value = *value;
                best_action = action;
            }
        }
        return best_action;
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

pub type EvalCallback = unsafe extern "C" fn(
    features: *const f32,
    priors_out: *mut f32,
    value_out: *mut f32,
    user_data: *mut c_void,
) -> c_int;

struct CallbackEvaluator {
    callback: EvalCallback,
    user_data: *mut c_void,
    status: Cell<c_int>,
}

impl Evaluator for CallbackEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32) {
        let action_size = state.action_size();
        if self.status.get() != 0 {
            return (vec![0.0; action_size], 0.0);
        }

        let mut features = vec![0.0_f32; state.feature_stride()];
        let mut priors = vec![0.0_f32; action_size];
        let mut value = 0.0_f32;
        state.write_features(&mut features);

        let status = unsafe {
            (self.callback)(
                features.as_ptr(),
                priors.as_mut_ptr(),
                &mut value,
                self.user_data,
            )
        };
        if status != 0 {
            self.status.set(status);
        }
        (priors, value)
    }
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_search(
    board: *const i8,
    to_play: i8,
    last_action: c_int,
    move_count: usize,
    winner: i8,
    terminal: u8,
    exactly_five: u8,
    c_puct: f32,
    num_simulations: usize,
    temperature: f32,
    root_noise: *const f32,
    root_noise_epsilon: f32,
    callback: Option<EvalCallback>,
    user_data: *mut c_void,
    out_action: *mut usize,
    out_policy: *mut f32,
    out_root_value: *mut f32,
) -> c_int {
    if board.is_null() || out_action.is_null() || out_policy.is_null() || out_root_value.is_null() {
        return -1;
    }
    let Some(callback) = callback else {
        return -2;
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        let action_size = DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE;
        let board_slice = slice::from_raw_parts(board, action_size);
        let mut state = GameState::with_exactly_five(DEFAULT_BOARD_SIZE, exactly_five != 0);
        state.board.copy_from_slice(board_slice);
        state.to_play = to_play;
        state.terminal = terminal != 0;
        state.winner = winner;
        state.last_action = if last_action < 0 {
            None
        } else {
            Some(last_action as usize)
        };
        state.move_count = move_count;

        let evaluator = CallbackEvaluator {
            callback,
            user_data,
            status: Cell::new(0),
        };
        let mcts = Mcts::new(c_puct, evaluator);
        let root_noise_slice = if root_noise.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(root_noise, action_size))
        };
        let result = mcts.search_with_root_noise(
            &state,
            num_simulations,
            temperature,
            root_noise_slice,
            root_noise_epsilon,
        );

        let callback_status = mcts.evaluator.status.get();
        if callback_status != 0 {
            return callback_status;
        }

        *out_action = result.action;
        *out_root_value = result.root_value;
        let out_policy = slice::from_raw_parts_mut(out_policy, action_size);
        out_policy.copy_from_slice(&result.visit_policy);
        0
    }));
    result.unwrap_or(-3)
}

#[derive(Clone)]
struct PendingEval {
    state: GameState,
    node: *mut TreeNode,
    path: Vec<*mut TreeNode>,
}

pub struct MctsTree {
    c_puct: f32,
    virtual_loss: f32,
    exactly_five: bool,
    board_size: usize,
    action_size: usize,
    feature_stride: usize,
    state: GameState,
    root: Box<TreeNode>,
    root_value: f32,
    pending_roots: Vec<PendingEval>,
    pending_leaves: Vec<PendingEval>,
}

impl MctsTree {
    fn new(board_size: usize, c_puct: f32, virtual_loss: f32, exactly_five: bool) -> Self {
        let action_size = board_size * board_size;
        Self {
            c_puct,
            virtual_loss,
            exactly_five,
            board_size,
            action_size,
            feature_stride: FEATURE_PLANES * action_size,
            state: GameState::with_exactly_five(board_size, exactly_five),
            root: Box::new(TreeNode::new(action_size, 1, 0.0)),
            root_value: 0.0,
            pending_roots: Vec::new(),
            pending_leaves: Vec::new(),
        }
    }

    fn root_ptr(&mut self) -> *mut TreeNode {
        self.root.as_mut() as *mut TreeNode
    }

    fn set_initial(
        &mut self,
        board: &[i8],
        to_play: i8,
        last_action: c_int,
        move_count: usize,
        winner: i8,
        terminal: bool,
    ) {
        self.state = GameState {
            board_size: self.board_size,
            board: board.to_vec(),
            to_play,
            terminal,
            winner,
            last_action: if last_action < 0 {
                None
            } else {
                Some(last_action as usize)
            },
            move_count,
            exactly_five: self.exactly_five,
        };
        self.root = Box::new(TreeNode::new(self.action_size, to_play, 0.0));
        self.root_value = 0.0;
        self.pending_roots.clear();
        self.pending_leaves.clear();
    }

    fn advance(&mut self, action: usize) -> bool {
        if self.state.terminal || action >= self.action_size || self.state.board[action] != 0 {
            return false;
        }
        let next = if self.root.children.is_empty() {
            None
        } else {
            self.root.children[action].take()
        };
        if !self.state.apply_action(action) {
            return false;
        }
        self.root = next
            .unwrap_or_else(|| Box::new(TreeNode::new(self.action_size, self.state.to_play, 0.0)));
        self.root_value = if self.root.visit_count > 0 {
            self.root.value()
        } else {
            0.0
        };
        self.pending_roots.clear();
        self.pending_leaves.clear();
        true
    }

    fn write_root_features_if_needed(&mut self, out: &mut [f32]) -> bool {
        self.pending_roots.clear();
        if self.state.terminal || self.root.expanded || out.len() < self.feature_stride {
            return false;
        }
        let node = self.root_ptr();
        self.state.write_features(out);
        self.pending_roots.push(PendingEval {
            state: self.state.clone(),
            node,
            path: vec![node],
        });
        true
    }

    fn feed_pending_roots(&mut self, priors: &[f32], value: f32) {
        let pending = std::mem::take(&mut self.pending_roots);
        for item in pending {
            let node = unsafe { &mut *item.node };
            Mcts::<CallbackEvaluator>::expand(node, &item.state, priors);
            self.root_value = value;
        }
    }

    fn collect_one_leaf(&mut self, out: &mut [f32]) -> bool {
        if self.state.terminal || out.len() < self.feature_stride {
            return false;
        }

        let mut state = self.state.clone();
        let mut node_ptr = self.root_ptr();
        let mut path = vec![node_ptr];

        loop {
            let node = unsafe { &mut *node_ptr };
            if !(node.expanded && node.has_children() && !state.terminal) {
                break;
            }
            let Some(action) = Mcts::<CallbackEvaluator>::select_child_action(node, self.c_puct)
            else {
                break;
            };
            if !state.apply_action(action) {
                return false;
            }
            let child = node
                .children
                .get_mut(action)
                .and_then(Option::as_mut)
                .expect("selected child must exist");
            node_ptr = child.as_mut() as *mut TreeNode;
            path.push(node_ptr);
        }

        if state.terminal {
            Mcts::<CallbackEvaluator>::backup(&path, state.outcome_for_player(state.to_play));
            return false;
        }

        apply_virtual_loss(&path, self.virtual_loss);
        state.write_features(out);
        self.pending_leaves.push(PendingEval {
            state,
            node: node_ptr,
            path,
        });
        true
    }

    fn feed_pending_leaves(&mut self, priors: &[f32], values: &[f32], offset: &mut usize) {
        let pending = std::mem::take(&mut self.pending_leaves);
        for item in pending {
            revert_virtual_loss(&item.path, self.virtual_loss);
            let start = *offset * self.action_size;
            let stop = start + self.action_size;
            let node = unsafe { &mut *item.node };
            Mcts::<CallbackEvaluator>::expand(node, &item.state, &priors[start..stop]);
            Mcts::<CallbackEvaluator>::backup(&item.path, values[*offset]);
            *offset += 1;
        }
    }
}

fn apply_virtual_loss(path: &[*mut TreeNode], virtual_loss: f32) {
    for node_ptr in path {
        let node = unsafe { &mut **node_ptr };
        node.visit_count += 1;
        node.value_sum += virtual_loss;
    }
}

fn revert_virtual_loss(path: &[*mut TreeNode], virtual_loss: f32) {
    for node_ptr in path {
        let node = unsafe { &mut **node_ptr };
        node.visit_count -= 1;
        node.value_sum -= virtual_loss;
    }
}

unsafe fn tree_from_ptr<'a>(tree: *mut MctsTree) -> Option<&'a mut MctsTree> {
    tree.as_mut()
}

unsafe fn tree_slice_from_ptr<'a>(
    trees: *const *mut MctsTree,
    num_trees: c_int,
) -> Option<&'a [*mut MctsTree]> {
    if trees.is_null() || num_trees < 0 {
        return None;
    }
    Some(slice::from_raw_parts(trees, num_trees as usize))
}

fn first_tree_geometry(trees: &[*mut MctsTree]) -> Option<(usize, usize)> {
    for tree_ptr in trees {
        let tree = unsafe { tree_from_ptr(*tree_ptr) };
        if let Some(tree) = tree {
            return Some((tree.action_size, tree.feature_stride));
        }
    }
    None
}

#[no_mangle]
pub extern "C" fn omok_rmcts_tree_new(
    board_size: c_int,
    c_puct: f32,
    virtual_loss: f32,
    exactly_five: c_int,
) -> *mut MctsTree {
    let board_size = board_size as usize;
    if !(MIN_BOARD_SIZE..=MAX_BOARD_SIZE).contains(&board_size) {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(MctsTree::new(
        board_size,
        c_puct,
        virtual_loss,
        exactly_five != 0,
    )))
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_tree_free(tree: *mut MctsTree) {
    if !tree.is_null() {
        drop(Box::from_raw(tree));
    }
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_tree_board_size(tree: *mut MctsTree) -> c_int {
    tree_from_ptr(tree)
        .map(|tree| tree.board_size as c_int)
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_tree_action_size(tree: *mut MctsTree) -> c_int {
    tree_from_ptr(tree)
        .map(|tree| tree.action_size as c_int)
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_tree_feature_stride(tree: *mut MctsTree) -> c_int {
    tree_from_ptr(tree)
        .map(|tree| tree.feature_stride as c_int)
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_tree_set_initial(
    tree: *mut MctsTree,
    board: *const i8,
    to_play: i8,
    last_action: c_int,
    move_count: usize,
    winner: i8,
    terminal: u8,
) {
    if tree.is_null() || board.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        if let Some(tree) = tree_from_ptr(tree) {
            let board_slice = slice::from_raw_parts(board, tree.action_size);
            tree.set_initial(
                board_slice,
                to_play,
                last_action,
                move_count,
                winner,
                terminal != 0,
            );
        }
    }));
    let _ = result;
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_tree_advance(tree: *mut MctsTree, action: c_int) -> c_int {
    if action < 0 {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        tree_from_ptr(tree)
            .map(|tree| tree.advance(action as usize))
            .unwrap_or(false)
    }));
    result.map(|ok| if ok { 1 } else { 0 }).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_prepare_roots(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    out_features: *mut f32,
    max_entries: c_int,
) -> c_int {
    if out_features.is_null() || max_entries <= 0 {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return 0;
        };
        let Some((_, feature_stride)) = first_tree_geometry(trees) else {
            return 0;
        };
        let features =
            slice::from_raw_parts_mut(out_features, max_entries as usize * feature_stride);
        let mut written = 0usize;
        for tree_ptr in trees {
            if written >= max_entries as usize {
                break;
            }
            if let Some(tree) = tree_from_ptr(*tree_ptr) {
                let start = written * feature_stride;
                let stop = start + feature_stride;
                if tree.write_root_features_if_needed(&mut features[start..stop]) {
                    written += 1;
                }
            }
        }
        written as c_int
    }));
    result.unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_feed_roots(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    priors: *const f32,
    values: *const f32,
) {
    if priors.is_null() || values.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return;
        };
        let mut offset = 0usize;
        for tree_ptr in trees {
            if let Some(tree) = tree_from_ptr(*tree_ptr) {
                if !tree.pending_roots.is_empty() {
                    let priors = slice::from_raw_parts(
                        priors.add(offset * tree.action_size),
                        tree.action_size,
                    );
                    let value = *values.add(offset);
                    tree.feed_pending_roots(priors, value);
                    offset += 1;
                }
            }
        }
    }));
    let _ = result;
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_apply_root_noise(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    noise: *const f32,
    offsets: *const i32,
    epsilon: f32,
) {
    if noise.is_null() || offsets.is_null() || epsilon <= 0.0 {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return;
        };
        let offsets = slice::from_raw_parts(offsets, trees.len() + 1);
        let total = offsets.last().copied().unwrap_or(0).max(0) as usize;
        let noise = slice::from_raw_parts(noise, total);
        for (idx, tree_ptr) in trees.iter().enumerate() {
            let Some(tree) = tree_from_ptr(*tree_ptr) else {
                continue;
            };
            if tree.state.terminal {
                continue;
            }
            let mut local = offsets[idx].max(0) as usize;
            for child in &mut tree.root.children {
                if let Some(child) = child {
                    if local < noise.len() {
                        child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[local];
                    }
                    local += 1;
                }
            }
        }
    }));
    let _ = result;
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_root_num_legal(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    out_counts: *mut i32,
) {
    if out_counts.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return;
        };
        let out_counts = slice::from_raw_parts_mut(out_counts, trees.len());
        for (idx, tree_ptr) in trees.iter().enumerate() {
            out_counts[idx] = tree_from_ptr(*tree_ptr)
                .filter(|tree| !tree.state.terminal)
                .map(|tree| {
                    tree.root
                        .children
                        .iter()
                        .filter(|child| child.is_some())
                        .count() as i32
                })
                .unwrap_or(0);
        }
    }));
    let _ = result;
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_get_root_values(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    out_values: *mut f32,
) {
    if out_values.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return;
        };
        let out_values = slice::from_raw_parts_mut(out_values, trees.len());
        for (idx, tree_ptr) in trees.iter().enumerate() {
            out_values[idx] = tree_from_ptr(*tree_ptr)
                .filter(|tree| !tree.state.terminal)
                .map(|tree| {
                    if tree.root.visit_count > 0 {
                        tree.root.value()
                    } else {
                        tree.root_value
                    }
                })
                .unwrap_or(0.0);
        }
    }));
    let _ = result;
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_collect_leaves(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    leaves_per_tree: c_int,
    out_features: *mut f32,
    max_entries: c_int,
) -> c_int {
    if out_features.is_null() || max_entries <= 0 {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return 0;
        };
        let Some((_, feature_stride)) = first_tree_geometry(trees) else {
            return 0;
        };
        for tree_ptr in trees {
            if let Some(tree) = tree_from_ptr(*tree_ptr) {
                tree.pending_leaves.clear();
            }
        }
        let leaves_per_tree = leaves_per_tree.max(1) as usize;
        let features =
            slice::from_raw_parts_mut(out_features, max_entries as usize * feature_stride);
        let mut written = 0usize;
        for tree_ptr in trees {
            let Some(tree) = tree_from_ptr(*tree_ptr) else {
                continue;
            };
            for _ in 0..leaves_per_tree {
                if written >= max_entries as usize {
                    return written as c_int;
                }
                let start = written * feature_stride;
                let stop = start + feature_stride;
                if tree.collect_one_leaf(&mut features[start..stop]) {
                    written += 1;
                }
            }
        }
        written as c_int
    }));
    result.unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_collect_leaves_threaded(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    leaves_per_tree: c_int,
    out_features: *mut f32,
    max_entries: c_int,
    num_threads: c_int,
) -> c_int {
    if out_features.is_null() || max_entries <= 0 {
        return 0;
    }
    if num_threads <= 1 || num_trees <= 1 {
        return omok_rmcts_batch_collect_leaves(
            trees,
            num_trees,
            leaves_per_tree,
            out_features,
            max_entries,
        );
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return 0;
        };
        let Some((_, feature_stride)) = first_tree_geometry(trees) else {
            return 0;
        };
        let leaves_per_tree = leaves_per_tree.max(1) as usize;
        let max_entries = max_entries as usize;
        let num_threads = (num_threads as usize).min(trees.len()).max(1);

        for tree_ptr in trees {
            if let Some(tree) = tree_from_ptr(*tree_ptr) {
                tree.pending_leaves.clear();
            }
        }

        let tree_addrs = trees
            .iter()
            .map(|tree_ptr| *tree_ptr as usize)
            .collect::<Vec<_>>();
        let out_addr = out_features as usize;
        let mut counts = vec![0usize; trees.len()];

        thread::scope(|scope| {
            let mut handles = Vec::new();
            for thread_idx in 0..num_threads {
                let tree_addrs = &tree_addrs;
                let start = (trees.len() * thread_idx) / num_threads;
                let end = (trees.len() * (thread_idx + 1)) / num_threads;
                handles.push(scope.spawn(move || {
                    let mut local_counts = Vec::new();
                    for tree_idx in start..end {
                        let segment_start = tree_idx * leaves_per_tree;
                        if segment_start >= max_entries {
                            continue;
                        }
                        let segment_capacity = leaves_per_tree.min(max_entries - segment_start);
                        let tree_ptr = tree_addrs[tree_idx] as *mut MctsTree;
                        let Some(tree) = tree_from_ptr(tree_ptr) else {
                            continue;
                        };
                        let segment_ptr =
                            (out_addr as *mut f32).add(segment_start * feature_stride);
                        let segment = slice::from_raw_parts_mut(
                            segment_ptr,
                            segment_capacity * feature_stride,
                        );
                        let mut written = 0usize;
                        for _ in 0..leaves_per_tree {
                            if written >= segment_capacity {
                                break;
                            }
                            let start = written * feature_stride;
                            let stop = start + feature_stride;
                            if tree.collect_one_leaf(&mut segment[start..stop]) {
                                written += 1;
                            }
                        }
                        local_counts.push((tree_idx, written));
                    }
                    local_counts
                }));
            }

            for handle in handles {
                for (tree_idx, count) in handle.join().expect("Rust MCTS collect worker panicked") {
                    counts[tree_idx] = count;
                }
            }
        });

        let mut compact_offset = 0usize;
        for (tree_idx, count) in counts.into_iter().enumerate() {
            if count == 0 {
                continue;
            }
            let segment_start = tree_idx * leaves_per_tree;
            if segment_start != compact_offset {
                ptr::copy(
                    out_features.add(segment_start * feature_stride),
                    out_features.add(compact_offset * feature_stride),
                    count * feature_stride,
                );
            }
            compact_offset += count;
        }
        compact_offset as c_int
    }));
    result.unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_feed_leaves(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    priors: *const f32,
    values: *const f32,
) {
    if priors.is_null() || values.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return;
        };
        let mut pending_count = 0usize;
        let mut action_size = 0usize;
        for tree_ptr in trees {
            if let Some(tree) = tree_from_ptr(*tree_ptr) {
                pending_count += tree.pending_leaves.len();
                action_size = tree.action_size;
            }
        }
        let priors = slice::from_raw_parts(priors, pending_count * action_size);
        let values = slice::from_raw_parts(values, pending_count);
        let mut offset = 0usize;
        for tree_ptr in trees {
            if let Some(tree) = tree_from_ptr(*tree_ptr) {
                tree.feed_pending_leaves(priors, values, &mut offset);
            }
        }
    }));
    let _ = result;
}

#[no_mangle]
pub unsafe extern "C" fn omok_rmcts_batch_extract_visit_counts(
    trees: *const *mut MctsTree,
    num_trees: c_int,
    out_counts: *mut f32,
) {
    if out_counts.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(trees) = tree_slice_from_ptr(trees, num_trees) else {
            return;
        };
        let Some((action_size, _)) = first_tree_geometry(trees) else {
            return;
        };
        let out_counts = slice::from_raw_parts_mut(out_counts, trees.len() * action_size);
        for (idx, tree_ptr) in trees.iter().enumerate() {
            let row = &mut out_counts[idx * action_size..(idx + 1) * action_size];
            row.fill(0.0);
            let Some(tree) = tree_from_ptr(*tree_ptr) else {
                continue;
            };
            if tree.state.terminal {
                continue;
            }
            for (action, child) in tree.root.children.iter().enumerate() {
                if let Some(child) = child {
                    row[action] = child.visit_count as f32;
                }
            }
        }
    }));
    let _ = result;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UniformEvaluator;

    impl Evaluator for UniformEvaluator {
        fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32) {
            let legal = state.legal_moves();
            let mut priors = vec![0.0_f32; state.action_size()];
            for i in 0..state.action_size() {
                priors[i] = if legal[i] { 1.0 } else { 0.0 };
            }
            (priors, 0.0)
        }
    }

    #[test]
    fn search_policy_tracks_board_size() {
        for board_size in [9, 13, 15] {
            let state = GameState::new(board_size);
            let mcts = Mcts::new(1.25, UniformEvaluator);
            let out = mcts.search(&state, 2, 0.0);
            assert_eq!(out.visit_policy.len(), board_size * board_size);
            assert!(out.action < board_size * board_size);
        }
    }

    #[test]
    fn exact_five_overline_is_not_terminal() {
        let mut state = GameState::with_exactly_five(15, true);
        for action in [0, 1, 2, 3, 5] {
            state.board[action] = 1;
        }
        state.move_count = 5;
        state.to_play = 1;
        assert!(state.apply_action(4));
        assert!(!state.terminal);
    }
}
