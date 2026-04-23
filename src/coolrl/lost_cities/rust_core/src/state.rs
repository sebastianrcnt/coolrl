use crate::config::Config;
use crate::error::EngineError;
use crate::proto;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Card {
    pub color: u32,
    pub rank: u32,
}

impl Card {
    pub fn is_handshake(self) -> bool {
        self.rank == 0
    }

    pub fn numeric_value(self, min_rank: u32) -> u32 {
        if self.is_handshake() {
            0
        } else {
            min_rank + self.rank - 1
        }
    }

    pub fn label(self, min_rank: u32) -> String {
        if self.is_handshake() {
            format!("[{}]H", self.color)
        } else {
            format!("[{}]{}", self.color, self.numeric_value(min_rank))
        }
    }

    pub fn to_proto(self, config: &Config) -> proto::Card {
        proto::Card {
            color: self.color,
            rank: self.rank,
            numeric_value: self.numeric_value(config.min_rank),
            label: self.label(config.min_rank),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Card,
    Draw,
}

impl Phase {
    pub fn to_proto(self) -> i32 {
        match self {
            Self::Card => proto::Phase::Card as i32,
            Self::Draw => proto::Phase::Draw as i32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub config: Config,
    pub deck: Vec<Card>,
    pub hands: [Vec<Card>; 2],
    pub expeditions: [Vec<Vec<Card>>; 2],
    pub discards: Vec<Vec<Card>>,
    pub current_player: usize,
    pub phase: Phase,
    pub pending_discarded_color: Option<u32>,
    pub turn_count: u32,
    pub terminal: bool,
}

impl GameState {
    pub fn new_game(config: Config) -> Result<Self, EngineError> {
        config.validate()?;
        let mut deck = build_deck(&config);
        match config.seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                deck.shuffle(&mut rng);
            }
            None => {
                let mut rng = rand::thread_rng();
                deck.shuffle(&mut rng);
            }
        }

        let mut state = Self::empty(config)?;
        state.deck = deck;
        for _ in 0..state.config.hand_size {
            for player in 0..2 {
                let card = state
                    .deck
                    .pop()
                    .expect("validated deck must contain both initial hands");
                state.hands[player].push(card);
            }
        }
        state.sort_hands();
        Ok(state)
    }

    pub fn empty(config: Config) -> Result<Self, EngineError> {
        config.validate()?;
        let n_colors = config.n_colors;
        Ok(Self {
            config,
            deck: Vec::new(),
            hands: [Vec::new(), Vec::new()],
            expeditions: std::array::from_fn(|_| vec![Vec::new(); n_colors]),
            discards: vec![Vec::new(); n_colors],
            current_player: 0,
            phase: Phase::Card,
            pending_discarded_color: None,
            turn_count: 0,
            terminal: false,
        })
    }

    pub fn sort_hands(&mut self) {
        self.sort_hand(0);
        self.sort_hand(1);
    }

    pub fn sort_hand(&mut self, player: usize) {
        self.hands[player].sort_by_key(|card| (card.color, card.rank));
    }

    pub fn last_numeric_rank(&self, player: usize, color: usize) -> u32 {
        self.expeditions[player][color]
            .iter()
            .filter(|card| !card.is_handshake())
            .map(|card| card.rank)
            .max()
            .unwrap_or(0)
    }

    pub fn can_play_card(&self, player: usize, card: Card) -> bool {
        let color = match usize::try_from(card.color) {
            Ok(value) if value < self.config.n_colors => value,
            _ => return false,
        };
        if card.rank > self.config.n_ranks {
            return false;
        }
        let last_numeric = self.last_numeric_rank(player, color);
        if card.is_handshake() {
            last_numeric == 0
        } else {
            card.rank > last_numeric
        }
    }

    pub fn legal_card_mask_phase(&self) -> Vec<bool> {
        let mut mask = vec![false; self.config.card_action_size()];
        if self.terminal {
            return mask;
        }

        let hand = &self.hands[self.current_player];
        for slot in 0..self.config.hand_size {
            let Some(card) = hand.get(slot).copied() else {
                continue;
            };
            mask[2 * slot] = self.can_play_card(self.current_player, card);
            mask[2 * slot + 1] = true;
        }
        mask
    }

    pub fn legal_draw_mask_phase(&self) -> Vec<bool> {
        let mut mask = vec![false; self.config.draw_action_size()];
        if self.terminal {
            return mask;
        }

        mask[0] = !self.deck.is_empty();
        for color in 0..self.config.n_colors {
            let pending = self.pending_discarded_color == Some(color as u32);
            mask[1 + color] = !self.discards[color].is_empty() && !pending;
        }
        mask
    }

    pub fn legal_unified_mask(&self) -> Vec<bool> {
        if self.terminal {
            return vec![false; self.config.action_space_size()];
        }
        match self.phase {
            Phase::Card => {
                let mut mask = self.legal_card_mask_phase();
                mask.resize(self.config.action_space_size(), false);
                mask
            }
            Phase::Draw => {
                let mut mask = vec![false; self.config.card_action_size()];
                mask.extend(self.legal_draw_mask_phase());
                mask
            }
        }
    }

    pub fn apply_unified_action(&mut self, action_id: u32) -> Result<(), EngineError> {
        if self.terminal {
            return Err(EngineError::failed_precondition("game is already terminal"));
        }

        let action_index = usize::try_from(action_id)
            .map_err(|_| EngineError::failed_precondition("action_id is out of range"))?;
        let mask = self.legal_unified_mask();
        if action_index >= mask.len() || !mask[action_index] {
            return Err(EngineError::failed_precondition(format!(
                "illegal action {} in phase {:?} for player {}",
                action_id, self.phase, self.current_player
            )));
        }

        match self.phase {
            Phase::Card => self.apply_card_action(action_index),
            Phase::Draw => self.apply_draw_action(action_index),
        }
        Ok(())
    }

    pub fn total_score(&self, player: usize) -> i32 {
        self.expeditions[player]
            .iter()
            .map(|expedition| score_expedition(expedition, &self.config))
            .sum()
    }

    pub fn score_diff(&self, player: usize) -> i32 {
        self.total_score(player) - self.total_score(1 - player)
    }

    pub fn validate_invariants(&self) -> Result<(), String> {
        let total_cards = self.deck.len()
            + self.hands.iter().map(Vec::len).sum::<usize>()
            + self
                .expeditions
                .iter()
                .flat_map(|player| player.iter())
                .map(Vec::len)
                .sum::<usize>()
            + self.discards.iter().map(Vec::len).sum::<usize>();
        if total_cards != self.config.deck_size() {
            return Err(format!(
                "card conservation failed: expected {}, found {}",
                self.config.deck_size(),
                total_cards
            ));
        }

        for (player_index, hand) in self.hands.iter().enumerate() {
            if !hand.windows(2).all(|pair| pair[0] <= pair[1]) {
                return Err(format!("hand {} is not sorted", player_index));
            }
        }

        for (player_index, expeditions) in self.expeditions.iter().enumerate() {
            for (color, expedition) in expeditions.iter().enumerate() {
                let mut seen_numeric = false;
                let mut last_numeric = 0;
                for card in expedition {
                    if card.is_handshake() {
                        if seen_numeric {
                            return Err(format!(
                                "player {} expedition {} has handshake after numeric",
                                player_index, color
                            ));
                        }
                        continue;
                    }
                    seen_numeric = true;
                    if card.rank <= last_numeric {
                        return Err(format!(
                            "player {} expedition {} is not strictly increasing",
                            player_index, color
                        ));
                    }
                    last_numeric = card.rank;
                }
            }
        }

        if self.phase == Phase::Card && self.pending_discarded_color.is_some() {
            return Err("pending_discarded_color must be None during card phase".to_string());
        }

        let any_legal = self.legal_unified_mask().into_iter().any(|value| value);
        if self.terminal && any_legal {
            return Err("terminal state must have no legal actions".to_string());
        }
        if !self.terminal && !any_legal {
            return Err("non-terminal state must have at least one legal action".to_string());
        }

        Ok(())
    }

    pub(crate) fn build_legal_action_set(
        &self,
        state_version: u64,
        include_actions: bool,
    ) -> proto::LegalActionSet {
        let action_space_size = self.config.action_space_size() as u32;
        if self.terminal || !include_actions {
            return proto::LegalActionSet {
                state_version,
                actions: Vec::new(),
                mask: vec![false; self.config.action_space_size()],
                action_space_size,
                phase: self.phase.to_proto(),
            };
        }

        let mask = self.legal_unified_mask();
        let actions = match self.phase {
            Phase::Card => self.build_card_actions(&mask),
            Phase::Draw => self.build_draw_actions(&mask),
        };

        proto::LegalActionSet {
            state_version,
            actions,
            mask,
            action_space_size,
            phase: self.phase.to_proto(),
        }
    }

    fn apply_card_action(&mut self, action_index: usize) {
        let slot = action_index / 2;
        let play = action_index % 2 == 0;
        let card = self.hands[self.current_player].remove(slot);
        let color = card.color as usize;

        if play {
            self.expeditions[self.current_player][color].push(card);
            self.pending_discarded_color = None;
        } else {
            self.discards[color].push(card);
            self.pending_discarded_color = Some(card.color);
        }

        self.phase = Phase::Draw;
        if self.deck.is_empty() && !self.legal_draw_mask_phase().into_iter().any(|value| value) {
            self.terminal = true;
        }
    }

    fn apply_draw_action(&mut self, action_index: usize) {
        let draw_index = action_index - self.config.card_action_size();
        let card = if draw_index == 0 {
            self.deck.pop().expect("legal deck draw")
        } else {
            let color = draw_index - 1;
            self.discards[color].pop().expect("legal discard draw")
        };

        self.hands[self.current_player].push(card);
        self.sort_hand(self.current_player);
        self.pending_discarded_color = None;
        self.turn_count += 1;

        if self.deck.is_empty() {
            self.terminal = true;
            return;
        }

        self.current_player = 1 - self.current_player;
        self.phase = Phase::Card;
    }

    fn build_card_actions(&self, mask: &[bool]) -> Vec<proto::Action> {
        let hand = &self.hands[self.current_player];
        let mut actions = Vec::new();
        for slot in 0..self.config.hand_size {
            let Some(card) = hand.get(slot).copied() else {
                continue;
            };
            let play_id = 2 * slot;
            if mask[play_id] {
                actions.push(proto::Action {
                    id: play_id as u32,
                    kind: proto::ActionKind::PlayCard as i32,
                    hand_slot: slot as u32,
                    card: Some(card.to_proto(&self.config)),
                    discard_color: 0,
                    description: format!("Play {}", card.label(self.config.min_rank)),
                });
            }
            let discard_id = play_id + 1;
            if mask[discard_id] {
                actions.push(proto::Action {
                    id: discard_id as u32,
                    kind: proto::ActionKind::DiscardCard as i32,
                    hand_slot: slot as u32,
                    card: Some(card.to_proto(&self.config)),
                    discard_color: 0,
                    description: format!("Discard {}", card.label(self.config.min_rank)),
                });
            }
        }
        actions
    }

    fn build_draw_actions(&self, mask: &[bool]) -> Vec<proto::Action> {
        let mut actions = Vec::new();
        let deck_draw_id = self.config.card_action_size();
        if mask[deck_draw_id] {
            actions.push(proto::Action {
                id: deck_draw_id as u32,
                kind: proto::ActionKind::DrawDeck as i32,
                hand_slot: 0,
                card: None,
                discard_color: 0,
                description: "Draw deck".to_string(),
            });
        }

        for color in 0..self.config.n_colors {
            let action_id = self.config.card_action_size() + 1 + color;
            if mask[action_id] {
                actions.push(proto::Action {
                    id: action_id as u32,
                    kind: proto::ActionKind::DrawDiscard as i32,
                    hand_slot: 0,
                    card: None,
                    discard_color: color as u32,
                    description: format!("Draw discard {}", color),
                });
            }
        }
        actions
    }
}

pub fn build_deck(config: &Config) -> Vec<Card> {
    let mut deck = Vec::with_capacity(config.deck_size());
    for color in 0..config.n_colors as u32 {
        for _ in 0..config.n_handshakes {
            deck.push(Card { color, rank: 0 });
        }
        for rank in 1..=config.n_ranks {
            deck.push(Card { color, rank });
        }
    }
    deck
}

pub fn score_expedition(expedition: &[Card], config: &Config) -> i32 {
    if expedition.is_empty() {
        return 0;
    }

    let handshakes = expedition.iter().filter(|card| card.is_handshake()).count() as i32;
    let numeric_sum = expedition
        .iter()
        .map(|card| card.numeric_value(config.min_rank) as i32)
        .sum::<i32>();
    let mut score = (numeric_sum + config.expedition_penalty) * (handshakes + 1);
    if expedition.len() >= config.bonus_threshold {
        score += config.bonus_amount;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::{build_deck, score_expedition, Card, GameState, Phase};
    use crate::Config;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    #[test]
    fn score_examples_match_rules() {
        let config = Config {
            n_colors: 3,
            n_ranks: 5,
            min_rank: 2,
            n_handshakes: 3,
            hand_size: 5,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        };

        assert_eq!(score_expedition(&[], &config), 0);
        assert_eq!(
            score_expedition(
                &[Card { color: 0, rank: 0 }, Card { color: 0, rank: 0 }],
                &config
            ),
            -60
        );
        assert_eq!(
            score_expedition(
                &[
                    Card { color: 1, rank: 0 },
                    Card { color: 1, rank: 1 },
                    Card { color: 1, rank: 3 },
                ],
                &config,
            ),
            (2 + 4 - 20) * 2
        );
    }

    #[test]
    fn deck_generation_matches_config() {
        let config = Config {
            n_colors: 3,
            n_ranks: 5,
            min_rank: 2,
            n_handshakes: 1,
            hand_size: 5,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        };
        assert_eq!(build_deck(&config).len(), config.deck_size());
    }

    #[test]
    fn handshake_after_number_is_illegal() {
        let config = Config {
            n_colors: 3,
            n_ranks: 5,
            min_rank: 2,
            n_handshakes: 1,
            hand_size: 5,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        };
        let mut state = GameState::empty(config).expect("valid config");
        state.hands[0] = vec![Card { color: 1, rank: 0 }];
        state.expeditions[0][1] = vec![Card { color: 1, rank: 1 }];
        assert!(!state.legal_card_mask_phase()[0]);
    }

    #[test]
    fn cannot_draw_just_discarded_color() {
        let config = Config {
            n_colors: 3,
            n_ranks: 5,
            min_rank: 2,
            n_handshakes: 1,
            hand_size: 5,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        };
        let mut state = GameState::empty(config).expect("valid config");
        state.hands[0] = vec![Card { color: 2, rank: 2 }];
        state.deck = vec![Card { color: 0, rank: 1 }];
        state
            .apply_unified_action(1)
            .expect("discard should be legal");
        let draw_mask = state.legal_draw_mask_phase();
        assert!(!draw_mask[1 + 2]);
    }

    #[test]
    fn deck_exhaustion_after_last_draw_is_terminal() {
        let config = Config {
            n_colors: 3,
            n_ranks: 5,
            min_rank: 2,
            n_handshakes: 1,
            hand_size: 5,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        };
        let mut state = GameState::empty(config).expect("valid config");
        state.hands[0] = vec![Card { color: 0, rank: 1 }];
        state.deck = vec![Card { color: 1, rank: 1 }];
        state
            .apply_unified_action(1)
            .expect("discard should be legal");
        state
            .apply_unified_action(state.config.card_action_size() as u32)
            .expect("deck draw should be legal");
        assert!(state.terminal);
        assert_eq!(state.phase, Phase::Draw);
        assert!(state.deck.is_empty());
    }

    #[test]
    fn card_phase_terminal_when_no_draw_sources_exist() {
        let config = Config {
            n_colors: 3,
            n_ranks: 5,
            min_rank: 2,
            n_handshakes: 1,
            hand_size: 5,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        };
        let mut state = GameState::empty(config).expect("valid config");
        state.hands[0] = vec![Card { color: 0, rank: 1 }];
        state.hands[1] = vec![Card { color: 1, rank: 1 }];
        state
            .apply_unified_action(1)
            .expect("discard should be legal");
        assert!(state.terminal);
        assert_eq!(state.phase, Phase::Draw);
    }

    #[test]
    fn random_games_preserve_invariants() {
        let config = Config {
            n_colors: 3,
            n_ranks: 5,
            min_rank: 2,
            n_handshakes: 1,
            hand_size: 5,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        };

        for seed in 0..128_u64 {
            let mut state = GameState::new_game(config.clone().with_seed(Some(seed)))
                .expect("seeded game should be valid");
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed);
            let mut steps = 0;
            while !state.terminal {
                let legal = state
                    .legal_unified_mask()
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, is_legal)| is_legal.then_some(index as u32))
                    .collect::<Vec<_>>();
                let action = *legal
                    .choose(&mut rng)
                    .expect("non-terminal has legal action");
                state
                    .apply_unified_action(action)
                    .expect("chosen legal action must apply");
                state
                    .validate_invariants()
                    .expect("invariants must hold after every action");
                steps += 1;
                assert!(steps < 1_000);
            }
        }
    }
}
