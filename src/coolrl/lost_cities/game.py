from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Literal

Phase = Literal["card", "draw"]


class IllegalMoveError(ValueError):
    """Raised when an action id is not legal for the current state."""


@dataclass(frozen=True, order=True)
class Card:
    color: int
    rank: int

    @property
    def is_handshake(self) -> bool:
        return self.rank == 0

    def numeric_value(self, min_rank: int) -> int:
        if self.is_handshake:
            return 0
        return min_rank + self.rank - 1

    def label(self, min_rank: int) -> str:
        if self.is_handshake:
            return f"[{self.color}]H"
        return f"[{self.color}]{self.numeric_value(min_rank)}"


@dataclass(frozen=True)
class LostCitiesConfig:
    n_colors: int = 3
    n_ranks: int = 5
    min_rank: int = 2
    n_handshakes: int = 1
    hand_size: int = 5
    expedition_penalty: int = -20
    bonus_threshold: int = 8
    bonus_amount: int = 20
    seed: int | None = None

    @property
    def deck_size(self) -> int:
        return self.n_colors * (self.n_ranks + self.n_handshakes)

    @property
    def max_rank(self) -> int:
        return self.min_rank + self.n_ranks - 1

    @property
    def card_action_size(self) -> int:
        return 2 * self.hand_size

    @property
    def draw_action_size(self) -> int:
        return 1 + self.n_colors

    @property
    def action_size(self) -> int:
        return self.card_action_size + self.draw_action_size

    def validate(self) -> None:
        if self.n_colors <= 0:
            raise ValueError("n_colors must be positive")
        if self.n_ranks <= 0:
            raise ValueError("n_ranks must be positive")
        if self.min_rank <= 0:
            raise ValueError("min_rank must be positive")
        if self.n_handshakes < 0:
            raise ValueError("n_handshakes cannot be negative")
        if self.hand_size <= 0:
            raise ValueError("hand_size must be positive")
        if self.deck_size < 2 * self.hand_size:
            raise ValueError("deck must contain at least both initial hands")
        if self.bonus_threshold <= 0:
            raise ValueError("bonus_threshold must be positive")


TIER_PRESETS: dict[str, tuple[int, int, int, int, int]] = {
    "tier0": (2, 3, 2, 0, 3),
    "tier1": (3, 5, 2, 1, 5),
    "tier2": (4, 7, 2, 2, 6),
    "tier3": (5, 9, 2, 3, 8),
}


def tier_config(name: str, *, seed: int | None = None) -> LostCitiesConfig:
    try:
        n_colors, n_ranks, min_rank, n_handshakes, hand_size = TIER_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(TIER_PRESETS))
        raise ValueError(f"unknown tier {name!r}; expected one of: {choices}") from exc
    return LostCitiesConfig(
        n_colors=n_colors,
        n_ranks=n_ranks,
        min_rank=min_rank,
        n_handshakes=n_handshakes,
        hand_size=hand_size,
        seed=seed,
    )


def config_from_mapping(data: dict[str, Any]) -> LostCitiesConfig:
    allowed = LostCitiesConfig.__dataclass_fields__.keys()
    kwargs = {key: value for key, value in data.items() if key in allowed}
    config = LostCitiesConfig(**kwargs)
    config.validate()
    return config


def load_config(path: str) -> LostCitiesConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load Lost Cities YAML configs") from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in config file: {path}")
    return config_from_mapping(data)


def build_deck(config: LostCitiesConfig) -> list[Card]:
    config.validate()
    deck: list[Card] = []
    for color in range(config.n_colors):
        deck.extend(Card(color, 0) for _ in range(config.n_handshakes))
        deck.extend(Card(color, rank) for rank in range(1, config.n_ranks + 1))
    return deck


@dataclass
class GameState:
    config: LostCitiesConfig
    deck: list[Card]
    hands: list[list[Card]]
    expeditions: list[list[list[Card]]]
    discards: list[list[Card]]
    current_player: int = 0
    phase: Phase = "card"
    # Only set during draw phase after a discard action.
    pending_discarded_color: int | None = None
    turn_count: int = 0
    terminal: bool = False

    @classmethod
    def new_game(
        cls,
        config: LostCitiesConfig | None = None,
        *,
        seed: int | None = None,
    ) -> GameState:
        config = config or LostCitiesConfig()
        config.validate()
        rng = random.Random(config.seed if seed is None else seed)
        deck = build_deck(config)
        rng.shuffle(deck)
        state = cls.empty(config)
        state.deck = deck
        for _ in range(config.hand_size):
            for player in range(2):
                state.hands[player].append(state.deck.pop())
        state.sort_hands()
        return state

    @classmethod
    def empty(cls, config: LostCitiesConfig | None = None) -> GameState:
        config = config or LostCitiesConfig()
        config.validate()
        return cls(
            config=config,
            deck=[],
            hands=[[], []],
            expeditions=[
                [[] for _ in range(config.n_colors)],
                [[] for _ in range(config.n_colors)],
            ],
            discards=[[] for _ in range(config.n_colors)],
        )

    @property
    def card_action_size(self) -> int:
        return self.config.card_action_size

    @property
    def draw_action_size(self) -> int:
        return self.config.draw_action_size

    @property
    def action_size(self) -> int:
        return self.config.action_size

    def sort_hands(self) -> None:
        for player in range(2):
            self.sort_hand(player)

    def sort_hand(self, player: int | None = None) -> None:
        player = self.current_player if player is None else player
        self.hands[player].sort(key=lambda card: (card.color, card.rank))

    def hand_slots(self, player: int | None = None) -> list[Card | None]:
        player = self.current_player if player is None else player
        hand = self.hands[player]
        return [hand[i] if i < len(hand) else None for i in range(self.config.hand_size)]

    def last_numeric_rank(self, player: int, color: int) -> int:
        ranks = [
            card.rank
            for card in self.expeditions[player][color]
            if not card.is_handshake
        ]
        return max(ranks, default=0)

    def has_numeric(self, player: int, color: int) -> bool:
        return self.last_numeric_rank(player, color) > 0

    def can_play_card(self, player: int, card: Card) -> bool:
        if card.color < 0 or card.color >= self.config.n_colors:
            return False
        if card.rank < 0 or card.rank > self.config.n_ranks:
            return False
        last_numeric = self.last_numeric_rank(player, card.color)
        if card.is_handshake:
            return last_numeric == 0
        return card.rank > last_numeric

    def legal_card_mask(self) -> list[bool]:
        mask = [False] * self.card_action_size
        if self.terminal:
            return mask
        hand = self.hands[self.current_player]
        for slot in range(self.config.hand_size):
            if slot >= len(hand):
                continue
            card = hand[slot]
            mask[2 * slot] = self.can_play_card(self.current_player, card)
            mask[2 * slot + 1] = True
        return mask

    def legal_draw_mask(self) -> list[bool]:
        mask = [False] * self.draw_action_size
        if self.terminal:
            return mask
        mask[0] = len(self.deck) > 0
        for color in range(self.config.n_colors):
            mask[1 + color] = (
                len(self.discards[color]) > 0
                and color != self.pending_discarded_color
            )
        return mask

    def legal_mask(self) -> list[bool]:
        if self.phase == "card":
            return self.legal_card_mask()
        return self.legal_draw_mask()

    def unified_legal_mask(self) -> list[bool]:
        if self.phase == "card":
            return self.legal_card_mask() + ([False] * self.draw_action_size)
        return ([False] * self.card_action_size) + self.legal_draw_mask()

    def to_unified_action(self, action_id: int, phase: Phase | None = None) -> int:
        phase = self.phase if phase is None else phase
        if phase == "card":
            if action_id < 0 or action_id >= self.card_action_size:
                raise IllegalMoveError(f"card action {action_id} is out of range")
            return action_id
        if action_id < 0 or action_id >= self.draw_action_size:
            raise IllegalMoveError(f"draw action {action_id} is out of range")
        return self.card_action_size + action_id

    def from_unified_action(self, action_id: int) -> int:
        if action_id < 0 or action_id >= self.action_size:
            raise IllegalMoveError(f"action {action_id} is out of range")
        if self.phase == "card":
            if action_id >= self.card_action_size:
                raise IllegalMoveError(
                    f"draw action {action_id} is illegal during card phase"
                )
            return action_id
        if action_id < self.card_action_size:
            raise IllegalMoveError(
                f"card action {action_id} is illegal during draw phase"
            )
        return action_id - self.card_action_size

    def apply_action(self, action_id: int) -> None:
        if self.terminal:
            raise IllegalMoveError("game is already terminal")
        mask = self.legal_mask()
        if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
            raise IllegalMoveError(
                f"illegal action {action_id} in phase {self.phase} "
                f"for player {self.current_player}"
            )
        if self.phase == "card":
            self._apply_card_action(action_id)
        else:
            self._apply_draw_action(action_id)

    def apply_unified_action(self, action_id: int) -> None:
        self.apply_action(self.from_unified_action(action_id))

    def _apply_card_action(self, action_id: int) -> None:
        slot = action_id // 2
        play = action_id % 2 == 0
        card = self.hands[self.current_player].pop(slot)
        if play:
            self.expeditions[self.current_player][card.color].append(card)
        else:
            self.discards[card.color].append(card)
            self.pending_discarded_color = card.color
        self.phase = "draw"
        if len(self.deck) == 0 and not any(
            len(self.discards[color]) > 0 and color != self.pending_discarded_color
            for color in range(self.config.n_colors)
        ):
            self.terminal = True

    def _apply_draw_action(self, action_id: int) -> None:
        if action_id == 0:
            card = self.deck.pop()
        else:
            color = action_id - 1
            card = self.discards[color].pop()
        self.hands[self.current_player].append(card)
        self.sort_hand(self.current_player)
        self.pending_discarded_color = None
        self.turn_count += 1
        if len(self.deck) == 0:
            self.terminal = True
            return
        self.current_player = 1 - self.current_player
        self.phase = "card"

    def expedition_score(self, player: int, color: int) -> int:
        return score_expedition(self.expeditions[player][color], self.config)

    def total_score(self, player: int) -> int:
        return sum(
            self.expedition_score(player, color)
            for color in range(self.config.n_colors)
        )

    def score_diff(self, player: int = 0) -> int:
        other = 1 - player
        return self.total_score(player) - self.total_score(other)


def score_expedition(expedition: list[Card], config: LostCitiesConfig) -> int:
    if not expedition:
        return 0
    handshakes = sum(1 for card in expedition if card.is_handshake)
    numeric_sum = sum(card.numeric_value(config.min_rank) for card in expedition)
    score = (numeric_sum + config.expedition_penalty) * (handshakes + 1)
    if len(expedition) >= config.bonus_threshold:
        score += config.bonus_amount
    return score
