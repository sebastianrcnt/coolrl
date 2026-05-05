from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
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

    def to_snapshot(self) -> dict[str, int]:
        return {"color": self.color, "rank": self.rank}

    @classmethod
    def from_snapshot(cls, data: Any) -> Card:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(color=int(data["color"]), rank=int(data["rank"]))
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return cls(color=int(data[0]), rank=int(data[1]))
        raise ValueError(f"invalid card snapshot: {data!r}")


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

    def to_snapshot(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


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


def config_to_mapping(config: LostCitiesConfig) -> dict[str, Any]:
    return config.to_snapshot()


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


def _card_counter(cards: list[Card]) -> Counter[Card]:
    return Counter(cards)


def _cards_from_snapshot(data: Any) -> list[Card]:
    if not isinstance(data, list):
        raise ValueError(f"expected card list snapshot, got {type(data).__name__}")
    return [Card.from_snapshot(card) for card in data]


def _cards_to_snapshot(cards: list[Card]) -> list[dict[str, int]]:
    return [card.to_snapshot() for card in cards]


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
        return cls.new_game_from_deck(deck, config)

    @classmethod
    def new_game_from_deck(
        cls,
        deck: list[Card] | list[dict[str, int]] | list[tuple[int, int]],
        config: LostCitiesConfig | None = None,
    ) -> GameState:
        config = config or LostCitiesConfig()
        config.validate()
        cards = [Card.from_snapshot(card) for card in deck]
        if _card_counter(cards) != _card_counter(build_deck(config)):
            raise ValueError("deck must contain exactly the cards defined by config")

        state = cls.empty(config)
        state.deck = list(cards)
        for _ in range(config.hand_size):
            for player in range(2):
                state.hands[player].append(state.deck.pop())
        state.sort_hands()
        state.validate_invariants()
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

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any],
        *,
        validate: bool = True,
    ) -> GameState:
        config = config_from_mapping(snapshot["config"])
        phase = snapshot.get("phase", "card")
        if phase not in ("card", "draw"):
            raise ValueError(f"invalid phase: {phase!r}")

        state = cls(
            config=config,
            deck=_cards_from_snapshot(snapshot["deck"]),
            hands=[
                _cards_from_snapshot(snapshot["hands"][0]),
                _cards_from_snapshot(snapshot["hands"][1]),
            ],
            expeditions=[
                [
                    _cards_from_snapshot(color_cards)
                    for color_cards in snapshot["expeditions"][0]
                ],
                [
                    _cards_from_snapshot(color_cards)
                    for color_cards in snapshot["expeditions"][1]
                ],
            ],
            discards=[
                _cards_from_snapshot(color_cards)
                for color_cards in snapshot["discards"]
            ],
            current_player=int(snapshot.get("current_player", 0)),
            phase=phase,
            pending_discarded_color=snapshot.get("pending_discarded_color"),
            turn_count=int(snapshot.get("turn_count", 0)),
            terminal=bool(snapshot.get("terminal", False)),
        )
        if state.pending_discarded_color is not None:
            state.pending_discarded_color = int(state.pending_discarded_color)
        if validate:
            state.validate_invariants()
        return state

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.to_snapshot(),
            "deck": _cards_to_snapshot(self.deck),
            "hands": [_cards_to_snapshot(hand) for hand in self.hands],
            "expeditions": [
                [_cards_to_snapshot(expedition) for expedition in player_expeditions]
                for player_expeditions in self.expeditions
            ],
            "discards": [_cards_to_snapshot(discard) for discard in self.discards],
            "current_player": self.current_player,
            "phase": self.phase,
            "pending_discarded_color": self.pending_discarded_color,
            "turn_count": self.turn_count,
            "terminal": self.terminal,
        }

    def clone(self) -> GameState:
        return GameState(
            config=self.config,
            deck=list(self.deck),
            hands=[list(self.hands[0]), list(self.hands[1])],
            expeditions=[
                [list(exp) for exp in self.expeditions[0]],
                [list(exp) for exp in self.expeditions[1]],
            ],
            discards=[list(pile) for pile in self.discards],
            current_player=self.current_player,
            phase=self.phase,
            pending_discarded_color=self.pending_discarded_color,
            turn_count=self.turn_count,
            terminal=self.terminal,
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

    def validate_invariants(self) -> None:
        self.config.validate()
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        if self.phase not in ("card", "draw"):
            raise ValueError(f"invalid phase: {self.phase!r}")
        if len(self.hands) != 2:
            raise ValueError("hands must contain two players")
        if len(self.expeditions) != 2:
            raise ValueError("expeditions must contain two players")
        if len(self.discards) != self.config.n_colors:
            raise ValueError("discard pile count must match n_colors")

        all_cards: list[Card] = []
        all_cards.extend(self.deck)
        for player, hand in enumerate(self.hands):
            if len(hand) > self.config.hand_size:
                raise ValueError(f"hand {player} exceeds hand_size")
            if hand != sorted(hand, key=lambda card: (card.color, card.rank)):
                raise ValueError(f"hand {player} is not sorted")
            all_cards.extend(hand)

        for player, expeditions in enumerate(self.expeditions):
            if len(expeditions) != self.config.n_colors:
                raise ValueError("expedition color count must match n_colors")
            for color, expedition in enumerate(expeditions):
                self._validate_expedition(player, color, expedition)
                all_cards.extend(expedition)
        for discard in self.discards:
            all_cards.extend(discard)

        for card in all_cards:
            self._validate_card(card)
        if _card_counter(all_cards) != _card_counter(build_deck(self.config)):
            raise ValueError("card conservation failed")

        if self.phase == "card" and self.pending_discarded_color is not None:
            raise ValueError("pending_discarded_color must be None during card phase")
        if self.pending_discarded_color is not None:
            color = self.pending_discarded_color
            if color < 0 or color >= self.config.n_colors:
                raise ValueError("pending_discarded_color is out of range")
            if not self.discards[color]:
                raise ValueError("pending discard color must have a discard pile card")

        any_legal = any(self.unified_legal_mask())
        if self.terminal and any_legal:
            raise ValueError("terminal state must have no legal actions")
        if not self.terminal and not any_legal:
            raise ValueError("non-terminal state must have at least one legal action")

    def _validate_card(self, card: Card) -> None:
        if card.color < 0 or card.color >= self.config.n_colors:
            raise ValueError(f"card color out of range: {card}")
        if card.rank < 0 or card.rank > self.config.n_ranks:
            raise ValueError(f"card rank out of range: {card}")

    def _validate_expedition(
        self,
        player: int,
        color: int,
        expedition: list[Card],
    ) -> None:
        seen_numeric = False
        last_numeric = 0
        for card in expedition:
            if card.color != color:
                raise ValueError(
                    f"player {player} expedition {color} contains wrong color"
                )
            if card.is_handshake:
                if seen_numeric:
                    raise ValueError(
                        f"player {player} expedition {color} has handshake after number"
                    )
                continue
            seen_numeric = True
            if card.rank <= last_numeric:
                raise ValueError(
                    f"player {player} expedition {color} is not strictly increasing"
                )
            last_numeric = card.rank


def score_expedition(expedition: list[Card], config: LostCitiesConfig) -> int:
    if not expedition:
        return 0
    handshakes = sum(1 for card in expedition if card.is_handshake)
    numeric_sum = sum(card.numeric_value(config.min_rank) for card in expedition)
    score = (numeric_sum + config.expedition_penalty) * (handshakes + 1)
    if len(expedition) >= config.bonus_threshold:
        score += config.bonus_amount
    return score
