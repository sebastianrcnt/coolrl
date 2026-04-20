from __future__ import annotations

import argparse
from dataclasses import replace

from rich.console import Console
from rich.panel import Panel

from .bots import RandomBot, SafeHeuristicBot, run_series
from .game import Card, GameState, IllegalMoveError, LostCitiesConfig, tier_config

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Footer, Header, Static

    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    App = object  # type: ignore[assignment,misc]
    ComposeResult = object  # type: ignore[assignment,misc]
    Footer = Header = Static = object  # type: ignore[assignment,misc]
    TEXTUAL_AVAILABLE = False


Mode = tuple[str, str]


def card_label(card: Card | None, config: LostCitiesConfig) -> str:
    if card is None:
        return "--"
    return card.label(config.min_rank)


def expedition_line(state: GameState, player: int, color: int) -> str:
    cards = state.expeditions[player][color]
    if not cards:
        body = "(empty)"
    else:
        hands = sum(1 for card in cards if card.is_handshake)
        nums = [
            str(card.numeric_value(state.config.min_rank))
            for card in cards
            if not card.is_handshake
        ]
        parts = []
        if hands:
            parts.append(f"H{hands}")
        if nums:
            parts.append(" ".join(nums))
        body = " | ".join(parts)
    score = state.expedition_score(player, color)
    return f"[Color {color}] {body}  ({score:+d})"


def render_state(
    state: GameState,
    *,
    selected_slot: int | None = None,
    status: str = "",
    hide_hands: bool = False,
) -> str:
    config = state.config
    player = state.current_player
    opponent = 1 - player
    lines: list[str] = []
    if state.terminal:
        p0 = state.total_score(0)
        p1 = state.total_score(1)
        lines.append(f"Game over. P0 {p0:+d} / P1 {p1:+d} / diff {p0 - p1:+d}")
        lines.append("")

    if hide_hands:
        lines.append(f"Player {player + 1} turn - press any key when ready")
        return "\n".join(lines)

    lines.append("Opponent expeditions")
    for color in range(config.n_colors):
        lines.append("  " + expedition_line(state, opponent, color))
    lines.append("")
    discard_parts = []
    for color in range(config.n_colors):
        pile = state.discards[color]
        top = card_label(pile[-1], config) if pile else "--"
        disabled = state.phase == "draw" and color == state.pending_discarded_color
        label = f"[{color}] {top} ({len(pile)})"
        if disabled:
            label = f"[dim]{label}[/dim]"
        discard_parts.append(label)
    lines.append("Discard piles (top / size)")
    lines.append("  " + "  ".join(discard_parts))
    lines.append("")
    lines.append(f"Deck: {len(state.deck)} cards remaining")
    lines.append("")
    lines.append("My expeditions")
    for color in range(config.n_colors):
        lines.append("  " + expedition_line(state, player, color))
    lines.append("")
    lines.append("My hand")
    hand_parts = []
    for index, card in enumerate(state.hand_slots(player), start=1):
        label = f"{index}:{card_label(card, config)}"
        if selected_slot == index - 1:
            label = f"[reverse]{label}[/reverse]"
        hand_parts.append(label)
    lines.append("  " + "  ".join(hand_parts))
    lines.append("")
    if state.phase == "card":
        if selected_slot is None:
            lines.append("Select card slot with 1-N, then [P]lay or [D]iscard")
        else:
            card = state.hand_slots(player)[selected_slot]
            lines.append(f"Selected: slot {selected_slot + 1} = {card_label(card, config)}")
            lines.append("Action: [P]lay  [D]iscard")
    else:
        draw_parts = ["[0]Deck" if state.legal_draw_mask()[0] else "[dim][0]Deck[/dim]"]
        for color in range(config.n_colors):
            action = 1 + color
            label = f"[{action}]Discard-c{color}"
            if not state.legal_draw_mask()[action]:
                label = f"[dim]{label}[/dim]"
            draw_parts.append(label)
        lines.append("Draw from: " + "  ".join(draw_parts))
    lines.append("")
    turn = f"P{player + 1}"
    lines.append(f"Status: {turn}, phase={state.phase}, turn={state.turn_count}")
    if status:
        lines.append(status)
    return "\n".join(lines)


if TEXTUAL_AVAILABLE:

    class LostCitiesApp(App):
        CSS = """
        Screen {
            background: #101820;
            color: #f4ead5;
        }
        #body {
            padding: 1 2;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
        ]

        def __init__(
            self,
            config: LostCitiesConfig,
            *,
            hide_on_swap: bool = False,
            seed: int | None = None,
        ):
            super().__init__()
            self.config = replace(config, seed=seed) if seed is not None else config
            self.hide_on_swap = hide_on_swap
            self.state: GameState | None = None
            self.selected_slot: int | None = None
            self.status = ""
            self.mode: Mode | None = None
            self.bots: list[RandomBot | SafeHeuristicBot | None] = [None, None]
            self.hide_hands = False
            self.body: Static | None = None

        def compose(self) -> ComposeResult:
            yield Header()
            self.body = Static(id="body")
            yield self.body
            yield Footer()

        def on_mount(self) -> None:
            self._render_menu()

        def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
            key = event.character or event.key
            if key == "q":
                self.exit()
                return
            if self.mode is None:
                self._handle_menu_key(key)
                return
            if self.hide_hands:
                self.hide_hands = False
                self._render()
                return
            if self.state is None or self.state.terminal:
                self._render()
                return
            if self._current_actor() != "human":
                self._advance_bots()
                self._render()
                return
            if self.state.phase == "card":
                self._handle_card_key(key)
            else:
                self._handle_draw_key(key)
            self._advance_bots()
            self._maybe_hide_for_swap()
            self._render()

        def _render_menu(self) -> None:
            text = "\n".join(
                [
                    "Lost Cities",
                    "",
                    "[1] Human vs Human (hot-seat)",
                    "[2] Human vs Random bot",
                    "[3] Human vs Safe heuristic bot",
                    "[4] Random vs Random auto (100 games)",
                    "[5] Random vs Safe heuristic auto (100 games)",
                    "",
                    "Press Q to quit.",
                ]
            )
            assert self.body is not None
            self.body.update(Panel(text, title="Menu"))

        def _handle_menu_key(self, key: str) -> None:
            if key == "1":
                self._start(("human", "human"))
            elif key == "2":
                self._start(("human", "random"))
            elif key == "3":
                self._start(("human", "safe"))
            elif key == "4":
                result = run_series(RandomBot(1), RandomBot(2), self.config, games=100)
                self.status = self._series_text("Random vs Random", result)
                self._render_menu_result()
            elif key == "5":
                result = run_series(RandomBot(1), SafeHeuristicBot(), self.config, games=100)
                self.status = self._series_text("Random vs Safe", result)
                self._render_menu_result()

        def _start(self, mode: Mode) -> None:
            self.mode = mode
            self.bots = [
                self._make_bot(mode[0], 1),
                self._make_bot(mode[1], 2),
            ]
            self.state = GameState.new_game(self.config)
            self.selected_slot = None
            self.status = ""
            self._advance_bots()
            self._render()

        def _make_bot(
            self,
            actor: str,
            seed: int,
        ) -> RandomBot | SafeHeuristicBot | None:
            if actor == "random":
                return RandomBot(seed)
            if actor == "safe":
                return SafeHeuristicBot()
            return None

        def _current_actor(self) -> str:
            assert self.state is not None and self.mode is not None
            return self.mode[self.state.current_player]

        def _handle_card_key(self, key: str) -> None:
            assert self.state is not None
            if key.isdigit():
                slot = int(key) - 1
                if 0 <= slot < self.config.hand_size:
                    self.selected_slot = slot
                    self.status = ""
                return
            if key.lower() not in {"p", "d"} or self.selected_slot is None:
                return
            action = 2 * self.selected_slot + (0 if key.lower() == "p" else 1)
            self._apply_human_action(action)
            self.selected_slot = None

        def _handle_draw_key(self, key: str) -> None:
            if not key.isdigit():
                return
            action = int(key)
            if 0 <= action <= self.config.n_colors:
                self._apply_human_action(action)

        def _apply_human_action(self, action: int) -> None:
            assert self.state is not None
            try:
                self.state.apply_action(action)
                self.status = ""
            except IllegalMoveError as exc:
                self.status = f"[red]{exc}[/red]"

        def _advance_bots(self) -> None:
            if self.state is None or self.mode is None:
                return
            for _ in range(10_000):
                if self.state.terminal or self._current_actor() == "human":
                    return
                bot = self.bots[self.state.current_player]
                assert bot is not None
                action = bot.act(self.state)
                self.state.apply_action(action)
            self.status = "[red]bot loop exceeded safety limit[/red]"

        def _maybe_hide_for_swap(self) -> None:
            if not self.hide_on_swap or self.state is None or self.mode is None:
                return
            if self.state.terminal or self.state.phase != "card":
                return
            if self.mode[self.state.current_player] == "human":
                self.hide_hands = True

        def _render(self) -> None:
            assert self.body is not None
            if self.state is None:
                self._render_menu()
                return
            text = render_state(
                self.state,
                selected_slot=self.selected_slot,
                status=self.status,
                hide_hands=self.hide_hands,
            )
            self.body.update(Panel(text, title="Lost Cities"))

        def _series_text(self, label: str, result: dict) -> str:
            return (
                f"{label}: games={result['games']}, avg_diff={result['avg_diff']:.2f}, "
                f"wins0={result['wins0']}, wins1={result['wins1']}, draws={result['draws']}"
            )

        def _render_menu_result(self) -> None:
            assert self.body is not None
            self.body.update(Panel(self.status + "\n\nPress 1-5 for another mode or Q.", title="Auto result"))


def run_cli(config: LostCitiesConfig) -> None:
    console = Console()
    console.print("Textual is not installed. Install `coolrl[lost-cities]` to use the TUI.")
    result = run_series(RandomBot(1), RandomBot(2), config, games=100)
    console.print(Panel(str(result), title="Random vs Random fallback"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lost Cities TUI")
    parser.add_argument("--tier", choices=["tier0", "tier1", "tier2", "tier3"], default="tier1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hide-on-swap", action="store_true")
    args = parser.parse_args(argv)

    config = tier_config(args.tier, seed=args.seed)
    if TEXTUAL_AVAILABLE:
        LostCitiesApp(config, hide_on_swap=args.hide_on_swap, seed=args.seed).run()
    else:
        run_cli(config)


if __name__ == "__main__":
    main()
