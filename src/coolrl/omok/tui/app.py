from __future__ import annotations

import argparse
import secrets
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rich.markup import escape
from rich.panel import Panel

from coolrl.omok.board import GameState
from coolrl.omok.mcts import MCTS

from .onnx import OnnxModelEvaluator

try:
    from textual import events
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, RichLog, Static

    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional env.
    App = object  # type: ignore[assignment,misc]
    ComposeResult = object  # type: ignore[assignment,misc]
    events = object  # type: ignore[assignment]
    Footer = Header = Horizontal = RichLog = Static = Vertical = object  # type: ignore[assignment,misc]
    TEXTUAL_AVAILABLE = False


Color = int
BLACK: Color = 1
WHITE: Color = -1
COLOR_NAMES = {BLACK: "Black", WHITE: "White"}
COLOR_MARKUP = {BLACK: "bold #f1c453", WHITE: "bold #d9f2ff"}
COLOR_SYMBOLS = {BLACK: "●", WHITE: "○"}


@dataclass(frozen=True, slots=True)
class ModelPaths:
    black: Path
    white: Path
    mode: str


@dataclass(frozen=True, slots=True)
class TuiConfig:
    model_paths: ModelPaths
    board_size: int
    device: str
    simulations: int
    leaves_per_batch: int
    c_puct: float
    temperature: float
    move_delay: float
    exactly_five: bool
    max_moves: int | None
    seed: int | None
    paused: bool
    debug_lines: int
    infinite: bool


@dataclass(slots=True)
class MatchScore:
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.black_wins + self.white_wins + self.draws

    def record(self, winner: Color) -> None:
        if winner == BLACK:
            self.black_wins += 1
        elif winner == WHITE:
            self.white_wins += 1
        else:
            self.draws += 1


@dataclass(frozen=True, slots=True)
class PlayerRuntime:
    color: Color
    model_path: Path
    evaluator: OnnxModelEvaluator

    @property
    def label(self) -> str:
        return self.model_path.name


@dataclass(frozen=True, slots=True)
class TopMove:
    action: int
    probability: float


@dataclass(frozen=True, slots=True)
class MoveDecision:
    action: int
    root_value: float
    top_moves: tuple[TopMove, ...]
    elapsed_s: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _resolve_model_paths(args: argparse.Namespace) -> ModelPaths:
    if args.model and (args.black_model or args.white_model):
        raise ValueError("use either --model, or --black-model/--white-model, not both")
    if args.model:
        path = Path(args.model)
        return ModelPaths(black=path, white=path, mode="single")
    if args.black_model and args.white_model:
        return ModelPaths(
            black=Path(args.black_model),
            white=Path(args.white_model),
            mode="versus",
        )
    if args.black_model:
        path = Path(args.black_model)
        return ModelPaths(black=path, white=path, mode="single")
    if args.white_model:
        path = Path(args.white_model)
        return ModelPaths(black=path, white=path, mode="single")
    raise ValueError("pass --model for self-play, or pass --black-model and --white-model")


def _validate_model_files(paths: ModelPaths) -> None:
    missing = sorted(
        {path for path in (paths.black, paths.white) if not path.exists()},
        key=str,
    )
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"model file not found: {joined}")
    non_files = sorted(
        {path for path in (paths.black, paths.white) if path.exists() and not path.is_file()},
        key=str,
    )
    if non_files:
        joined = ", ".join(str(path) for path in non_files)
        raise ValueError(f"model path is not a file: {joined}")


def config_from_args(args: argparse.Namespace) -> TuiConfig:
    model_paths = _resolve_model_paths(args)
    _validate_model_files(model_paths)
    return TuiConfig(
        model_paths=model_paths,
        board_size=int(args.board_size),
        device=str(args.device),
        simulations=int(args.simulations),
        leaves_per_batch=int(args.leaves_per_batch),
        c_puct=float(args.c_puct),
        temperature=float(args.temperature),
        move_delay=float(args.move_delay),
        exactly_five=bool(args.exactly_five),
        max_moves=args.max_moves,
        seed=None if args.infinite else args.seed,
        paused=bool(args.paused),
        debug_lines=int(args.debug_lines),
        infinite=bool(args.infinite),
    )


def _action_label(action: int, board_size: int) -> str:
    row, col = divmod(int(action), board_size)
    return f"{chr(ord('A') + col)}{row + 1}"


def _winner_text(state: GameState) -> str:
    if not state.terminal:
        return "In progress"
    if state.winner == 0:
        return "Draw"
    return f"{COLOR_NAMES[state.winner]} wins"


def _color_label(color: Color) -> str:
    return f"[{COLOR_MARKUP[color]}]{COLOR_SYMBOLS[color]} {COLOR_NAMES[color]}[/]"


def _format_seconds(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    return f"{seconds:.2f} s"


def _top_moves(policy: np.ndarray, board_size: int, *, limit: int = 5) -> tuple[TopMove, ...]:
    if policy.size == 0:
        return ()
    limit = min(limit, int(policy.size))
    indices = np.argpartition(policy, -limit)[-limit:]
    indices = indices[np.argsort(policy[indices])[::-1]]
    return tuple(TopMove(action=int(action), probability=float(policy[action])) for action in indices)


def _search_move(
    state: GameState,
    evaluator: OnnxModelEvaluator,
    *,
    simulations: int,
    leaves_per_batch: int,
    c_puct: float,
    temperature: float,
) -> MoveDecision:
    started = time.perf_counter()
    search = MCTS(
        c_puct=c_puct,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        evaluator=evaluator,
    )
    result = search.search_batch(
        [state],
        num_simulations=simulations,
        temperature=[temperature],
        add_noise=False,
        leaves_per_batch=leaves_per_batch,
    )[0]
    return MoveDecision(
        action=int(result.action),
        root_value=float(result.root_value),
        top_moves=_top_moves(result.visit_policy, state.board_size),
        elapsed_s=time.perf_counter() - started,
    )


def load_players(config: TuiConfig) -> dict[Color, PlayerRuntime]:
    cache: dict[Path, OnnxModelEvaluator] = {}
    players: dict[Color, PlayerRuntime] = {}
    for color, model_path in ((BLACK, config.model_paths.black), (WHITE, config.model_paths.white)):
        resolved = model_path.resolve()
        evaluator = cache.get(resolved)
        if evaluator is None:
            evaluator = OnnxModelEvaluator(resolved, device=config.device)
            evaluator.evaluate([GameState(board_size=config.board_size, exactly_five=config.exactly_five)])
            cache[resolved] = evaluator
        players[color] = PlayerRuntime(color=color, model_path=resolved, evaluator=evaluator)
    return players


if TEXTUAL_AVAILABLE:

    class BoardView(Static):
        def update_board(self, state: GameState) -> None:
            self.update(Panel(_render_board(state), title="Board", border_style="#7dd3fc"))


    class InfoView(Static):
        pass


    class DebugView(RichLog):
        pass


    class SplitterView(Static):
        def on_mouse_down(self, event: events.MouseDown) -> None:
            if event.button != 1:
                return
            self.capture_mouse()
            getattr(self.app, "_start_panel_drag")()
            event.stop()

        def on_mouse_move(self, event: events.MouseMove) -> None:
            if event.screen_x is None:
                return
            if not getattr(self.app, "panel_dragging", False):
                return
            getattr(self.app, "_drag_panel_to")(int(event.screen_x))
            event.stop()

        def on_mouse_up(self, event: events.MouseUp) -> None:
            if event.button != 1:
                return
            if getattr(self.app, "panel_dragging", False):
                getattr(self.app, "_stop_panel_drag")()
            self.release_mouse()
            event.stop()


    class OmokTuiApp(App[None]):
        CSS = """
        Screen {
            background: #101318;
            color: #edf2f7;
        }

        Header {
            background: #17202a;
            color: #edf2f7;
        }

        Footer {
            background: #17202a;
            color: #edf2f7;
        }

        #main {
            height: 1fr;
            padding: 1 2;
        }

        #board {
            height: 1fr;
            min-width: 48;
        }

        #splitter {
            width: 1;
            height: 1fr;
            background: #263241;
            color: #a78bfa;
            content-align: center middle;
            margin: 0 1;
        }

        #side {
            min-width: 42;
            height: 1fr;
        }

        #info {
            height: auto;
            margin-bottom: 1;
        }

        #debug {
            height: 1fr;
            border: solid #a78bfa;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("space", "toggle_pause", "Pause"),
            ("n", "step", "Step"),
            ("r", "reset", "Reset"),
            ("ctrl+left", "narrow_side", "Side-"),
            ("ctrl+right", "widen_side", "Side+"),
            ("ctrl+0", "reset_panels", "Panels"),
        ]

        def __init__(self, config: TuiConfig, players: dict[Color, PlayerRuntime]) -> None:
            super().__init__()
            self.config = config
            self.players = players
            self.state = GameState(board_size=config.board_size, exactly_five=config.exactly_five)
            self.paused = config.paused
            self.step_requested = False
            self.next_move_at = time.perf_counter()
            self.future: Future[MoveDecision] | None = None
            self.future_color: Color | None = None
            self.future_version = -1
            self.version = 0
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omok-tui-search")
            self.logs: list[str] = []
            self.status = "Paused" if self.paused else "Ready"
            self.score = MatchScore()
            self.game_index = 1
            self.current_seed = config.seed
            self.terminal_recorded = False
            self.result_hold_until: float | None = None
            self.board_view: BoardView | None = None
            self.splitter_view: SplitterView | None = None
            self.side_view: Vertical | None = None
            self.info_view: InfoView | None = None
            self.debug_view: DebugView | None = None
            self.side_fraction = 0.34
            self.panel_dragging = False

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="main"):
                self.board_view = BoardView(id="board")
                yield self.board_view
                self.splitter_view = SplitterView("│", id="splitter")
                yield self.splitter_view
                self.side_view = Vertical(id="side")
                with self.side_view:
                    self.info_view = InfoView(id="info")
                    self.debug_view = DebugView(
                        id="debug",
                        max_lines=self.config.debug_lines,
                        markup=True,
                        wrap=True,
                        auto_scroll=True,
                    )
                    self.debug_view.border_title = "Debug Console"
                    yield self.info_view
                    yield self.debug_view
            yield Footer()

        def on_mount(self) -> None:
            if self.config.infinite:
                self._start_new_game(reason="game 1")
                self._log("[bold #facc15]seed[/] infinite mode uses a fresh seed per game")
            elif self.config.seed is not None:
                np.random.seed(self.config.seed)
            self._log(
                "[bold #7dd3fc]session[/] "
                f"board={self.config.board_size}x{self.config.board_size} "
                f"mode={escape(self.config.model_paths.mode)} "
                f"sims={self.config.simulations} "
                f"leaves={self.config.leaves_per_batch}"
            )
            for color in (BLACK, WHITE):
                player = self.players[color]
                self._log(
                    f"{_color_label(color)} "
                    f"{escape(player.label)} via {escape(player.evaluator.provider)}"
                )
            self._apply_panel_layout()
            self.set_interval(0.1, self._tick)
            self._refresh()

        def on_unmount(self) -> None:
            if self.future is not None and not self.future.done():
                self.future.cancel()
            self.executor.shutdown(wait=False, cancel_futures=True)

        def action_toggle_pause(self) -> None:
            self.paused = not self.paused
            self.status = "Paused" if self.paused else "Running"
            self._log("[bold #facc15]pause[/] on" if self.paused else "[bold #86efac]pause[/] off")
            self._refresh()

        def action_step(self) -> None:
            self.step_requested = True
            self.paused = True
            self.status = "Step requested"
            self._refresh()

        def action_reset(self) -> None:
            if self.future is not None and not self.future.done():
                self.future.cancel()
            self._start_new_game(reason="reset")
            self._refresh()

        def action_narrow_side(self) -> None:
            self._resize_side(-0.04)

        def action_widen_side(self) -> None:
            self._resize_side(0.04)

        def action_reset_panels(self) -> None:
            self.side_fraction = 0.34
            self._apply_panel_layout()

        def _resize_side(self, delta: float) -> None:
            self.side_fraction = self._clamp_side_fraction(self.side_fraction + delta)
            self._apply_panel_layout()

        def _start_panel_drag(self) -> None:
            self.panel_dragging = True

        def _stop_panel_drag(self) -> None:
            self.panel_dragging = False

        def _drag_panel_to(self, screen_x: int) -> None:
            main = self.query_one("#main")
            left = main.region.x
            width = max(1, main.region.width)
            right = left + width
            self.side_fraction = self._clamp_side_fraction((right - screen_x) / width)
            self._apply_panel_layout()

        @staticmethod
        def _clamp_side_fraction(value: float) -> float:
            return min(0.6, max(0.22, float(value)))

        def _apply_panel_layout(self) -> None:
            if self.board_view is None or self.side_view is None:
                return
            side_units = max(22, int(round(self.side_fraction * 100)))
            board_units = max(40, 100 - side_units)
            self.board_view.styles.width = f"{board_units}fr"
            self.side_view.styles.width = f"{side_units}fr"

        def _tick(self) -> None:
            self._poll_future()
            self._advance_infinite_game()
            if self.future is None and self._should_start_move():
                self._start_move()
            self._refresh()

        def _start_new_game(self, *, reason: str) -> None:
            self.future = None
            self.future_color = None
            self.future_version = -1
            self.version += 1
            self.terminal_recorded = False
            self.result_hold_until = None
            self.state = GameState(board_size=self.config.board_size, exactly_five=self.config.exactly_five)
            if self.config.infinite:
                if reason != "game 1":
                    self.game_index += 1
                self.current_seed = secrets.randbits(32)
                np.random.seed(self.current_seed)
            else:
                self.current_seed = self.config.seed
                if self.current_seed is not None:
                    np.random.seed(self.current_seed)
            self.next_move_at = time.perf_counter()
            self.status = "Paused" if self.paused else "Running"
            seed_text = "none" if self.current_seed is None else str(self.current_seed)
            self._log(f"[bold #7dd3fc]{escape(reason)}[/] seed={seed_text}")

        def _advance_infinite_game(self) -> None:
            if not self.config.infinite or self.result_hold_until is None:
                return
            if time.perf_counter() < self.result_hold_until:
                return
            self._start_new_game(reason=f"game {self.game_index + 1}")

        def _should_start_move(self) -> bool:
            if self.state.terminal:
                return False
            if self.config.max_moves is not None and self.state.move_count >= self.config.max_moves:
                return False
            if self.step_requested:
                return True
            if self.paused:
                return False
            return time.perf_counter() >= self.next_move_at

        def _start_move(self) -> None:
            color = self.state.to_play
            player = self.players[color]
            self.future_color = color
            self.future_version = self.version
            self.step_requested = False
            self.status = f"{COLOR_NAMES[color]} thinking"
            self.future = self.executor.submit(
                _search_move,
                self.state.clone(),
                player.evaluator,
                simulations=self.config.simulations,
                leaves_per_batch=self.config.leaves_per_batch,
                c_puct=self.config.c_puct,
                temperature=self.config.temperature,
            )

        def _poll_future(self) -> None:
            future = self.future
            if future is None or not future.done():
                return

            color = self.future_color
            version = self.future_version
            self.future = None
            self.future_color = None
            self.future_version = -1

            if future.cancelled() or color is None:
                return
            if version != self.version:
                return
            try:
                decision = future.result()
            except Exception as exc:
                self.status = "Search failed"
                self.paused = True
                self._log(f"[bold #fb7185]error[/] {escape(str(exc))}")
                return

            legal = self.state.legal_moves()
            if not (0 <= decision.action < legal.size and legal[decision.action]):
                self.status = "Illegal move"
                self.paused = True
                self._log(
                    f"[bold #fb7185]illegal[/] {COLOR_NAMES[color]} chose "
                    f"{decision.action}"
                )
                return

            move_no = self.state.move_count + 1
            label = _action_label(decision.action, self.config.board_size)
            self.state.apply_action(decision.action)
            self.version += 1
            self.next_move_at = time.perf_counter() + self.config.move_delay
            self.status = _winner_text(self.state) if self.state.terminal else "Running"
            self._log_move(move_no, color, label, decision)
            if self.state.terminal:
                self._finish_game()
            elif self.config.max_moves is not None and self.state.move_count >= self.config.max_moves:
                self.paused = True
                self.status = f"Stopped at {self.config.max_moves} moves"
                self._log(f"[bold #facc15]stop[/] max moves reached: {self.config.max_moves}")

        def _finish_game(self) -> None:
            if not self.state.terminal or self.terminal_recorded:
                return
            self.score.record(self.state.winner)
            self.terminal_recorded = True
            self.status = _winner_text(self.state)
            self.result_hold_until = time.perf_counter() + 0.5 if self.config.infinite else None
            self._log(
                f"[bold #86efac]result[/] game={self.game_index} "
                f"{escape(_winner_text(self.state))} "
                f"score=B {self.score.black_wins} / W {self.score.white_wins} / D {self.score.draws}"
            )
            if self.config.infinite:
                self._log("[dim]next[/] new game in 0.5s")

        def _log_move(
            self,
            move_no: int,
            color: Color,
            label: str,
            decision: MoveDecision,
        ) -> None:
            top = ", ".join(
                f"{_action_label(move.action, self.config.board_size)} {move.probability:.0%}"
                for move in decision.top_moves
            )
            self._log(
                f"[dim]{move_no:03d}[/] "
                f"{_color_label(color)} "
                f"[bold]{label}[/] "
                f"value={decision.root_value:+.3f} "
                f"time={_format_seconds(decision.elapsed_s)}"
            )
            if top:
                self._log(f"[dim]top[/] {escape(top)}")

        def _log(self, message: str) -> None:
            self.logs.append(message)
            if len(self.logs) > self.config.debug_lines:
                del self.logs[: len(self.logs) - self.config.debug_lines]
            if self.debug_view is not None:
                self.debug_view.write(message)

        def _refresh(self) -> None:
            if self.board_view is not None:
                self.board_view.update_board(self.state)
            if self.info_view is not None:
                next_game_in = (
                    None
                    if self.result_hold_until is None
                    else max(0.0, self.result_hold_until - time.perf_counter())
                )
                self.info_view.update(
                    _render_info(
                        self.state,
                        self.players,
                        self.config,
                        self.status,
                        self.future_color,
                        self.score,
                        self.game_index,
                        self.current_seed,
                        next_game_in,
                    )
                )


def _stone_markup(state: GameState, row: int, col: int) -> str:
    stone = int(state.board[row, col])
    action = row * state.board_size + col
    is_last = state.last_action == action
    if stone == BLACK:
        symbol = "●"
        style = "#111827 on #facc15" if is_last else "#111827 on #c9a24e"
    elif stone == WHITE:
        symbol = "○"
        style = "#f8fafc on #2563eb" if is_last else "#f8fafc"
    else:
        symbol = "·"
        style = "dim #6b7280"
    return f"[{style}]{symbol}[/]"


def _render_board(state: GameState) -> str:
    size = state.board_size
    header = "   " + " ".join(chr(ord("A") + col) for col in range(size))
    lines = [header]
    for row in range(size):
        cells = " ".join(_stone_markup(state, row, col) for col in range(size))
        lines.append(f"{row + 1:>2} {cells}")
    return "\n".join(lines)


def _render_info(
    state: GameState,
    players: dict[Color, PlayerRuntime],
    config: TuiConfig,
    status: str,
    thinking_color: Color | None,
    score: MatchScore,
    game_index: int,
    current_seed: int | None,
    next_game_in: float | None,
) -> Panel:
    thinking = "none" if thinking_color is None else COLOR_NAMES[thinking_color]
    seed_text = "none" if current_seed is None else str(current_seed)
    next_text = "" if next_game_in is None else f"[bold #facc15]Next[/] {next_game_in:.1f}s\n"
    lines = [
        f"[bold #7dd3fc]Status[/] {escape(status)}",
        f"[bold #7dd3fc]Turn[/] {_color_label(state.to_play)}",
        f"[bold #7dd3fc]Thinking[/] {escape(thinking)}",
        f"[bold #7dd3fc]Game[/] {game_index}",
        f"[bold #7dd3fc]Seed[/] {escape(seed_text)}",
        f"[bold #7dd3fc]Moves[/] {state.move_count}/{state.action_size}",
        f"[bold #7dd3fc]Result[/] {escape(_winner_text(state))}",
        next_text.rstrip(),
        "",
        "[bold #7dd3fc]Score[/]",
        f"{_color_label(BLACK)} W/L/D {score.black_wins}/{score.white_wins}/{score.draws}",
        f"{_color_label(WHITE)} W/L/D {score.white_wins}/{score.black_wins}/{score.draws}",
        "",
        f"{_color_label(BLACK)} {escape(players[BLACK].label)}",
        f"[dim]provider[/] {escape(players[BLACK].evaluator.provider)}",
        f"{_color_label(WHITE)} {escape(players[WHITE].label)}",
        f"[dim]provider[/] {escape(players[WHITE].evaluator.provider)}",
        "",
        f"[bold #7dd3fc]Board[/] {config.board_size}x{config.board_size}",
        f"[bold #7dd3fc]MCTS[/] {config.simulations} sims, leaves {config.leaves_per_batch}",
        f"[bold #7dd3fc]Temp[/] {config.temperature:g}",
        f"[bold #7dd3fc]Delay[/] {config.move_delay:g}s",
        "",
        "[dim]Space pause  N step  R reset  Ctrl+←/→ resize  Q quit[/]",
    ]
    return Panel("\n".join(lines), title="Match", border_style="#38bdf8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch one or two ONNX Omok models play in a Textual TUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="FILE.onnx",
        help="Single ONNX model used for both black and white self-play.",
    )
    parser.add_argument(
        "--black-model",
        type=Path,
        default=None,
        metavar="FILE.onnx",
        help="ONNX model used by black. If this is the only model option, it is used for both sides.",
    )
    parser.add_argument(
        "--white-model",
        type=Path,
        default=None,
        metavar="FILE.onnx",
        help="ONNX model used by white. If this is the only model option, it is used for both sides.",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        choices=(9, 15),
        default=9,
        help="Board size. The ONNX policy output must be board_size squared.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "tensorrt", "coreml"),
        default="auto",
        help="ONNX Runtime execution provider preference.",
    )
    parser.add_argument(
        "--simulations",
        type=_positive_int,
        default=256,
        metavar="N",
        help="MCTS simulations per move.",
    )
    parser.add_argument(
        "--leaves-per-batch",
        type=_positive_int,
        default=8,
        metavar="N",
        help="MCTS leaves evaluated together per search round.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        metavar="X",
        help="PUCT exploration constant used by MCTS.",
    )
    parser.add_argument(
        "--temperature",
        type=_non_negative_float,
        default=0.0,
        metavar="X",
        help="Move sampling temperature. Zero selects the most visited move.",
    )
    parser.add_argument(
        "--move-delay",
        type=_non_negative_float,
        default=0.05,
        metavar="SEC",
        help="Delay between displayed moves.",
    )
    parser.add_argument(
        "--exactly-five",
        action="store_true",
        help="Use exact-five win detection instead of five-or-more.",
    )
    parser.add_argument(
        "--max-moves",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Pause automatically after this many moves.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Seed for stochastic temperature sampling. Ignored when --infinite is set.",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Keep starting new games after terminal positions. Each game uses a fresh seed.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Launch paused; press N for a single step or Space to run.",
    )
    parser.add_argument(
        "--debug-lines",
        type=_positive_int,
        default=1000,
        metavar="N",
        help="Number of debug lines kept in the side console.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not TEXTUAL_AVAILABLE:
        print(
            "Textual is required for the Omok TUI. "
            "Install it with `uv sync --extra omok-tui` or run with "
            "`uv run --extra omok-tui ...`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        players = load_players(config)
    except Exception as exc:
        print(f"failed to initialize Omok TUI: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    OmokTuiApp(config, players).run()


if __name__ == "__main__":
    main()
