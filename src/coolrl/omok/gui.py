from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "0")

import pygame

from .board import GameState
from .checkpoint import list_checkpoints, load_checkpoint
from .config import load_config
from .device import configure_device
from .evaluator import ModelEvaluator
from .mcts import MCTS
from .network import PolicyValueNet
from .openings import sample_balanced_openings


B612_FONT_DIR = Path(__file__).with_name("assets") / "fonts"
B612_REGULAR = B612_FONT_DIR / "B612-Regular.ttf"
B612_BOLD = B612_FONT_DIR / "B612-Bold.ttf"
DEFAULT_WINDOW_SIZE = (1280, 960)
DEFAULT_RENDER_SCALE = 2
MAX_RENDER_SCALE = 3
MIN_BOARD_PX = 360
SIDEBAR_WIDTH = 320
COCKPIT_WHITE = (238, 246, 250)
COCKPIT_CYAN = (64, 210, 232)
COCKPIT_GREEN = (96, 232, 148)
COCKPIT_AMBER = (245, 188, 74)
COCKPIT_RED = (255, 108, 92)


class OmokGUI:
    def __init__(
        self,
        checkpoint: str | None,
        config_path: str,
        device_name: str,
        simulations: int,
        human_color: int = -1,
        seed: int = 0,
        render_scale: int = DEFAULT_RENDER_SCALE,
    ) -> None:
        self.config = load_config(config_path)
        requested_device = self.config.device if device_name == "auto" else device_name
        self.device = configure_device(requested_device)
        self.simulations = simulations
        self.checkpoint_source = checkpoint
        self.model = PolicyValueNet(self.config.rules.board_size, self.config.network)
        self.metadata: dict[str, object] = {}
        self.state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
        self.current_opening: list[int] = []
        self.position_version = 0
        self.ai_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omok-ai")
        self.ai_future: Future[int] | None = None
        self.ai_future_version = -1
        self.ai_error: str | None = None
        self.checkpoints = self._discover_checkpoints(checkpoint)
        self.checkpoint_index = len(self.checkpoints) - 1 if self.checkpoints else 0
        if self.checkpoints:
            self._load_checkpoint(self.checkpoints[self.checkpoint_index])
        self.human_color = 1 if int(human_color) >= 1 else -1
        self.seed = int(seed)
        self.render_scale = 1  # supersampling removed; SCALED handles physical pixels

        pygame.init()
        pygame.display.set_caption("coolrl 9x9 Omok")
        self.display_flags = pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.SCALED
        self.window_surface = pygame.display.set_mode(DEFAULT_WINDOW_SIZE, self.display_flags, vsync=1)
        self.screen = self.window_surface
        self.clock = pygame.time.Clock()
        self.font = self._load_font(B612_BOLD, 22 * self.render_scale)
        self.small_font = self._load_font(B612_REGULAR, 18 * self.render_scale)
        self.margin = 48 * self.render_scale
        self.board_px = 720 * self.render_scale
        self.cell = self.board_px / (self.config.rules.board_size - 1)
        self.board_origin = (self.margin, self.margin)
        self.sidebar_origin = (self.margin + self.board_px + 40 * self.render_scale, self.margin)
        self.sidebar_width = SIDEBAR_WIDTH * self.render_scale
        self.sidebar_line_height = 26 * self.render_scale
        self._update_layout()

    @staticmethod
    def _load_font(path: Path, size: int) -> pygame.font.Font:
        if path.exists():
            return pygame.font.Font(str(path), size)
        return pygame.font.SysFont("DejaVu Sans", size)

    def _update_layout(self) -> None:
        width, height = self.screen.get_size()
        short_side = max(1, min(width, height))
        min_margin = 24 * self.render_scale
        max_margin = 64 * self.render_scale
        min_gap = 28 * self.render_scale
        max_gap = 56 * self.render_scale
        sidebar_width = SIDEBAR_WIDTH * self.render_scale
        min_board_px = MIN_BOARD_PX * self.render_scale
        self.margin = max(min_margin, min(max_margin, short_side // 16))
        gap = max(min_gap, min(max_gap, width // 24))
        board_space_w = width - self.margin * 2 - gap - sidebar_width
        board_space_h = height - self.margin * 2

        if board_space_w >= min_board_px:
            self.board_px = max(min_board_px, min(board_space_w, board_space_h))
            board_x = self.margin
            board_y = max(self.margin, (height - self.board_px) // 2)
            sidebar_x = board_x + self.board_px + gap
            sidebar_y = max(self.margin, board_y)
            self.sidebar_width = max(220 * self.render_scale, width - sidebar_x - self.margin)
        else:
            sidebar_height = 260 * self.render_scale
            board_space_w = width - self.margin * 2
            board_space_h = height - self.margin * 3 - sidebar_height
            self.board_px = max(240 * self.render_scale, min(board_space_w, board_space_h))
            board_x = max(self.margin, (width - self.board_px) // 2)
            board_y = self.margin
            sidebar_x = self.margin
            sidebar_y = board_y + self.board_px + self.margin
            self.sidebar_width = max(220 * self.render_scale, width - self.margin * 2)

        self.cell = self.board_px / (self.config.rules.board_size - 1)
        self.board_origin = (int(board_x), int(board_y))
        self.sidebar_origin = (int(sidebar_x), int(sidebar_y))

    def _discover_checkpoints(self, checkpoint: str | None) -> list[Path]:
        if checkpoint:
            target = Path(checkpoint)
            if target.is_dir():
                return list_checkpoints(target)
            return [target]
        candidates = [
            self.config.checkpoint_dir,
            Path("checkpoints/omok_quick"),
            Path("checkpoints/omok_smoke"),
            Path("checkpoints/omok_default"),
        ]
        for directory in candidates:
            found = list_checkpoints(directory)
            if found:
                return found
        return []

    def _load_checkpoint(self, path: Path) -> None:
        try:
            model, config, metadata = load_checkpoint(path)
        except Exception as exc:
            print(f"[gui] failed to load {path}: {exc}")
            return
        self.config = config
        self.model = model
        self.metadata = metadata
        if path in self.checkpoints:
            self.checkpoint_index = self.checkpoints.index(path)
        self.state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
        self.current_opening = []
        self._invalidate_ai()

    def _refresh_checkpoints(self) -> None:
        previous = self.checkpoints[self.checkpoint_index] if self.checkpoints else None
        self.checkpoints = self._discover_checkpoints(self.checkpoint_source)
        if not self.checkpoints:
            self.checkpoint_index = 0
            return
        if previous and previous in self.checkpoints:
            self.checkpoint_index = self.checkpoints.index(previous)
        else:
            self.checkpoint_index = len(self.checkpoints) - 1
        self._load_checkpoint(self.checkpoints[self.checkpoint_index])

    def _reset_state(self, apply_opening: bool) -> None:
        self.state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
        self.current_opening = []
        if apply_opening:
            rng = random.Random(self.seed)
            openings = sample_balanced_openings(self.config.rules.board_size, 32, rng)
            if openings:
                for action in openings[0]:
                    if self.state.terminal:
                        break
                    legal = self.state.legal_moves()
                    if not legal[action]:
                        break
                    self.state.apply_action(action)
                    self.current_opening.append(action)
        self._invalidate_ai()

    def run(self) -> None:
        running = True
        try:
            pygame.event.pump()
            self._poll_ai_move()
            self._render()
            pygame.display.flip()
            if not self.state.terminal and self.state.to_play != self.human_color:
                self._start_ai_move()

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED):
                        self._handle_resize(event)
                    elif event.type == pygame.KEYDOWN:
                        running = self._handle_key(event.key)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)

                self._poll_ai_move()
                self._render()
                pygame.display.flip()

                if not self.state.terminal and self.state.to_play != self.human_color:
                    self._start_ai_move()

                self.clock.tick(30)
        finally:
            self._cancel_ai_move()
            self.ai_executor.shutdown(wait=True, cancel_futures=True)
            pygame.quit()

    def _handle_resize(self, event: pygame.event.Event) -> None:
        if hasattr(event, "size"):
            event_width, event_height = event.size
        else:
            event_width = getattr(event, "w", getattr(event, "x", self.screen.get_width()))
            event_height = getattr(event, "h", getattr(event, "y", self.screen.get_height()))
        width = max(640, int(event_width))
        height = max(560, int(event_height))
        self.window_surface = pygame.display.set_mode((width, height), self.display_flags, vsync=1)
        self.screen = self.window_surface
        self._update_layout()

    def _handle_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_r:
            self._reset_state(apply_opening=bool(self.current_opening))
        elif key == pygame.K_s:
            self.human_color *= -1
            self._reset_state(apply_opening=False)
        elif key == pygame.K_m and not self.state.terminal:
            self._start_ai_move(force=True)
        elif key == pygame.K_n and self.checkpoints:
            self.checkpoint_index = (self.checkpoint_index + 1) % len(self.checkpoints)
            self._load_checkpoint(self.checkpoints[self.checkpoint_index])
        elif key == pygame.K_p and self.checkpoints:
            self.checkpoint_index = (self.checkpoint_index - 1) % len(self.checkpoints)
            self._load_checkpoint(self.checkpoints[self.checkpoint_index])
        elif key == pygame.K_l:
            self._refresh_checkpoints()
        elif key == pygame.K_o:
            self._reset_state(apply_opening=True)
        elif key == pygame.K_LEFTBRACKET:
            self.seed = max(0, self.seed - 1)
        elif key == pygame.K_RIGHTBRACKET:
            self.seed += 1
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.state.terminal or self.state.to_play != self.human_color:
            return
        action = self._pos_to_action(pos)
        if action is None or not self.state.legal_moves()[action]:
            return
        self._apply_action(action)

    def _apply_action(self, action: int) -> None:
        self.state.apply_action(action)
        self.position_version += 1
        self.ai_error = None

    def _cancel_ai_move(self) -> None:
        if self.ai_future is not None and not self.ai_future.done():
            self.ai_future.cancel()
        self.ai_future_version = -1

    def _invalidate_ai(self) -> None:
        self.position_version += 1
        self.ai_error = None
        self._cancel_ai_move()

    def _start_ai_move(self, force: bool = False) -> None:
        if self.ai_future is not None and self.ai_future.done():
            self._poll_ai_move()
        if self.state.terminal:
            return
        if not force and self.state.to_play == self.human_color:
            return
        if self.ai_error and not force:
            return
        if self.ai_future is not None and not self.ai_future.done():
            return

        self.ai_future_version = self.position_version
        self.ai_future = self.ai_executor.submit(
            self._search_ai_action,
            self.state.clone(),
            self.model,
            self.device,
            self.config.selfplay.c_puct,
            self.config.selfplay.dirichlet_alpha,
            self.simulations,
            self.config.selfplay.leaves_per_batch,
        )

    def _poll_ai_move(self) -> None:
        future = self.ai_future
        if future is None or not future.done():
            return

        version = self.ai_future_version
        self.ai_future = None
        self.ai_future_version = -1

        if future.cancelled():
            return

        try:
            action = future.result()
        except Exception as exc:
            self.ai_error = str(exc)
            print(f"[gui] AI move failed: {exc}")
            return

        if version != self.position_version:
            return
        legal = self.state.legal_moves()
        if 0 <= action < legal.size and legal[action]:
            self._apply_action(action)
        else:
            self.ai_error = f"illegal AI move: {action}"
            print(f"[gui] illegal AI move: {action}")

    @staticmethod
    def _search_ai_action(
        state: GameState,
        model: PolicyValueNet,
        device: str,
        c_puct: float,
        dirichlet_alpha: float,
        simulations: int,
        leaves_per_batch: int,
    ) -> int:
        evaluator = ModelEvaluator(model, device=device)
        search = MCTS(
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=0.0,
            evaluator=evaluator,
        )
        result = search.search_batch(
            [state],
            simulations,
            [0.0],
            add_noise=False,
            leaves_per_batch=leaves_per_batch,
        )[0]
        return result.action

    def _render(self) -> None:
        self._update_layout()
        self.screen.fill((0, 0, 0))
        self._draw_board()
        self._draw_sidebar()
        self._present()

    def _present(self) -> None:
        if self.screen.get_size() == self.window_surface.get_size():
            self.window_surface.blit(self.screen, (0, 0))
            return
        pygame.transform.smoothscale(self.screen, self.window_surface.get_size(), self.window_surface)

    def _draw_board(self) -> None:
        ox, oy = self.board_origin
        size = self.config.rules.board_size
        board_end_x = ox + self.board_px
        board_end_y = oy + self.board_px
        stone_radius = max(18 * self.render_scale, int(self.cell * 0.32))
        grid_width = max(1, self.render_scale)
        outline_width = max(2, 3 * self.render_scale)
        marker_radius = max(4, 5 * self.render_scale)
        black = (0, 0, 0)
        white = (245, 245, 245)

        for i in range(size):
            x = int(ox + i * self.cell)
            y = int(oy + i * self.cell)
            pygame.draw.line(self.screen, white, (ox, y), (board_end_x, y), grid_width)
            pygame.draw.line(self.screen, white, (x, oy), (x, board_end_y), grid_width)

        for row in range(size):
            for col in range(size):
                stone = self.state.board[row, col]
                if stone == 0:
                    continue
                x = int(ox + col * self.cell)
                y = int(oy + row * self.cell)
                if stone == 1:
                    pygame.draw.circle(self.screen, black, (x, y), stone_radius)
                    pygame.draw.circle(self.screen, white, (x, y), stone_radius, outline_width)
                else:
                    pygame.draw.circle(self.screen, white, (x, y), stone_radius)

        if self.state.last_action is not None:
            row, col = divmod(self.state.last_action, size)
            x = int(ox + col * self.cell)
            y = int(oy + row * self.cell)
            marker = black if self.state.board[row, col] == -1 else white
            pygame.draw.circle(self.screen, marker, (x, y), marker_radius)

    def _draw_sidebar(self) -> None:
        x, y = self.sidebar_origin
        lines = [
            "Checkpoint",
            self.checkpoints[self.checkpoint_index].name if self.checkpoints else "random-init",
            "",
            f"Device: {self.device}",
            f"Human: {'Black' if self.human_color == 1 else 'White'}",
            f"Turn: {'Black' if self.state.to_play == 1 else 'White'}",
            f"Moves: {self.state.move_count}",
            f"Simulations: {self.simulations}",
            "",
        ]
        if self.ai_future is not None and not self.ai_future.done():
            lines.append("AI: thinking...")
        elif self.ai_error:
            lines.append(f"AI error: {self.ai_error[:28]}")
        else:
            lines.append("AI: idle")
        lines.append("")
        if self.state.terminal:
            if self.state.winner == 0:
                lines.append("Result: Draw")
            else:
                lines.append(f"Result: {'Black' if self.state.winner == 1 else 'White'} wins")
        else:
            lines.append("Result: In progress")

        lines.extend(
            [
                "",
                f"Seed: {self.seed}",
                f"Opening: {len(self.current_opening)} plies" if self.current_opening else "Opening: none",
                "",
                "Controls",
                "LMB: move",
                "R: reset",
                "O: apply opening",
                "[ / ]: seed -/+",
                "S: swap side",
                "N/P: next/prev",
                "L: reload list",
                "M: force AI",
                "Esc: quit",
            ]
        )

        if self.metadata:
            lines.extend(
                [
                    "",
                    f"Iteration: {self.metadata.get('iteration', '-')}",
                    f"Best: {self.metadata.get('best_iteration', '-')}",
                    f"Role: {self.metadata.get('checkpoint_role', '-')}",
                ]
            )

        for index, text in enumerate(lines):
            font = self.font if index == 0 or text == "Controls" else self.small_font
            surface = self._render_sidebar_text(font, text, self._sidebar_color(text))
            self.screen.blit(surface, (x, y + index * self.sidebar_line_height))

    def _sidebar_color(self, text: str) -> tuple[int, int, int]:
        if not text:
            return COCKPIT_WHITE
        if text in {"Checkpoint", "Controls"}:
            return COCKPIT_CYAN
        if text.startswith("AI error"):
            return COCKPIT_RED
        if text == "AI: thinking...":
            return COCKPIT_AMBER
        if text.startswith("Result:") and "In progress" not in text:
            return COCKPIT_AMBER if "Draw" in text else COCKPIT_GREEN
        if ":" in text:
            return COCKPIT_GREEN
        return COCKPIT_WHITE

    def _render_sidebar_text(
        self,
        font: pygame.font.Font,
        text: str,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        if ":" in text and not text.startswith("AI error"):
            label, value = text.split(":", 1)
            return self._render_sidebar_pair(font, f"{label}:", value, color)
        if font.size(text)[0] <= self.sidebar_width:
            return font.render(text, True, color)
        suffix = "..."
        available = max(1, self.sidebar_width - font.size(suffix)[0])
        trimmed = text
        while trimmed and font.size(trimmed)[0] > available:
            trimmed = trimmed[:-1]
        return font.render(f"{trimmed}{suffix}", True, color)

    def _render_sidebar_pair(
        self,
        font: pygame.font.Font,
        label: str,
        value: str,
        value_color: tuple[int, int, int],
    ) -> pygame.Surface:
        label_surface = font.render(label, True, COCKPIT_CYAN)
        label_width = label_surface.get_width()
        surface_width = max(1, self.sidebar_width)
        surface_height = max(label_surface.get_height(), self.sidebar_line_height)
        surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)
        if label_width >= surface_width:
            surface.blit(self._render_truncated_text(font, label, COCKPIT_CYAN, surface_width), (0, 0))
            return surface

        surface.blit(label_surface, (0, 0))
        value_width = surface_width - label_width
        value_surface = self._render_truncated_text(font, value, value_color, value_width)
        surface.blit(value_surface, (label_width, 0))
        return surface

    def _render_truncated_text(
        self,
        font: pygame.font.Font,
        text: str,
        color: tuple[int, int, int],
        width: int,
    ) -> pygame.Surface:
        if font.size(text)[0] <= width:
            return font.render(text, True, color)
        suffix = "..."
        available = max(1, width - font.size(suffix)[0])
        trimmed = text
        while trimmed and font.size(trimmed)[0] > available:
            trimmed = trimmed[:-1]
        return font.render(f"{trimmed}{suffix}", True, color)

    def _pos_to_action(self, pos: tuple[int, int]) -> int | None:
        self._update_layout()
        ox, oy = self.board_origin
        x = pos[0] * self.render_scale
        y = pos[1] * self.render_scale
        size = self.config.rules.board_size
        tolerance = 24 * self.render_scale
        if (
            x < ox - tolerance
            or y < oy - tolerance
            or x > ox + self.board_px + tolerance
            or y > oy + self.board_px + tolerance
        ):
            return None
        col = round((x - ox) / self.cell)
        row = round((y - oy) / self.cell)
        if not (0 <= row < size and 0 <= col < size):
            return None
        return row * size + col


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play against a 9x9 Omok tinygrad checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/omok_quick.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--human-color", type=str, default="white", choices=["black", "white"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render-scale",
        type=int,
        default=DEFAULT_RENDER_SCALE,
        choices=range(1, MAX_RENDER_SCALE + 1),
        metavar=f"1-{MAX_RENDER_SCALE}",
        help="Internal supersampling scale for sharper text and stones.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.render_scale != DEFAULT_RENDER_SCALE:
        print("[gui] --render-scale is deprecated; HiDPI is handled automatically via pygame.SCALED")
    human_color = 1 if args.human_color == "black" else -1
    app = OmokGUI(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device_name=args.device,
        simulations=args.simulations,
        human_color=human_color,
        seed=args.seed,
        render_scale=args.render_scale,
    )
    app.run()


if __name__ == "__main__":
    main()
