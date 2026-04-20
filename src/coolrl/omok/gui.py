from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "0")

import numpy as np
import pygame

from .board import GameState
from .features import states_to_feature_planes
from .mcts import MCTS
from .openings import sample_balanced_openings


BOARD_SIZE = 9
C_PUCT = 1.0
LEAVES_PER_BATCH = 8

B612_FONT_DIR = Path(__file__).with_name("assets") / "fonts"
B612_REGULAR = B612_FONT_DIR / "B612-Regular.ttf"
B612_BOLD = B612_FONT_DIR / "B612-Bold.ttf"
DEFAULT_WINDOW_SIZE = (1280, 960)
MIN_BOARD_PX = 360
SIDEBAR_WIDTH = 320
COCKPIT_WHITE = (238, 246, 250)
COCKPIT_CYAN = (64, 210, 232)
COCKPIT_GREEN = (96, 232, 148)
COCKPIT_AMBER = (245, 188, 74)
COCKPIT_RED = (255, 108, 92)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _ort_providers(device: str) -> list[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device in ("coreml", "metal", "mps"):
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if device == "auto":
        return ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxEvaluator:
    def __init__(self, model_path: str, device: str) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=_ort_providers(device))
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        features = np.ascontiguousarray(states_to_feature_planes(states))
        logits, values = self.session.run(None, {self.input_name: features})
        return _softmax(logits).astype(np.float32), values.astype(np.float32)


class OmokGUI:
    def __init__(
        self,
        model_path: str | None,
        device: str,
        simulations: int,
        human_color: int = -1,
        seed: int = 0,
    ) -> None:
        self.model_path = model_path
        self.simulations = simulations
        self.evaluator: OnnxEvaluator | None = None
        self.provider = "none"
        if model_path:
            self.evaluator = OnnxEvaluator(model_path, device)
            self.provider = self.evaluator.provider
        self.state = GameState(BOARD_SIZE, False)
        self.current_opening: list[int] = []
        self.position_version = 0
        self.ai_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omok-ai")
        self.ai_future: Future[int] | None = None
        self.ai_future_version = -1
        self.ai_error: str | None = None
        self.human_color = 1 if int(human_color) >= 1 else -1
        self.seed = int(seed)

        pygame.init()
        pygame.display.set_caption("coolrl 9x9 Omok")
        flags = pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.SCALED
        self.screen = pygame.display.set_mode(DEFAULT_WINDOW_SIZE, flags, vsync=1)
        self.clock = pygame.time.Clock()
        self.font = self._load_font(B612_BOLD, 16)
        self.small_font = self._load_font(B612_REGULAR, 13)
        self.sidebar_line_height = 20
        self._update_layout()

    @staticmethod
    def _load_font(path: Path, size: int) -> pygame.font.Font:
        if path.exists():
            return pygame.font.Font(str(path), size)
        return pygame.font.SysFont("DejaVu Sans", size)

    def _update_layout(self) -> None:
        width, height = self.screen.get_size()
        short_side = max(1, min(width, height))
        margin = max(24, min(64, short_side // 16))
        gap = max(28, min(56, width // 24))

        board_space_w = width - margin * 2 - gap - SIDEBAR_WIDTH
        board_space_h = height - margin * 2

        if board_space_w >= MIN_BOARD_PX:
            self.board_px = max(MIN_BOARD_PX, min(board_space_w, board_space_h))
            board_x = margin
            board_y = max(margin, (height - self.board_px) // 2)
            sidebar_x = board_x + self.board_px + gap
            sidebar_y = max(margin, board_y)
            self.sidebar_width = max(220, width - sidebar_x - margin)
        else:
            sidebar_height = 260
            board_space_w = width - margin * 2
            board_space_h = height - margin * 3 - sidebar_height
            self.board_px = max(240, min(board_space_w, board_space_h))
            board_x = max(margin, (width - self.board_px) // 2)
            board_y = margin
            sidebar_x = margin
            sidebar_y = board_y + self.board_px + margin
            self.sidebar_width = max(220, width - margin * 2)

        self.cell = self.board_px / (BOARD_SIZE - 1)
        self.board_origin = (int(board_x), int(board_y))
        self.sidebar_origin = (int(sidebar_x), int(sidebar_y))

    def _reset_state(self, apply_opening: bool) -> None:
        self.state = GameState(BOARD_SIZE, False)
        self.current_opening = []
        if apply_opening:
            rng = random.Random(self.seed)
            openings = sample_balanced_openings(BOARD_SIZE, 32, rng)
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
            w, h = event.size
        else:
            w = getattr(event, "w", getattr(event, "x", self.screen.get_width()))
            h = getattr(event, "h", getattr(event, "y", self.screen.get_height()))
        self.screen = pygame.display.set_mode(
            (max(640, int(w)), max(560, int(h))),
            pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.SCALED,
            vsync=1,
        )
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
        if self.state.terminal or self.evaluator is None:
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
            self.evaluator,
            self.simulations,
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
    def _search_ai_action(state: GameState, evaluator: OnnxEvaluator, simulations: int) -> int:
        search = MCTS(
            c_puct=C_PUCT,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.0,
            evaluator=evaluator,
        )
        result = search.search_batch(
            [state],
            simulations,
            [0.0],
            add_noise=False,
            leaves_per_batch=LEAVES_PER_BATCH,
        )[0]
        return result.action

    def _render(self) -> None:
        self._update_layout()
        self.screen.fill((0, 0, 0))
        self._draw_board()
        self._draw_sidebar()

    def _draw_board(self) -> None:
        ox, oy = self.board_origin
        board_end_x = ox + self.board_px
        board_end_y = oy + self.board_px
        stone_radius = max(18, int(self.cell * 0.32))
        black = (0, 0, 0)
        white = (245, 245, 245)

        for i in range(BOARD_SIZE):
            x = int(ox + i * self.cell)
            y = int(oy + i * self.cell)
            pygame.draw.line(self.screen, white, (ox, y), (board_end_x, y), 1)
            pygame.draw.line(self.screen, white, (x, oy), (x, board_end_y), 1)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                stone = self.state.board[row, col]
                if stone == 0:
                    continue
                x = int(ox + col * self.cell)
                y = int(oy + row * self.cell)
                if stone == 1:
                    pygame.draw.circle(self.screen, black, (x, y), stone_radius)
                    pygame.draw.circle(self.screen, white, (x, y), stone_radius, 3)
                else:
                    pygame.draw.circle(self.screen, white, (x, y), stone_radius)

        if self.state.last_action is not None:
            row, col = divmod(self.state.last_action, BOARD_SIZE)
            x = int(ox + col * self.cell)
            y = int(oy + row * self.cell)
            marker = black if self.state.board[row, col] == -1 else white
            pygame.draw.circle(self.screen, marker, (x, y), 5)

    def _draw_sidebar(self) -> None:
        x, y = self.sidebar_origin
        model_label = Path(self.model_path).name if self.model_path else "no model"
        lines = [
            "Model",
            model_label,
            "",
            f"Provider: {self.provider}",
            f"Human: {'Black' if self.human_color == 1 else 'White'}",
            f"Turn: {'Black' if self.state.to_play == 1 else 'White'}",
            f"Moves: {self.state.move_count}",
            f"Simulations: {self.simulations}",
            "",
        ]
        if self.evaluator is None:
            lines.append("AI: no model loaded")
        elif self.ai_future is not None and not self.ai_future.done():
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
                "M: force AI",
                "Esc: quit",
            ]
        )

        for index, text in enumerate(lines):
            font = self.font if index == 0 or text == "Controls" else self.small_font
            surface = self._render_sidebar_text(font, text, self._sidebar_color(text))
            self.screen.blit(surface, (x, y + index * self.sidebar_line_height))

    def _sidebar_color(self, text: str) -> tuple[int, int, int]:
        if not text:
            return COCKPIT_WHITE
        if text in {"Model", "Controls"}:
            return COCKPIT_CYAN
        if text.startswith("AI error") or text == "AI: no model loaded":
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
        return self._render_truncated_text(font, text, color, self.sidebar_width)

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
        value_surface = self._render_truncated_text(font, value, value_color, surface_width - label_width)
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
        x, y = pos
        tolerance = 24
        if x < ox - tolerance or y < oy - tolerance or x > ox + self.board_px + tolerance or y > oy + self.board_px + tolerance:
            return None
        col = round((x - ox) / self.cell)
        row = round((y - oy) / self.cell)
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return None
        return row * BOARD_SIZE + col


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="coolrl Omok: play against a trained AI on a 9×9 board.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="FILE.onnx",
        help="Path to an ONNX model file (.onnx). If omitted, the board launches "
             "without an AI (useful for two-player local play or testing the UI).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "coreml"],
        help="ONNX Runtime execution provider. 'auto' tries CoreML → CUDA → CPU in order.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=64,
        metavar="N",
        help="Number of MCTS simulations the AI runs per move. "
             "Higher values improve move quality but increase think time.",
    )
    parser.add_argument(
        "--human-color",
        type=str,
        default="white",
        choices=["black", "white"],
        help="Which color the human plays. Black moves first.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Random seed used when sampling opening sequences (O key in-game). "
             "Change with [ / ] keys during play.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    human_color = 1 if args.human_color == "black" else -1
    app = OmokGUI(
        model_path=args.model,
        device=args.device,
        simulations=args.simulations,
        human_color=human_color,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
