from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Literal

from .game import Card, GameState, LostCitiesConfig, build_deck, score_expedition, tier_config

BackendName = Literal["python", "rust"]
LOGGER = logging.getLogger("coolrl.lost_cities.pvp")


COLOR_PALETTE = [
    (255, 78, 83),
    (83, 141, 244),
    (116, 174, 82),
    (232, 181, 59),
    (164, 99, 202),
    (67, 205, 196),
    (255, 139, 66),
    (220, 92, 170),
]

BG = (0, 0, 0)
CARD_BG = (3, 4, 5)
LINE = (58, 58, 64)
LINE_DARK = (35, 35, 39)
TEXT = (222, 222, 228)
MUTED = (158, 155, 160)
GOLD = (232, 178, 0)
B612_FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "omok"
    / "assets"
    / "fonts"
    / "B612-Regular.ttf"
)

COLOR_NAME_PALETTE = ["Red", "Blue", "Green", "Gold", "Violet", "Cyan", "Orange", "Rose"]


@dataclass
class Snapshot:
    config: LostCitiesConfig
    deck: list[Card]
    hands: list[list[Card]]
    expeditions: list[list[list[Card]]]
    discards: list[list[Card]]
    current_player: int
    phase: str
    pending_discarded_color: int | None
    turn_count: int
    terminal: bool
    legal_mask: list[bool]

    @property
    def card_action_size(self) -> int:
        return self.config.card_action_size

    @property
    def draw_action_size(self) -> int:
        return self.config.draw_action_size

    def expedition_score(self, player: int, color: int) -> int:
        return score_expedition(self.expeditions[player][color], self.config)

    def total_score(self, player: int) -> int:
        return sum(
            self.expedition_score(player, color)
            for color in range(self.config.n_colors)
        )

    def score_diff(self, player: int = 0) -> int:
        return self.total_score(player) - self.total_score(1 - player)


@dataclass(frozen=True)
class ActionTarget:
    rect: Any
    action_id: int
    label: str


class GameBackend:
    name: BackendName

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        self.config = config
        self.seed = seed

    def snapshot(self) -> Snapshot:
        raise NotImplementedError

    def apply(self, action_id: int) -> None:
        raise NotImplementedError

    def can_undo(self) -> bool:
        raise NotImplementedError

    def undo(self) -> bool:
        raise NotImplementedError


class PythonBackend(GameBackend):
    name: BackendName = "python"

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        super().__init__(config, seed)
        self.state = GameState.new_game(config, seed=seed)
        self.history: list[GameState] = []
        LOGGER.debug("파이썬 백엔드 초기화: %s", snapshot_summary(self.snapshot()))

    def snapshot(self) -> Snapshot:
        return _snapshot_from_state(self.state)

    def apply(self, action_id: int) -> None:
        before = self.snapshot()
        self.history.append(self.state.clone())
        self.state.apply_unified_action(action_id)
        LOGGER.debug(
            "파이썬 액션 적용: 액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            action_id,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.history),
        )

    def can_undo(self) -> bool:
        return bool(self.history)

    def undo(self) -> bool:
        if not self.history:
            LOGGER.debug("파이썬 되돌리기 무시: 기록이 비어 있음")
            return False
        before = self.snapshot()
        self.state = self.history.pop()
        LOGGER.debug(
            "파이썬 되돌리기: 이전={%s} 이후={%s} 되돌리기깊이=%s",
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.history),
        )
        return True


class RustBackend(GameBackend):
    name: BackendName = "rust"

    def __init__(self, config: LostCitiesConfig, seed: int | None):
        super().__init__(config, seed)
        self.initial_deck = _shuffled_deck(config, seed)
        self.actions: list[int] = []
        self._snapshot = self._run_trace()
        LOGGER.debug("러스트 백엔드 초기화: %s", snapshot_summary(self.snapshot()))

    def snapshot(self) -> Snapshot:
        return self._snapshot

    def apply(self, action_id: int) -> None:
        before = self.snapshot()
        self.actions.append(action_id)
        try:
            self._snapshot = self._run_trace()
        except Exception:
            self.actions.pop()
            raise
        LOGGER.debug(
            "러스트 액션 적용: 액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            action_id,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.actions),
        )

    def can_undo(self) -> bool:
        return bool(self.actions)

    def undo(self) -> bool:
        if not self.actions:
            LOGGER.debug("러스트 되돌리기 무시: 액션 기록이 비어 있음")
            return False
        before = self.snapshot()
        removed = self.actions.pop()
        self._snapshot = self._run_trace()
        LOGGER.debug(
            "러스트 되돌리기: 제거한액션=%s 이전={%s} 이후={%s} 되돌리기깊이=%s",
            removed,
            snapshot_summary(before),
            snapshot_summary(self.snapshot()),
            len(self.actions),
        )
        return True

    def _run_trace(self) -> Snapshot:
        fixture = {
            "config": self.config.to_snapshot(),
            "initial_deck": [card.to_snapshot() for card in self.initial_deck],
            "steps": [{"action": None}]
            + [{"action": action} for action in self.actions],
        }
        rust_core = Path(__file__).resolve().parent / "rust_core"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(fixture, handle)
            fixture_path = Path(handle.name)
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--bin",
                    "lost_cities_probe",
                    "--",
                    "trace",
                    str(fixture_path),
                ],
                cwd=rust_core,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(f"rust backend failed: {message}") from exc
        finally:
            fixture_path.unlink(missing_ok=True)

        trace = json.loads(result.stdout)
        return _snapshot_from_trace(trace["config"], trace["steps"][-1])


def build_backend(
    backend: BackendName,
    config: LostCitiesConfig,
    seed: int | None,
) -> GameBackend:
    if backend == "python":
        return PythonBackend(config, seed)
    if backend == "rust":
        return RustBackend(config, seed)
    raise ValueError(f"unknown backend: {backend}")


def _shuffled_deck(config: LostCitiesConfig, seed: int | None) -> list[Card]:
    deck = build_deck(config)
    rng = random.Random(config.seed if seed is None else seed)
    rng.shuffle(deck)
    return deck


def _snapshot_from_state(state: GameState) -> Snapshot:
    return Snapshot(
        config=state.config,
        deck=list(state.deck),
        hands=[list(hand) for hand in state.hands],
        expeditions=[
            [list(expedition) for expedition in player_expeditions]
            for player_expeditions in state.expeditions
        ],
        discards=[list(discard) for discard in state.discards],
        current_player=state.current_player,
        phase=state.phase,
        pending_discarded_color=state.pending_discarded_color,
        turn_count=state.turn_count,
        terminal=state.terminal,
        legal_mask=state.unified_legal_mask(),
    )


def _snapshot_from_trace(config_data: dict[str, Any], step: dict[str, Any]) -> Snapshot:
    config = LostCitiesConfig(**config_data)
    return Snapshot(
        config=config,
        deck=_cards_from_json(step["deck"]),
        hands=[_cards_from_json(hand) for hand in step["hands"]],
        expeditions=[
            [_cards_from_json(expedition) for expedition in player_expeditions]
            for player_expeditions in step["expeditions"]
        ],
        discards=[_cards_from_json(discard) for discard in step["discards"]],
        current_player=int(step["current_player"]),
        phase=str(step["phase"]),
        pending_discarded_color=step.get("pending_discarded_color"),
        turn_count=int(step["turn_count"]),
        terminal=bool(step["terminal"]),
        legal_mask=list(step["legal_mask"]),
    )


def _cards_from_json(cards: list[dict[str, int]]) -> list[Card]:
    return [Card.from_snapshot(card) for card in cards]


def color_rgb(color: int) -> tuple[int, int, int]:
    return COLOR_PALETTE[color % len(COLOR_PALETTE)]


def color_name(color: int) -> str:
    if color < len(COLOR_NAME_PALETTE):
        return COLOR_NAME_PALETTE[color]
    return f"Color {color}"


def card_value_label(card: Card, config: LostCitiesConfig) -> str:
    if card.is_handshake:
        return "H"
    return str(card.numeric_value(config.min_rank))


def card_label(card: Card, config: LostCitiesConfig) -> str:
    return f"{color_name(card.color)} {card_value_label(card, config)}"


def snapshot_summary(snapshot: Snapshot) -> str:
    scores = [snapshot.total_score(0), snapshot.total_score(1)]
    hand_sizes = [len(hand) for hand in snapshot.hands]
    discard_sizes = [len(discard) for discard in snapshot.discards]
    phase = "카드" if snapshot.phase == "card" else "뽑기"
    return (
        f"플레이어={snapshot.current_player} 단계={phase} "
        f"턴={snapshot.turn_count} 종료={snapshot.terminal} "
        f"덱={len(snapshot.deck)} 손패수={hand_sizes} 점수={scores} "
        f"직전버린색={snapshot.pending_discarded_color} "
        f"버린더미수={discard_sizes}"
    )


def turn_identity_summary(identity: tuple[int, str, int] | None) -> str:
    if identity is None:
        return "없음"
    player, phase, turn = identity
    phase_name = "카드" if phase == "card" else "뽑기"
    return f"플레이어={player} 단계={phase_name} 턴={turn}"


def configure_debug_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(logging.DEBUG)
    LOGGER.setLevel(logging.DEBUG)


def preferred_font_path() -> Path | None:
    if B612_FONT_PATH.exists():
        return B612_FONT_PATH
    LOGGER.debug("B612 폰트 없음, pygame 대체 폰트 사용: %s", B612_FONT_PATH)
    return None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Lost Cities PVP in one pygame window.")
    parser.add_argument("--backend", choices=("python", "rust"), default="python")
    parser.add_argument("--tier", choices=("tier0", "tier1", "tier2", "tier3"), default="tier3")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=964)
    return parser


class PvpApp:
    def __init__(
        self,
        *,
        backend_name: BackendName,
        tier_name: str = "tier3",
        seed: int | None = None,
        width: int = 1536,
        height: int = 964,
    ):
        import pygame
        import pygame_gui

        configure_debug_logging()
        pygame.init()
        pygame.display.set_caption("COOLRL LOST CITIES PVP")
        self.pygame = pygame
        self.pygame_gui = pygame_gui
        self.window_size = (width, height)
        self.screen = pygame.display.set_mode(self.window_size)
        self.font_path = preferred_font_path()
        self.font_cache: dict[tuple[int, bool], Any] = {}
        theme_path = Path(__file__).resolve().parent / "assets" / "pygame_pvp_theme.json"
        self.manager = pygame_gui.UIManager(self.window_size)
        self._configure_ui_theme(theme_path)
        self.clock = pygame.time.Clock()

        self.seed = seed
        self.tier_name = tier_name
        self.selected_backend: BackendName = backend_name
        self.config = tier_config(tier_name)
        self.backend = build_backend(self.selected_backend, self.config, self.seed)
        self.ui_elements: list[Any] = []
        self.hand_card_rects: dict[int, Any] = {}
        self.board_targets: list[ActionTarget] = []
        self.backend_dropdown: Any = None
        self.tier_dropdown: Any = None
        self.new_game_button: Any = None
        self.undo_button: Any = None
        self.selected_card_slot: int | None = None
        self.error_text: str | None = None
        self.last_turn_identity: tuple[int, str, int] | None = None
        self.turn_flash_until_ms = 0
        self.rebuild_ui()
        LOGGER.debug(
            "PVP 앱 초기화: 백엔드=%s 티어=%s 시드=%s 크기=%sx%s 색상수=%s 손패=%s 덱=%s",
            self.selected_backend,
            self.tier_name,
            self.seed,
            width,
            height,
            self.config.n_colors,
            self.config.hand_size,
            self.config.deck_size,
        )

    def _configure_ui_theme(self, theme_path: Path) -> None:
        with open(theme_path, "r", encoding="utf-8") as handle:
            theme = json.load(handle)
        if self.font_path is not None:
            font_path = str(self.font_path)
            self.manager.add_font_paths("b612", font_path)
            self.manager.preload_fonts(
                [
                    {"name": "b612", "point_size": size}
                    for size in (14, 16, 18, 20)
                ]
            )
            LOGGER.debug("B612 폰트 사용: %s", self.font_path)
        else:
            LOGGER.debug("pygame_gui 기본 대체 폰트 사용")
        self.manager.get_theme().update_theming(theme)

    def run(self) -> None:
        pygame = self.pygame
        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    LOGGER.debug("pygame 종료 이벤트")
                    running = False
                    continue
                self.manager.process_events(event)
                self.handle_ui_event(event)

            self.manager.update(time_delta)
            self.draw()
            self.manager.draw_ui(self.screen)
            pygame.display.flip()
        pygame.quit()

    def handle_ui_event(self, event: Any) -> None:
        pygame_gui = self.pygame_gui
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.new_game_button:
                LOGGER.debug("새 게임 버튼 클릭")
                self.reset_game()
                return
            if event.ui_element == self.undo_button:
                LOGGER.debug("되돌리기 버튼 클릭")
                self.undo()
                return
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.backend_dropdown:
                LOGGER.debug("백엔드 변경: %s -> %s", self.selected_backend, event.text)
                self.selected_backend = event.text
                self.reset_game()
            elif event.ui_element == self.tier_dropdown:
                LOGGER.debug("티어 변경: %s -> %s", self.tier_name, event.text)
                self.tier_name = event.text
                self.config = tier_config(self.tier_name)
                self.reset_game()
        elif event.type == self.pygame.KEYDOWN:
            if event.key == self.pygame.K_z and (event.mod & self.pygame.KMOD_CTRL):
                LOGGER.debug("컨트롤+Z 입력")
                self.undo()
        elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
            LOGGER.debug("마우스 클릭: 위치=%s", event.pos)
            self.handle_board_click(event.pos)

    def handle_board_click(self, pos: tuple[int, int]) -> None:
        snapshot = self.backend.snapshot()
        if snapshot.terminal:
            LOGGER.debug("보드 클릭 무시: 종료 상태 위치=%s 상태={%s}", pos, snapshot_summary(snapshot))
            return
        for target in self.board_targets:
            if target.rect.collidepoint(pos):
                LOGGER.debug(
                    "타겟 클릭: 라벨=%s 액션=%s 위치=%s 상태={%s}",
                    target.label,
                    target.action_id,
                    pos,
                    snapshot_summary(snapshot),
                )
                self.apply_action(target.action_id)
                return
        if snapshot.phase != "card":
            LOGGER.debug("보드 클릭 무시: 카드 단계 아님 위치=%s 상태={%s}", pos, snapshot_summary(snapshot))
            return
        for slot, rect in self.hand_card_rects.items():
            if rect.collidepoint(pos):
                self.selected_card_slot = slot
                card = snapshot.hands[snapshot.current_player][slot]
                LOGGER.debug(
                    "카드 선택: 슬롯=%s 카드=%s 상태={%s}",
                    slot,
                    card_label(card, snapshot.config),
                    snapshot_summary(snapshot),
                )
                return
        LOGGER.debug("보드 클릭: 대상 없음 위치=%s 상태={%s}", pos, snapshot_summary(snapshot))

    def reset_game(self) -> None:
        self.selected_card_slot = None
        self.hand_card_rects = {}
        self.board_targets = []
        self.last_turn_identity = None
        self.turn_flash_until_ms = 0
        try:
            self.backend = build_backend(self.selected_backend, self.config, self.seed)
            self.error_text = None
            LOGGER.debug(
                "게임 초기화 완료: 백엔드=%s 시드=%s 상태={%s}",
                self.selected_backend,
                self.seed,
                snapshot_summary(self.backend.snapshot()),
            )
        except Exception as exc:
            self.error_text = str(exc)
            LOGGER.exception("게임 초기화 실패: 백엔드=%s 시드=%s", self.selected_backend, self.seed)
        self.rebuild_ui()

    def apply_action(self, action_id: int) -> None:
        try:
            before = self.backend.snapshot()
            LOGGER.debug("액션 적용 요청: 액션=%s 상태={%s}", action_id, snapshot_summary(before))
            self.backend.apply(action_id)
            self.selected_card_slot = None
            self.hand_card_rects = {}
            self.board_targets = []
            self.error_text = None
            LOGGER.debug(
                "액션 적용 완료: 액션=%s 상태={%s}",
                action_id,
                snapshot_summary(self.backend.snapshot()),
            )
        except Exception as exc:
            self.error_text = str(exc)
            LOGGER.exception("액션 적용 실패: 액션=%s", action_id)
        self.rebuild_ui()

    def undo(self) -> None:
        try:
            before = self.backend.snapshot()
            changed = self.backend.undo()
            self.selected_card_slot = None
            self.hand_card_rects = {}
            self.board_targets = []
            self.error_text = None if changed else "되돌릴 수 없음"
            LOGGER.debug(
                "되돌리기 요청: 변경됨=%s 이전={%s} 이후={%s}",
                changed,
                snapshot_summary(before),
                snapshot_summary(self.backend.snapshot()),
            )
        except Exception as exc:
            self.error_text = str(exc)
            LOGGER.exception("되돌리기 실패")
        self.rebuild_ui()

    def _hand_layout(self, snapshot: Snapshot, player: int, y: int) -> tuple[int, int, int, int, int]:
        width, _ = self.window_size
        status_x = width - 404
        card_x = 340
        available = max(360, status_x - card_x - 40)
        count = max(1, snapshot.config.hand_size)
        gap = 14
        card_w = min(126, max(54, (available - gap * (count - 1)) // count))
        card_h = max(70, int(card_w * 1.22))
        if count > 1:
            gap = max(8, min(18, (available - card_w * count) // (count - 1)))
        card_y = y + (14 if player == 1 else 28)
        return card_x, card_y, card_w, card_h, gap

    def rebuild_ui(self) -> None:
        pygame = self.pygame
        pygame_gui = self.pygame_gui
        width, _ = self.window_size
        for element in self.ui_elements:
            element.kill()
        self.ui_elements = []

        self.backend_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["python", "rust"],
            starting_option=self.selected_backend,
            relative_rect=pygame.Rect(130, 17, 174, 46),
            manager=self.manager,
        )
        self.tier_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["tier0", "tier1", "tier2", "tier3"],
            starting_option=self.tier_name,
            relative_rect=pygame.Rect(388, 17, 118, 46),
            manager=self.manager,
        )
        self.new_game_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(526, 17, 148, 46),
            text="NEW GAME",
            manager=self.manager,
        )
        self.undo_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(width - 248, 17, 110, 46),
            text="UNDO",
            manager=self.manager,
        )
        if not self.backend.can_undo():
            self.undo_button.disable()
        self.ui_elements.extend(
            [self.backend_dropdown, self.tier_dropdown, self.new_game_button, self.undo_button]
        )

    def draw(self) -> None:
        pygame = self.pygame
        snapshot = self.backend.snapshot()
        self._sync_turn_identity(snapshot)
        self.hand_card_rects = {}
        self.board_targets = []
        self.screen.fill(BG)
        self._draw_header(snapshot)
        self._draw_player_area(snapshot, player=1, y=110)
        self._draw_center(snapshot)
        self._draw_player_area(snapshot, player=0, y=self.window_size[1] - 257)
        if self.error_text:
            self._draw_text(self.error_text.upper(), (740, 50), color_rgb(0), 18, bold=True)

    def _draw_header(self, snapshot: Snapshot) -> None:
        pygame = self.pygame
        width, _ = self.window_size
        compact = width < 1450
        pygame.draw.line(self.screen, LINE, (0, 0), (width, 0), 1)
        pygame.draw.line(self.screen, LINE, (0, 90), (width, 90), 1)
        pygame.draw.line(self.screen, LINE, (700, 0), (700, 90), 1)
        self._draw_text("BACKEND", (31, 31), MUTED, 18)
        self._draw_text("TIER", (322, 31), MUTED, 18)
        if snapshot.terminal:
            diff = snapshot.score_diff(0)
            if diff > 0:
                status = "PLAYER 0 WINS"
            elif diff < 0:
                status = "PLAYER 1 WINS"
            else:
                status = "DRAW"
            detail = "START A NEW GAME TO PLAY AGAIN."
        elif snapshot.phase == "card":
            card = self._selected_card(snapshot)
            if card is None:
                status = (
                    f"P{snapshot.current_player}: CHOOSE CARD"
                    if compact
                    else f"PLAYER {snapshot.current_player}: CHOOSE A CARD"
                )
                detail = "Click active hand." if compact else "Click a card in the active hand."
            else:
                status = (
                    f"P{snapshot.current_player}: CHOOSE TARGET"
                    if compact
                    else f"PLAYER {snapshot.current_player}: CHOOSE CARD DESTINATION"
                )
                detail = (
                    f"{card_label(card, snapshot.config)} selected."
                    if compact
                    else f"{card_label(card, snapshot.config)} selected. Click expedition or discard."
                )
        else:
            status = (
                f"P{snapshot.current_player}: DRAW"
                if compact
                else f"PLAYER {snapshot.current_player}: DRAW A CARD"
            )
            detail = "Click deck/discard." if compact else "Click the deck or a highlighted discard pile."
        title_size = 30 if not compact else 24
        detail_size = 18 if not compact else 14
        title_x = 720
        self._draw_text(status, (title_x, 21), TEXT, title_size, bold=True)
        self._draw_text(detail, (title_x, 56), MUTED, detail_size)
        if width >= 1450:
            meta = f"BACKEND: {self.selected_backend}   TURN: {snapshot.turn_count}"
            meta_x = width - 465
        else:
            meta = f"TURN: {snapshot.turn_count}"
            meta_x = width - 135
        self._draw_text(meta, (meta_x, 56), MUTED, detail_size)

    def _draw_player_area(self, snapshot: Snapshot, *, player: int, y: int) -> None:
        pygame = self.pygame
        width, _ = self.window_size
        active = not snapshot.terminal and snapshot.current_player == player
        border = GOLD if active else LINE
        panel = pygame.Rect(26, y, width - 52, 182 if player == 1 else 210)
        border_width = 4 if active and self._turn_flash_active(player) else 2 if active else 1
        self._draw_panel_rect(panel, border, width=border_width)
        self._draw_text(f"PLAYER {player}", (48, y + 23), TEXT, 28)
        self._draw_text(f"SCORE {snapshot.total_score(player)}", (204, y + 29), TEXT, 18)
        self._draw_text("HAND", (80, y + 76), MUTED, 18)
        pygame.draw.line(self.screen, LINE, (44, y + 77), (67, y + 77), 1)
        pygame.draw.line(self.screen, LINE, (49, y + 72), (49, y + 87), 1)

        card_x, card_y, card_w, card_h, card_gap = self._hand_layout(snapshot, player, y)

        for slot, card in enumerate(snapshot.hands[player]):
            selectable = active and snapshot.phase == "card"
            selected = selectable and slot == self.selected_card_slot
            rect = self._draw_card(
                card,
                (card_x + slot * (card_w + card_gap), card_y),
                snapshot.config,
                large=True,
                size=(card_w, card_h),
                selected=selected,
                selectable=selectable,
            )
            if selectable:
                self.hand_card_rects[slot] = rect

        if active:
            status_box = pygame.Rect(width - 404, y + (45 if player == 0 else 42), 330, 92)
            pygame.draw.rect(self.screen, BG, status_box)
            pygame.draw.rect(self.screen, LINE, status_box, width=1)
            self._draw_text("ACTIVE", (status_box.x + 22, status_box.y + 22), GOLD, 20)
            if snapshot.phase == "card":
                prompt = (
                    "Yellow border: expedition or discard"
                    if self.selected_card_slot is not None
                    else "Select card, then click a target"
                )
            else:
                prompt = "Click deck or discard"
            self._draw_text(prompt, (status_box.x + 22, status_box.y + 56), MUTED, 16)

    def _sync_turn_identity(self, snapshot: Snapshot) -> None:
        identity = (snapshot.current_player, snapshot.phase, snapshot.turn_count)
        if identity == self.last_turn_identity:
            return
        previous = self.last_turn_identity
        self.last_turn_identity = identity
        if snapshot.phase == "card" and not snapshot.terminal:
            self.turn_flash_until_ms = self.pygame.time.get_ticks() + 1200
            LOGGER.debug(
                "턴 전환 감지: 이전=%s 현재플레이어=%s 단계=%s 턴=%s",
                turn_identity_summary(previous),
                snapshot.current_player,
                "카드" if snapshot.phase == "card" else "뽑기",
                snapshot.turn_count,
            )

    def _turn_flash_active(self, player: int) -> bool:
        snapshot = self.backend.snapshot()
        return (
            snapshot.current_player == player
            and snapshot.phase == "card"
            and self.pygame.time.get_ticks() < self.turn_flash_until_ms
        )

    def _draw_center(self, snapshot: Snapshot) -> None:
        pygame = self.pygame
        width, height = self.window_size
        bottom_panel_y = height - 257
        board_y = 313
        board_h = max(220, bottom_panel_y - board_y - 20)
        board_x = 263 if width >= 1450 else 196
        board_rect = pygame.Rect(board_x, board_y, width - board_x - 30, board_h)
        pygame.draw.rect(self.screen, BG, board_rect)
        pygame.draw.rect(self.screen, LINE, board_rect, width=1)

        deck_h = min(264, board_h - 44)
        deck_rect = pygame.Rect(44, board_y + 44, 173, deck_h)
        deck_action = snapshot.card_action_size
        deck_target = (
            snapshot.phase == "draw"
            and self._is_legal(snapshot, deck_action)
        )
        self._draw_deck(snapshot, deck_rect, target=deck_target)
        if deck_target:
            self._register_target(deck_rect, deck_action, "덱에서 뽑기")

        selected_card = self._selected_card(snapshot)
        lane_count = snapshot.config.n_colors
        lane_gap = 14
        lane_x = board_rect.x + 26
        lane_y = board_rect.y + 63
        lane_w = (board_rect.width - 52 - lane_gap * (lane_count - 1)) // lane_count
        zone_h = max(42, (board_rect.height - 86 - 24) // 3)

        for color in range(lane_count):
            x = lane_x + color * (lane_w + lane_gap)
            lane_rect = pygame.Rect(x, board_rect.y, lane_w, board_rect.height)
            lane_color = color_rgb(color)
            if color > 0:
                pygame.draw.line(self.screen, LINE, (x - 8, board_rect.y), (x - 8, board_rect.bottom), 1)
            self._draw_text(
                color_name(color).upper(),
                (x + 12, board_rect.y + 24),
                lane_color,
                23,
                bold=True,
            )
            pygame.draw.line(
                self.screen,
                lane_color,
                (x + 12, board_rect.y + 51),
                (x + lane_w - 12, board_rect.y + 51),
                2,
            )

            p1_rect = pygame.Rect(x + 26, lane_y, lane_w - 52, zone_h)
            discard_rect = pygame.Rect(x + 26, lane_y + zone_h + 12, lane_w - 52, zone_h)
            p0_rect = pygame.Rect(x + 26, lane_y + 2 * (zone_h + 12), lane_w - 52, zone_h)

            play_target_player = None
            play_action = None
            discard_action = None
            if snapshot.phase == "card" and selected_card and selected_card.color == color:
                play_target_player = snapshot.current_player
                play_action = 2 * (self.selected_card_slot or 0)
                discard_action = play_action + 1

            for player, rect in ((1, p1_rect), (0, p0_rect)):
                action = play_action if play_target_player == player else None
                is_target = action is not None and self._is_legal(snapshot, action)
                self._draw_zone(
                    rect,
                    f"P{player} expedition",
                    f"Score {snapshot.expedition_score(player, color)}",
                    snapshot.expeditions[player][color],
                    snapshot.config,
                    target=is_target,
                    target_label="Play here" if is_target else None,
                )
                if is_target and action is not None:
                    self._register_target(rect, action, "탐험대에 놓기")

            draw_discard_action = snapshot.card_action_size + 1 + color
            discard_target = False
            discard_label = None
            target_action = None
            if (
                snapshot.phase == "card"
                and selected_card
                and selected_card.color == color
                and discard_action is not None
                and self._is_legal(snapshot, discard_action)
            ):
                discard_target = True
                discard_label = "Discard here"
                target_action = discard_action
            elif snapshot.phase == "draw" and self._is_legal(snapshot, draw_discard_action):
                discard_target = True
                discard_label = "Draw"
                target_action = draw_discard_action

            self._draw_zone(
                discard_rect,
                "Discard",
                "Stack",
                snapshot.discards[color],
                snapshot.config,
                target=discard_target,
                target_label=discard_label,
            )
            if discard_target and target_action is not None:
                target_label = "버린 더미에 버리기" if discard_label == "Discard here" else "버린 더미에서 뽑기"
                self._register_target(discard_rect, target_action, target_label)

    def _draw_deck(self, snapshot: Snapshot, rect: Any, *, target: bool = False) -> None:
        pygame = self.pygame
        border = GOLD if target else LINE
        pygame.draw.rect(self.screen, BG, rect)
        pygame.draw.rect(self.screen, border, rect, width=2 if target else 1)
        compact = rect.height < 230
        self._draw_text_center("DECK", pygame.Rect(rect.x, rect.y + 18, rect.width, 36), TEXT, 20 if compact else 22)
        card_w, card_h = (42, 58) if compact else (52, 72)
        card_back = pygame.Rect(rect.centerx - card_w // 2 - 4, rect.y + (68 if compact else 72), card_w, card_h)
        pygame.draw.rect(self.screen, BG, card_back)
        pygame.draw.rect(self.screen, TEXT, card_back, width=2)
        pygame.draw.rect(self.screen, LINE, card_back.move(8, 8), width=2)
        count_rect = (
            pygame.Rect(rect.x, card_back.bottom + 8, rect.width, 40)
            if compact
            else pygame.Rect(rect.x, rect.bottom - 92, rect.width, 64)
        )
        self._draw_text_center(str(len(snapshot.deck)), count_rect, TEXT, 36 if compact else 42, bold=True)
        if target:
            self._draw_text_center("DRAW", pygame.Rect(rect.x, rect.bottom - 34, rect.width, 24), GOLD, 15 if compact else 17)

    def _draw_zone(
        self,
        rect: Any,
        title: str,
        subtitle: str,
        cards: list[Card],
        config: LostCitiesConfig,
        *,
        target: bool = False,
        target_label: str | None = None,
    ) -> None:
        pygame = self.pygame
        border = GOLD if target else LINE
        pygame.draw.rect(self.screen, BG, rect)
        pygame.draw.rect(self.screen, border, width=2 if target else 1, rect=rect)
        if rect.height < 58:
            if target and target_label:
                self._draw_target_badge(rect, target_label)
            if not cards:
                self._draw_text(self._zone_title(title, rect.width), (rect.x + 18, rect.y + 12), TEXT, 14)
                self._draw_text("EMPTY", (rect.right - 56, rect.y + 13), MUTED, 12)
            else:
                self._draw_mini_card_row(
                    cards,
                    pygame.Rect(rect.x + 8, rect.y + 9, rect.width - 16, 24),
                    config,
                    preferred_size=24,
                    min_size=14,
                )
            return
        title_text = self._zone_title(title, rect.width)
        self._draw_text(title_text, (rect.x + 20, rect.y + 16), TEXT, 17 if rect.width < 220 else 18)
        if target and target_label:
            self._draw_target_badge(rect, target_label)
        if not cards:
            self._draw_text(subtitle, (rect.x + 20, rect.y + 51), MUTED, 15)
            self._draw_text("EMPTY", (rect.right - 72, rect.y + 51), MUTED, 15)
            return

        # Once cards exist, reserve the bottom band exclusively for cards.
        # This prevents multi-card expeditions from covering score/top-card text.
        self._draw_text_right(subtitle, (rect.right - 18, rect.y + 20), MUTED, 12 if rect.width < 220 else 13)
        row_size = min(30, max(18, rect.height - 70))
        row_rect = pygame.Rect(rect.x + 12, rect.bottom - row_size - 10, rect.width - 24, row_size)
        self._draw_mini_card_row(cards, row_rect, config, preferred_size=row_size)

    def _zone_title(self, title: str, width: int) -> str:
        text = title.upper()
        if width >= 220:
            return text
        return text.replace(" EXPEDITION", " EXP")

    def _mini_card_layout(
        self,
        width: int,
        count: int,
        *,
        preferred_size: int = 30,
        min_size: int = 16,
    ) -> tuple[int, int]:
        gap = 5
        usable = max(32, width)
        if count <= 1:
            return min(preferred_size, usable), gap

        size = min(preferred_size, (usable - gap * (count - 1)) // count)
        if size >= min_size:
            return size, gap

        # Keep cards readable before shrinking too hard: negative gap means overlap.
        size = min(preferred_size, max(min_size, usable // min(count, 5)))
        gap = (usable - size * count) // (count - 1)
        return size, min(5, gap)

    def _draw_mini_card_row(
        self,
        cards: list[Card],
        row_rect: Any,
        config: LostCitiesConfig,
        *,
        preferred_size: int = 30,
        min_size: int = 16,
    ) -> None:
        if not cards:
            return
        size, gap = self._mini_card_layout(
            row_rect.width,
            len(cards),
            preferred_size=preferred_size,
            min_size=min_size,
        )
        step = size + gap
        total_width = size + max(0, len(cards) - 1) * step
        start_x = row_rect.x + max(0, (row_rect.width - total_width) // 2)
        y = row_rect.y + max(0, (row_rect.height - size) // 2)
        for index, card in enumerate(cards):
            self._draw_mini_card(card, (start_x + index * step, y), config, size=size)

    def _draw_target_badge(self, rect: Any, label: str) -> None:
        if rect.width < 220:
            return
        pygame = self.pygame
        text = label.upper()
        font = self._font(12)
        surface = font.render(text, True, GOLD)
        badge = surface.get_rect()
        badge.width += 12
        badge.height += 6
        badge.topright = (rect.right - 8, rect.y - 11)
        pygame.draw.rect(self.screen, BG, badge)
        pygame.draw.rect(self.screen, GOLD, badge, width=1)
        self.screen.blit(surface, (badge.x + 6, badge.y + 3))

    def _draw_card(
        self,
        card: Card,
        pos: tuple[int, int],
        config: LostCitiesConfig,
        *,
        large: bool = False,
        size: tuple[int, int] | None = None,
        selected: bool = False,
        selectable: bool = False,
    ) -> Any:
        pygame = self.pygame
        x, y = pos
        width, height = size or ((126, 154) if large else (30, 30))
        color = color_rgb(card.color)
        rect = pygame.Rect(x, y, width, height)
        if selected:
            pygame.draw.rect(
                self.screen,
                GOLD,
                rect.inflate(10, 10),
                width=2,
            )
        pygame.draw.rect(self.screen, CARD_BG, rect)
        pygame.draw.rect(self.screen, color, rect, width=2)
        if large:
            label_size = max(10, min(19, width // 6))
            value_size = max(26, min(62, int(width * 0.48)))
            self._draw_text(color_name(card.color).upper(), (x + 10, y + 10), color, label_size)
            self._draw_text_center(
                card_value_label(card, config),
                pygame.Rect(x, y + max(24, height // 4), width, height - max(32, height // 3)),
                color,
                value_size,
                bold=True,
            )
        else:
            self._draw_text_center(
                card_value_label(card, config),
                rect,
                color,
                16,
                bold=True,
            )
        return rect

    def _draw_mini_card(
        self,
        card: Card,
        pos: tuple[int, int],
        config: LostCitiesConfig,
        *,
        size: int = 30,
    ) -> None:
        pygame = self.pygame
        x, y = pos
        rect = pygame.Rect(x, y, size, size)
        color = color_rgb(card.color)
        pygame.draw.rect(self.screen, CARD_BG, rect)
        pygame.draw.rect(self.screen, color, rect, width=1)
        self._draw_text_center(card_value_label(card, config), rect, color, max(11, size // 2), bold=True)

    def _selected_card(self, snapshot: Snapshot) -> Card | None:
        if snapshot.phase != "card" or self.selected_card_slot is None:
            return None
        hand = snapshot.hands[snapshot.current_player]
        if self.selected_card_slot >= len(hand):
            self.selected_card_slot = None
            return None
        return hand[self.selected_card_slot]

    def _is_legal(self, snapshot: Snapshot, action_id: int) -> bool:
        return 0 <= action_id < len(snapshot.legal_mask) and snapshot.legal_mask[action_id]

    def _register_target(self, rect: Any, action_id: int, label: str) -> None:
        self.board_targets.append(ActionTarget(rect=rect, action_id=action_id, label=label))

    def _draw_panel(self, rect: tuple[int, int, int, int], border: tuple[int, int, int]) -> None:
        pygame = self.pygame
        panel = pygame.Rect(*rect)
        self._draw_panel_rect(panel, border)

    def _draw_panel_rect(
        self,
        panel: Any,
        border: tuple[int, int, int],
        *,
        width: int | None = None,
    ) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, BG, panel)
        border_width = width if width is not None else 2 if border == GOLD else 1
        pygame.draw.rect(self.screen, border, panel, width=border_width)

    def _draw_text(
        self,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
        *,
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold=bold)
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def _draw_text_center(
        self,
        text: str,
        rect: Any,
        color: tuple[int, int, int],
        size: int,
        *,
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold=bold)
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def _draw_text_right(
        self,
        text: str,
        top_right: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
        *,
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold=bold)
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(topright=top_right))

    def _font(self, size: int, *, bold: bool = False) -> Any:
        key = (size, bold)
        if key not in self.font_cache:
            if self.font_path is not None:
                font = self.pygame.font.Font(str(self.font_path), size)
                font.set_bold(bold)
            else:
                font = self.pygame.font.SysFont("dejavusansmono", size, bold=bold)
            self.font_cache[key] = font
        return self.font_cache[key]


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    app = PvpApp(
        backend_name=args.backend,
        tier_name=args.tier,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    app.run()


if __name__ == "__main__":
    main()
