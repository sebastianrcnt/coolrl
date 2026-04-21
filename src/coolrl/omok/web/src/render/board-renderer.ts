import type { GameState, Player } from "../core/game-state";
import {
  actionToRowCol,
  cellX,
  cellY,
  computeCellMetrics,
  starPoints,
  type CellMetrics,
} from "./board-geometry";
import { clamp01, easeInCubic, easeOutBack, easeOutCubic } from "./easing";
import {
  drawGlassStone,
  drawPendingGhostStone,
  drawThinkingGhostStone,
} from "./stone-renderer";
import type { ThinkingGhostSlot } from "../ui/thinking-ghosts";
import { STONE_ANIM_MS } from "../ui/stone-animations";

const DEFAULT_BOARD_MARGIN_RATIO = 0.03;

export interface BoardThemeColors {
  line: string;
  star: string;
}

export interface BoardRenderInput {
  ctx: CanvasRenderingContext2D;
  game: GameState;
  metrics: CellMetrics;
  theme: BoardThemeColors;
  pendingAction: number | null;
  humanPlayer: Player;
  isHumansTurn: boolean;
  stoneAnimAt(action: number): number | null;
  ghostSlots: readonly ThinkingGhostSlot[];
  now: number;
}

export function readBoardThemeColors(root: HTMLElement = document.documentElement): BoardThemeColors {
  const style = getComputedStyle(root);
  return {
    line: style.getPropertyValue("--board-line").trim(),
    star: style.getPropertyValue("--board-star").trim(),
  };
}

export function readBoardMarginRatio(root: HTMLElement = document.documentElement): number {
  const value = parseFloat(getComputedStyle(root).getPropertyValue("--board-grid-margin"));
  return Number.isFinite(value) ? value : DEFAULT_BOARD_MARGIN_RATIO;
}

export function makeMetricsForCanvas(
  canvas: HTMLCanvasElement,
  boardSize: number,
  marginRatio: number = DEFAULT_BOARD_MARGIN_RATIO
): CellMetrics {
  return computeCellMetrics(canvas.width, canvas.height, boardSize, marginRatio);
}

export function renderBoard(input: BoardRenderInput): void {
  const { ctx, game, metrics, theme, pendingAction, humanPlayer, isHumansTurn, stoneAnimAt, ghostSlots, now } = input;
  const stoneRadius = metrics.step * 0.44;

  ctx.clearRect(0, 0, metrics.width, metrics.height);

  ctx.strokeStyle = theme.line;
  ctx.lineWidth = Math.max(1, metrics.square * 0.0014);
  for (let i = 0; i < game.boardSize; i++) {
    const px = cellX(metrics, i);
    const py = cellY(metrics, i);
    ctx.beginPath();
    ctx.moveTo(cellX(metrics, 0), py);
    ctx.lineTo(cellX(metrics, game.boardSize - 1), py);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px, cellY(metrics, 0));
    ctx.lineTo(px, cellY(metrics, game.boardSize - 1));
    ctx.stroke();
  }

  ctx.fillStyle = theme.star;
  for (const [r, c] of starPoints(game.boardSize)) {
    ctx.beginPath();
    ctx.arc(cellX(metrics, c), cellY(metrics, r), Math.max(2.2, metrics.step * 0.06), 0, Math.PI * 2);
    ctx.fill();
  }

  for (let i = 0; i < game.actionSize; i++) {
    const v = game.board[i];
    if (v === 0) continue;
    const player = v as Player;
    const { row, col } = actionToRowCol(i, game.boardSize);
    const cx = cellX(metrics, col);
    const cy = cellY(metrics, row);
    const t = stoneAnimAt(i);
    if (t !== null) {
      const scale = 0.35 + easeOutBack(t) * 0.65;
      const alpha = easeOutCubic(Math.min(1, t * 1.4));
      drawGlassStone(ctx, { cx, cy, radius: stoneRadius * scale, player, alpha });
    } else {
      drawGlassStone(ctx, { cx, cy, radius: stoneRadius, player, alpha: 1.0 });
    }
  }

  for (const slot of ghostSlots) {
    if (game.board[slot.action] !== 0) continue;
    const age = now - slot.start;
    if (age < 0 || age >= slot.life) continue;
    const t = age / slot.life;
    const appear = easeOutCubic(clamp01(t / 0.18));
    const disappear = 1 - easeInCubic(clamp01((t - 0.22) / 0.78));
    const bubble = Math.max(0, Math.min(appear, disappear));
    const alpha = Math.min(easeOutCubic(clamp01(t / 0.08)), disappear) * 0.28;
    const scale = 0.24 + bubble * 0.58;
    const bob = -easeOutCubic(t) * 0.05 * stoneRadius;
    const { row, col } = actionToRowCol(slot.action, game.boardSize);
    drawThinkingGhostStone(ctx, {
      cx: cellX(metrics, col),
      cy: cellY(metrics, row) + bob,
      radius: stoneRadius * scale,
      player: slot.player,
      alpha,
      progress: t,
    });
  }

  if (pendingAction !== null && isHumansTurn) {
    const { row, col } = actionToRowCol(pendingAction, game.boardSize);
    drawPendingGhostStone(ctx, {
      cx: cellX(metrics, col),
      cy: cellY(metrics, row),
      radius: stoneRadius,
      player: humanPlayer,
    });
  }

  if (
    game.lastAction !== null &&
    pendingAction === null &&
    stoneAnimAt(game.lastAction) === null
  ) {
    const v = game.board[game.lastAction];
    const { row, col } = actionToRowCol(game.lastAction, game.boardSize);
    ctx.fillStyle = v === 1 ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.75)";
    ctx.beginPath();
    ctx.arc(cellX(metrics, col), cellY(metrics, row), stoneRadius * 0.2, 0, Math.PI * 2);
    ctx.fill();
  }
  // STONE_ANIM_MS reserved for layout consumers; keeping the import alive.
  void STONE_ANIM_MS;
}
