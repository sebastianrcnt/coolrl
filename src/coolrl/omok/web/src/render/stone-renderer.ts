import type { Player } from "../core/game-state";
import { clamp01, easeInCubic, easeOutCubic } from "./easing";

export interface StoneRenderOptions {
  cx: number;
  cy: number;
  radius: number;
  player: Player;
  alpha?: number;
}

export function drawGlassStone(
  ctx: CanvasRenderingContext2D,
  options: StoneRenderOptions
): void {
  const { cx, cy, radius: r, player, alpha = 1.0 } = options;
  ctx.save();
  ctx.globalAlpha = alpha;

  drawCastShadow(ctx, cx, cy, r, player);

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.clip();
  if (player === 1) drawBlackBody(ctx, cx, cy, r);
  else drawWhiteBody(ctx, cx, cy, r);
  ctx.restore();

  ctx.strokeStyle = player === 1 ? "rgba(255,255,255,0.18)" : "rgba(30,30,50,0.2)";
  ctx.lineWidth = Math.max(0.6, r * 0.035);
  ctx.beginPath();
  ctx.arc(cx, cy, r - ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

export function drawPendingGhostStone(
  ctx: CanvasRenderingContext2D,
  options: StoneRenderOptions
): void {
  const { cx, cy, radius: r, player } = options;
  ctx.save();
  // Black ghosts need more opacity to read clearly over the light glass.
  ctx.globalAlpha = player === 1 ? 0.72 : 0.55;

  const shadow = ctx.createRadialGradient(cx, cy + r * 0.15, r * 0.2, cx, cy + r * 0.18, r * 1.15);
  shadow.addColorStop(0, player === 1 ? "rgba(0,0,0,0.22)" : "rgba(0,0,0,0.16)");
  shadow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = shadow;
  ctx.beginPath();
  ctx.ellipse(cx, cy + r * 0.18, r * 1.1, r * 0.38, 0, 0, Math.PI * 2);
  ctx.fill();

  const body =
    player === 1
      ? solidBlackBody(ctx, cx, cy, r)
      : solidWhiteBody(ctx, cx, cy, r);
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalAlpha = player === 1 ? 0.95 : 0.8;
  ctx.strokeStyle = player === 1 ? "rgba(0,0,0,0.85)" : "rgba(20,20,35,0.35)";
  ctx.lineWidth = Math.max(0.9, r * (player === 1 ? 0.055 : 0.045));
  ctx.beginPath();
  ctx.arc(cx, cy, r - ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

export interface ThinkingGhostRenderOptions extends StoneRenderOptions {
  progress: number;
}

export function drawThinkingGhostStone(
  ctx: CanvasRenderingContext2D,
  options: ThinkingGhostRenderOptions
): void {
  const { cx, cy, radius: r, player, alpha = 0.3, progress } = options;
  ctx.save();
  ctx.globalAlpha = Math.max(0, Math.min(0.4, alpha));

  if (player !== 1) {
    const ringAlpha = Math.max(0, alpha * (1 - clamp01(progress / 0.58)));
    if (ringAlpha > 0.01) {
      ctx.save();
      ctx.globalAlpha = ringAlpha * 0.55;
      ctx.strokeStyle = "rgba(40,40,58,0.22)";
      ctx.lineWidth = Math.max(1, r * 0.08);
      ctx.beginPath();
      ctx.arc(cx, cy, r * (1.04 + easeOutCubic(clamp01(progress / 0.58)) * 0.24), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  const shadow = ctx.createRadialGradient(cx, cy + r * 0.16, r * 0.2, cx, cy + r * 0.18, r * 1.1);
  shadow.addColorStop(0, player === 1 ? "rgba(0,0,0,0.13)" : "rgba(0,0,0,0.08)");
  shadow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = shadow;
  ctx.beginPath();
  ctx.ellipse(cx, cy + r * 0.2, r * 1.0, r * 0.3, 0, 0, Math.PI * 2);
  ctx.fill();

  const body = ctx.createRadialGradient(cx - r * 0.25, cy - r * 0.3, r * 0.05, cx, cy, r);
  if (player === 1) {
    body.addColorStop(0, "#15151a");
    body.addColorStop(0.62, "#060608");
    body.addColorStop(1, "#000000");
  } else {
    body.addColorStop(0, "#ffffff");
    body.addColorStop(0.7, "#f0f0f4");
    body.addColorStop(1, "#c0c0cc");
  }
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalAlpha = Math.max(0, Math.min(0.55, alpha * 1.2));
  ctx.strokeStyle = player === 1 ? "rgba(0,0,0,0.65)" : "rgba(30,30,50,0.26)";
  ctx.lineWidth = Math.max(0.8, r * 0.05);
  ctx.beginPath();
  ctx.arc(cx, cy, r - ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
  // easeInCubic is kept in scope for bundler tree-shaking awareness.
  void easeInCubic;
}

function drawCastShadow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  player: Player
): void {
  const sh = ctx.createRadialGradient(
    cx + r * 0.04,
    cy + r * 0.18,
    r * 0.25,
    cx + r * 0.04,
    cy + r * 0.22,
    r * 1.3
  );
  sh.addColorStop(0, player === 1 ? "rgba(0,0,0,0.38)" : "rgba(0,0,0,0.22)");
  sh.addColorStop(0.55, player === 1 ? "rgba(0,0,0,0.14)" : "rgba(0,0,0,0.08)");
  sh.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = sh;
  ctx.beginPath();
  ctx.ellipse(cx + r * 0.04, cy + r * 0.22, r * 1.2, r * 0.5, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawBlackBody(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number): void {
  const body = ctx.createRadialGradient(
    cx - r * 0.3,
    cy - r * 0.35,
    r * 0.05,
    cx + r * 0.1,
    cy + r * 0.18,
    r * 1.1
  );
  body.addColorStop(0, "#6e6e78");
  body.addColorStop(0.28, "#2a2a32");
  body.addColorStop(0.7, "#0e0e14");
  body.addColorStop(1, "#020206");
  ctx.fillStyle = body;
  ctx.fillRect(cx - r, cy - r, r * 2, r * 2);

  const glow = ctx.createRadialGradient(
    cx - r * 0.3,
    cy - r * 0.4,
    0,
    cx - r * 0.3,
    cy - r * 0.4,
    r * 0.95
  );
  glow.addColorStop(0, "rgba(255,255,255,0.38)");
  glow.addColorStop(0.4, "rgba(255,255,255,0.1)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(cx - r, cy - r, r * 2, r * 2);

  const spec = ctx.createRadialGradient(
    cx - r * 0.38,
    cy - r * 0.46,
    0,
    cx - r * 0.38,
    cy - r * 0.46,
    r * 0.34
  );
  spec.addColorStop(0, "rgba(255,255,255,0.78)");
  spec.addColorStop(0.45, "rgba(255,255,255,0.25)");
  spec.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = spec;
  ctx.beginPath();
  ctx.ellipse(cx - r * 0.38, cy - r * 0.46, r * 0.32, r * 0.18, -Math.PI / 5, 0, Math.PI * 2);
  ctx.fill();

  const cool = ctx.createRadialGradient(
    cx + r * 0.35,
    cy + r * 0.3,
    0,
    cx + r * 0.35,
    cy + r * 0.3,
    r * 0.7
  );
  cool.addColorStop(0, "rgba(120, 140, 200, 0.18)");
  cool.addColorStop(1, "rgba(120, 140, 200, 0)");
  ctx.fillStyle = cool;
  ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
}

function drawWhiteBody(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number): void {
  const body = ctx.createRadialGradient(
    cx - r * 0.22,
    cy - r * 0.28,
    r * 0.05,
    cx + r * 0.12,
    cy + r * 0.22,
    r * 1.05
  );
  body.addColorStop(0, "#ffffff");
  body.addColorStop(0.45, "#f4f4f6");
  body.addColorStop(0.85, "#c2c2ca");
  body.addColorStop(1, "#a8a8b2");
  ctx.fillStyle = body;
  ctx.fillRect(cx - r, cy - r, r * 2, r * 2);

  const glow = ctx.createRadialGradient(
    cx - r * 0.3,
    cy - r * 0.35,
    0,
    cx - r * 0.3,
    cy - r * 0.35,
    r * 0.85
  );
  glow.addColorStop(0, "rgba(255,255,255,0.95)");
  glow.addColorStop(0.5, "rgba(255,255,255,0.4)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(cx - r, cy - r, r * 2, r * 2);

  const spec = ctx.createRadialGradient(
    cx - r * 0.35,
    cy - r * 0.4,
    0,
    cx - r * 0.35,
    cy - r * 0.4,
    r * 0.3
  );
  spec.addColorStop(0, "rgba(255,255,255,1)");
  spec.addColorStop(0.5, "rgba(255,255,255,0.5)");
  spec.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = spec;
  ctx.beginPath();
  ctx.ellipse(cx - r * 0.35, cy - r * 0.4, r * 0.28, r * 0.17, -Math.PI / 5, 0, Math.PI * 2);
  ctx.fill();

  const cool = ctx.createRadialGradient(
    cx + r * 0.4,
    cy + r * 0.35,
    0,
    cx + r * 0.4,
    cy + r * 0.35,
    r * 0.75
  );
  cool.addColorStop(0, "rgba(140, 160, 200, 0.18)");
  cool.addColorStop(1, "rgba(140, 160, 200, 0)");
  ctx.fillStyle = cool;
  ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
}

function solidBlackBody(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  const body = ctx.createRadialGradient(cx - r * 0.25, cy - r * 0.3, r * 0.05, cx, cy, r);
  body.addColorStop(0, "#1c1c22");
  body.addColorStop(0.55, "#08080c");
  body.addColorStop(1, "#000000");
  return body;
}

function solidWhiteBody(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  const body = ctx.createRadialGradient(cx - r * 0.25, cy - r * 0.3, r * 0.05, cx, cy, r);
  body.addColorStop(0, "#ffffff");
  body.addColorStop(0.7, "#ededf1");
  body.addColorStop(1, "#b4b4bf");
  return body;
}
