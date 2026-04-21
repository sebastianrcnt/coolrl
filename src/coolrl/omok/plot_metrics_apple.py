"""Apple-style training metrics plotter for omok RL runs.

Reads metrics.jsonl produced by the trainer and renders a polished
keynote-style dashboard with gradient lines, KPI cards, and dark theme.

Usage:
    python plot_metrics_apple.py /path/to/checkpoints/run_name
    python plot_metrics_apple.py /path/to/metrics.jsonl -o out.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon


# ---------------------------------------------------------------------------
# Apple-inspired palette (dark mode, keynote-style)
# ---------------------------------------------------------------------------
BG = "#000000"
PANEL_BG = "#0e0e10"
GRID = "#1c1c1e"
TEXT_PRIMARY = "#f5f5f7"
TEXT_SECONDARY = "#a1a1a6"
TEXT_TERTIARY = "#6e6e73"

ACCENTS = {
    "policy": ("#ff453a", "#ff9f0a"),  # red to orange
    "value": ("#30d158", "#64d2ff"),  # green to cyan
    "total": ("#5e5ce6", "#bf5af2"),  # indigo to purple
    "lr": ("#ff375f", "#ff9f0a"),
    "winrate": ("#bf5af2", "#ff375f"),  # purple to pink
    "white": ("#ff9f0a", "#ffd60a"),  # orange to yellow
    "moves": ("#ff6482", "#ff2d55"),  # pink shades
    "buffer": ("#64d2ff", "#0a84ff"),  # cyan to blue
    "elapsed": ("#8e8e93", "#c7c7cc"),
}

REPO_ROOT = Path(__file__).resolve().parents[3]
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
SF_PRO_DIR = REPO_ROOT / "static" / "fonts"


def configure_fonts() -> None:
    font_names: list[str] = []
    # Load all static SF Pro Display faces so matplotlib can pick per weight.
    sf_pro_faces = sorted(SF_PRO_DIR.glob("SF-Pro-Display-*.otf"))
    for face in sf_pro_faces:
        font_manager.fontManager.addfont(str(face))
        name = font_manager.FontProperties(fname=str(face)).get_name()
        if name not in font_names:
            font_names.append(name)

    # Fallbacks: bundled B612 (regular + bold), then DejaVu.
    for fallback in (FONT_DIR / "B612-Regular.ttf", FONT_DIR / "B612-Bold.ttf"):
        if not fallback.exists():
            continue
        font_manager.fontManager.addfont(str(fallback))
        name = font_manager.FontProperties(fname=str(fallback)).get_name()
        if name not in font_names:
            font_names.append(name)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [*font_names, "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_metrics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_rows(rows: list[dict]) -> list[dict]:
    by_iter: dict[int, dict] = {}
    extras: list[dict] = []
    for row in rows:
        it = row.get("iteration")
        if isinstance(it, int):
            by_iter[it] = row
        else:
            extras.append(row)
    return extras + [by_iter[i] for i in sorted(by_iter)]


def field(rows: Sequence[dict], key: str, default=np.nan) -> np.ndarray:
    return np.array([r.get(key, default) for r in rows], dtype=float)


def trained_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "trained"]


def infer_board_size(rows: Sequence[dict], metrics_path: Path) -> int:
    for row in reversed(rows):
        raw = row.get("board_size")
        if raw is not None:
            return int(raw)
    for sidecar in ("latest.json", "best.json", "iter_0000.json"):
        p = metrics_path.parent / sidecar
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        bs = payload.get("config", {}).get("rules", {}).get("board_size")
        if bs is not None:
            return int(bs)
    return 9


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def make_cmap(c0: str, c1: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("g", [c0, c1], N=256)


def style_axis(ax, *, percent: bool = False, hours: bool = False) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8.5, pad=5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    if hours:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}h"))


def gradient_line(
    ax, x, y, c0: str, c1: str, lw: float = 2.4, glow: bool = False
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = make_cmap(c0, c1)
    norm = plt.Normalize(x.min(), x.max())
    lc = LineCollection(
        segs,
        cmap=cmap,
        norm=norm,
        linewidth=lw,
        alpha=1.0,
        capstyle="round",
        joinstyle="round",
        zorder=4,
    )
    lc.set_array(x)
    ax.add_collection(lc)


def gradient_fill(ax, x, y, c0: str, c1: str, alpha: float = 0.18) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return
    ymin, ymax = ax.get_ylim()
    cmap = make_cmap(c0, c1)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    grad = np.tile(grad, (1, 2))
    extent = (x.min(), x.max(), ymin, max(np.nanmax(y), ymin + 1e-6))
    img = ax.imshow(
        grad,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        zorder=1,
    )
    poly_xy = np.column_stack(
        [
            np.concatenate([x, x[::-1]]),
            np.concatenate([y, np.full_like(x, ymin)]),
        ]
    )
    img.set_clip_path(Polygon(poly_xy, closed=True, transform=ax.transData))


def rounded_card(
    ax,
    xy,
    w,
    h,
    *,
    accent: str | None = None,
    dot_cx: float | None = None,
    dot_cy: float | None = None,
) -> None:
    card = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.002,rounding_size=0.020",
        transform=ax.transAxes,
        facecolor="#141418",
        edgecolor="#2d2d30",
        linewidth=0.8,
    )
    ax.add_patch(card)
    # Faint inset highlight gives the liquid-glass edge.
    inset = FancyBboxPatch(
        (xy[0] + 0.0015, xy[1] + 0.003),
        w - 0.003,
        h - 0.006,
        boxstyle="round,pad=0,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="#3a3a3f",
        linewidth=0.5,
        alpha=0.55,
    )
    ax.add_patch(inset)
    if accent:
        cx = dot_cx if dot_cx is not None else xy[0] + 0.022
        cy = dot_cy if dot_cy is not None else xy[1] + h * 0.78
        ax.scatter(
            [cx],
            [cy],
            s=30,
            c=accent,
            marker="o",
            edgecolors="none",
            transform=ax.transAxes,
            zorder=5,
        )


def _format_lr(v: float, _pos: int) -> str:
    if v <= 0:
        return "0"
    micro = v * 1e4
    if micro < 10:
        return (
            f"{int(round(micro))}e-4"
            if abs(micro - round(micro)) < 1e-6
            else f"{micro:.1f}e-4"
        )
    milli = v * 1e3
    return (
        f"{int(round(milli))}e-3"
        if abs(milli - round(milli)) < 1e-6
        else f"{milli:.1f}e-3"
    )


def panel_title(ax, title: str, subtitle: str = "") -> None:
    ax.text(
        0.0,
        1.18,
        title,
        transform=ax.transAxes,
        fontsize=12.5,
        color=TEXT_PRIMARY,
        fontweight=900,
    )
    if subtitle:
        ax.text(
            0.0,
            1.05,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------
def build_figure(rows: list[dict], metrics_path: Path) -> plt.Figure:
    configure_fonts()

    rows = normalize_rows(rows)
    trained = trained_rows(rows)
    board_size = infer_board_size(rows, metrics_path)
    uniform_entropy = float(np.log(board_size * board_size))

    iters = field(rows, "iteration")
    elapsed = field(rows, "elapsed_hours", 0.0)
    moves = field(rows, "selfplay_avg_moves")
    buffer = field(rows, "replay_games", 0.0)

    t_iters = field(trained, "iteration")
    policy = field(trained, "policy_loss")
    value = field(trained, "value_loss")
    total = field(trained, "train_loss")
    lr = field(trained, "learning_rate")
    arena = field(trained, "arena_win_rate") * 100
    white = field(trained, "arena_candidate_white_win_rate") * 100
    accepted = np.array([bool(r.get("accepted")) for r in trained])

    accept_thresh = (
        (trained[0].get("arena_accept_win_rate") or 0.55) * 100 if trained else 55
    )
    white_thresh = (
        (trained[0].get("arena_white_win_rate_threshold") or 0.1667) * 100
        if trained
        else 17
    )

    last = rows[-1]
    cur_iter = int(iters[-1]) if len(iters) else 0
    cur_elapsed = float(elapsed[-1]) if len(elapsed) else 0.0
    cur_policy = float(policy[-1]) if len(policy) else float("nan")
    cur_value = float(value[-1]) if len(value) else float("nan")
    cur_total = float(total[-1]) if len(total) else float("nan")
    cur_buffer = int(buffer[-1]) if len(buffer) else 0
    cur_moves = (
        float(moves[-1]) if len(moves) and np.isfinite(moves[-1]) else float("nan")
    )

    best_iter = int(last.get("best_iteration", 0))
    best_wr = float(last.get("best_arena_win_rate", 0.0))
    accepted_count = int(accepted.sum())

    avg_iter_sec = (cur_elapsed * 3600 / cur_iter) if cur_iter else 0

    # Layout
    fig = plt.figure(figsize=(13, 17), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=6,
        ncols=4,
        height_ratios=[0.55, 1.0, 1.0, 1.05, 1.0, 0.85],
        hspace=0.80,
        wspace=0.32,
        left=0.055,
        right=0.97,
        top=0.96,
        bottom=0.04,
    )

    # ============================== Header row ==============================
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1)

    run_name = metrics_path.parent.name
    ax_h.text(
        0.0,
        0.92,
        run_name.upper(),
        fontsize=10,
        color=TEXT_TERTIARY,
        fontweight="semibold",
        transform=ax_h.transAxes,
    )
    ax_h.text(
        0.0,
        0.62,
        f"Iteration {cur_iter:,}",
        fontsize=30,
        color=TEXT_PRIMARY,
        fontweight=900,
        transform=ax_h.transAxes,
        va="center",
    )
    ax_h.text(
        0.0,
        0.20,
        f"{cur_elapsed:.1f}h  ·  {avg_iter_sec:.0f}s/iter  ·  "
        f"{accepted_count} accepted",
        fontsize=11,
        color=TEXT_SECONDARY,
        transform=ax_h.transAxes,
    )

    # KPI cards — liquid-glass style: soft translucent panel + tiny accent dot.
    kpis = [
        ("BEST ITER", f"{best_iter}", ACCENTS["winrate"][0]),
        ("BEST WIN RATE", f"{best_wr * 100:.1f}%", ACCENTS["white"][0]),
        ("POLICY LOSS", f"{cur_policy:.2f}", ACCENTS["policy"][0]),
        ("VALUE LOSS", f"{cur_value:.2f}", ACCENTS["value"][0]),
    ]
    card_w = 0.165
    card_gap = 0.014
    card_h = 0.78
    card_y = 0.12
    total_w = card_w * len(kpis) + card_gap * (len(kpis) - 1)
    start_x = 1.0 - total_w
    for i, (label, val, accent) in enumerate(kpis):
        x0 = start_x + i * (card_w + card_gap)
        label_y = card_y + card_h * 0.80
        dot_cx = x0 + 0.020
        rounded_card(
            ax_h,
            (x0, card_y),
            card_w,
            card_h,
            accent=accent,
            dot_cx=dot_cx,
            dot_cy=label_y,
        )
        ax_h.text(
            x0 + 0.036,
            label_y,
            label,
            fontsize=8,
            color=TEXT_SECONDARY,
            fontweight="semibold",
            transform=ax_h.transAxes,
            va="center",
        )
        ax_h.text(
            x0 + 0.036,
            card_y + card_h * 0.32,
            val,
            fontsize=22,
            color=TEXT_PRIMARY,
            fontweight=900,
            transform=ax_h.transAxes,
            va="center",
        )

    # ============================== Charts ==============================
    def add_chart(slot):
        ax = fig.add_subplot(slot)
        ax.set_facecolor(BG)
        return ax

    n_max = cur_iter + max(5, cur_iter // 50)

    # ---- Policy Loss (wide left) ----
    ax = add_chart(gs[1, :2])
    panel_title(
        ax,
        "Policy Loss",
        f"Now {cur_policy:.2f}  ·  Uniform ln({board_size * board_size}) = {uniform_entropy:.2f}",
    )
    style_axis(ax)
    ax.set_xlim(0, n_max)
    if len(policy):
        ymin = float(np.nanmin(policy)) - 0.15
        ymax = max(uniform_entropy + 0.12, float(np.nanmax(policy)) + 0.05)
        ax.set_ylim(ymin, ymax)
    ax.axhline(
        uniform_entropy,
        color=TEXT_TERTIARY,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        alpha=0.7,
        zorder=2,
    )
    cs, ce = ACCENTS["policy"]
    gradient_fill(ax, t_iters, policy, cs, ce, alpha=0.16)
    gradient_line(ax, t_iters, policy, cs, ce, lw=2.4)

    # ---- Value Loss (wide right) ----
    ax = add_chart(gs[1, 2:])
    panel_title(ax, "Value Loss", f"Now {cur_value:.2f}  ·  Lower is better")
    style_axis(ax)
    ax.set_xlim(0, n_max)
    if len(value):
        ax.set_ylim(0, max(float(np.nanmax(value)) * 1.12, 1.0))
    cs, ce = ACCENTS["value"]
    gradient_fill(ax, t_iters, value, cs, ce, alpha=0.16)
    gradient_line(ax, t_iters, value, cs, ce, lw=2.4)

    # ---- Total Loss + Learning Rate ----
    ax = add_chart(gs[2, :2])
    panel_title(ax, "Total Loss", f"Now {cur_total:.2f}  ·  policy + 1.5 × value")
    style_axis(ax)
    ax.set_xlim(0, n_max)
    if len(total):
        ax.set_ylim(float(np.nanmin(total)) - 0.3, float(np.nanmax(total)) * 1.05)
    cs, ce = ACCENTS["total"]
    gradient_fill(ax, t_iters, total, cs, ce, alpha=0.16)
    gradient_line(ax, t_iters, total, cs, ce, lw=2.4)

    ax = add_chart(gs[2, 2:])
    cur_lr = float(lr[-1]) if len(lr) and np.isfinite(lr[-1]) else 0.0
    panel_title(ax, "Learning Rate", f"Constant {cur_lr:.0e}")
    style_axis(ax)
    ax.set_xlim(0, n_max)
    if len(lr):
        ax.set_ylim(0, float(np.nanmax(lr)) * 1.35)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_lr))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    cs, ce = ACCENTS["lr"]
    gradient_line(ax, t_iters, lr, cs, ce, lw=2.4, glow=False)

    # ---- Arena Win Rate (full width) ----
    ax = add_chart(gs[3, :])
    panel_title(
        ax,
        "Arena Win Rate",
        f"{accepted_count} candidates accepted  ·  Threshold {accept_thresh:.0f}%",
    )
    style_axis(ax, percent=True)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, 100)
    ax.axhline(
        accept_thresh,
        color=TEXT_TERTIARY,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        alpha=0.7,
        zorder=2,
    )
    cs, ce = ACCENTS["winrate"]
    gradient_fill(ax, t_iters, arena, cs, ce, alpha=0.13)
    gradient_line(ax, t_iters, arena, cs, ce, lw=1.9)
    if accepted.any():
        ax.scatter(
            t_iters[accepted],
            arena[accepted],
            s=22,
            color=ACCENTS["white"][1],
            zorder=6,
            edgecolors="none",
        )

    # ---- White Win Rate + Selfplay Moves ----
    ax = add_chart(gs[4, :2])
    cur_white = (
        float(white[-1]) if len(white) and np.isfinite(white[-1]) else float("nan")
    )
    panel_title(
        ax,
        "Candidate White Win Rate",
        f"Now {cur_white:.0f}%  ·  Healthy band 30-70%  ·  Floor {white_thresh:.0f}%",
    )
    style_axis(ax, percent=True)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, 100)
    ax.axhspan(30, 70, color=ACCENTS["value"][0], alpha=0.05, zorder=0)
    ax.axhline(
        white_thresh,
        color=TEXT_TERTIARY,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        alpha=0.7,
        zorder=2,
    )
    cs, ce = ACCENTS["white"]
    gradient_line(ax, t_iters, white, cs, ce, lw=2.0)

    ax = add_chart(gs[4, 2:])
    panel_title(
        ax,
        "Selfplay Avg Moves",
        f"Now {cur_moves:.1f} moves per game"
        if np.isfinite(cur_moves)
        else "Selfplay length",
    )
    style_axis(ax)
    ax.set_xlim(0, n_max)
    if len(moves):
        ax.set_ylim(0, max(float(np.nanmax(moves)) * 1.1, 30))
    cs, ce = ACCENTS["moves"]
    gradient_fill(ax, iters, moves, cs, ce, alpha=0.12)
    gradient_line(ax, iters, moves, cs, ce, lw=2.0)

    # ---- Replay Buffer + Elapsed ----
    ax = add_chart(gs[5, :2])
    panel_title(ax, "Replay Buffer", f"{cur_buffer:,} games  ·  Capacity 200,000")
    style_axis(ax)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, max(cur_buffer * 1.12, 1000))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}"
        )
    )
    cs, ce = ACCENTS["buffer"]
    gradient_fill(ax, iters, buffer, cs, ce, alpha=0.16)
    gradient_line(ax, iters, buffer, cs, ce, lw=2.0)

    ax = add_chart(gs[5, 2:])
    panel_title(
        ax,
        "Elapsed",
        f"{cur_elapsed:.2f} hours total  ·  {avg_iter_sec:.0f}s per iteration",
    )
    style_axis(ax, hours=True)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, max(cur_elapsed * 1.1, 1.0))
    cs, ce = ACCENTS["elapsed"]
    gradient_fill(ax, iters, elapsed, cs, ce, alpha=0.10)
    gradient_line(ax, iters, elapsed, cs, ce, lw=2.0)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apple-style metrics plotter for omok."
    )
    parser.add_argument(
        "metrics", type=str, help="Path to metrics.jsonl or a checkpoint directory"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: <dir>/metrics_apple.png)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Open interactive matplotlib window"
    )
    args = parser.parse_args()

    p = Path(args.metrics)
    metrics_path = p / "metrics.jsonl" if p.is_dir() else p
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    out = (
        Path(args.output) if args.output else metrics_path.parent / "metrics_apple.png"
    )
    rows = load_metrics(metrics_path)
    if not rows:
        print(f"No data in {metrics_path}")
        return

    fig = build_figure(rows, metrics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {out}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
