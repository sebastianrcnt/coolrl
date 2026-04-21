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
from dataclasses import dataclass
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
# Apple-inspired palettes (dark + light, keynote-style)
# ---------------------------------------------------------------------------
@dataclass
class Theme:
    name: str
    bg: str
    panel_bg: str
    grid: str
    text_primary: str
    text_secondary: str
    text_tertiary: str
    accents: dict  # name -> (gradient_start, gradient_end)
    line_alpha: float = 1.0


DARK = Theme(
    name="dark",
    bg="#000000",
    panel_bg="#0e0e10",
    grid="#1c1c1e",
    text_primary="#f5f5f7",
    text_secondary="#a1a1a6",
    text_tertiary="#6e6e73",
    # Rainbow by row: red → orange → yellow → green → blue.
    accents={
        "policy": ("#ff3b30", "#ff6961"),
        "value": ("#ff3b30", "#ff6961"),
        "total": ("#ff9500", "#ffb347"),
        "lr": ("#ff9500", "#ffb347"),
        "winrate": ("#ffcc00", "#ffe066"),
        "white": ("#34c759", "#5ce07c"),
        "moves": ("#34c759", "#5ce07c"),
        "buffer": ("#0a84ff", "#64d2ff"),
        "elapsed": ("#0a84ff", "#64d2ff"),
    },
)

LIGHT = Theme(
    name="light",
    bg="#ffffff",
    panel_bg="#f5f5f7",
    grid="#d2d2d7",
    text_primary="#1d1d1f",
    text_secondary="#424245",
    text_tertiary="#6e6e73",
    # Rainbow by row, muted for white background.
    accents={
        "policy": ("#d70015", "#ff3b30"),
        "value": ("#d70015", "#ff3b30"),
        "total": ("#c93400", "#ff9500"),
        "lr": ("#c93400", "#ff9500"),
        "winrate": ("#946200", "#d2a200"),
        "white": ("#248a3d", "#34c759"),
        "moves": ("#248a3d", "#34c759"),
        "buffer": ("#0040dd", "#007aff"),
        "elapsed": ("#0040dd", "#007aff"),
    },
    line_alpha=0.85,
)

THEMES: dict[str, Theme] = {"dark": DARK, "light": LIGHT}

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


def style_axis(
    ax, theme: Theme, *, percent: bool = False, hours: bool = False
) -> None:
    ax.set_facecolor(theme.bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=theme.text_secondary, length=0, labelsize=10.5, pad=5)
    ax.grid(True, axis="y", color=theme.grid, linewidth=0.6, zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    if hours:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}h"))


def gradient_line(
    ax,
    x,
    y,
    c0: str,
    c1: str,
    lw: float = 2.4,
    glow: bool = False,
    alpha: float = 1.0,
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
        alpha=alpha,
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


def _easing_smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average easing for noisy curves. Preserves NaN positions."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y
    finite_mask = np.isfinite(y)
    if not finite_mask.any():
        return y
    idx = np.arange(n)
    y_clean = y.copy()
    if (~finite_mask).any():
        y_clean[~finite_mask] = np.interp(
            idx[~finite_mask], idx[finite_mask], y[finite_mask]
        )
    pad = w // 2
    padded = np.concatenate([y_clean[:pad][::-1], y_clean, y_clean[-pad:][::-1]])
    kernel = np.ones(w) / w
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[~finite_mask] = np.nan
    return smoothed


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


def panel_title(ax, theme: Theme, title: str, subtitle: str = "") -> None:
    ax.text(
        0.0,
        1.34,
        title,
        transform=ax.transAxes,
        fontsize=17,
        color=theme.text_primary,
        fontweight=900,
    )
    if subtitle:
        ax.text(
            0.0,
            1.15,
            subtitle,
            transform=ax.transAxes,
            fontsize=13,
            color=theme.text_secondary,
        )


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------
def build_figure(
    rows: list[dict],
    metrics_path: Path,
    *,
    smooth: bool = True,
    theme: Theme = DARK,
) -> plt.Figure:
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

    if smooth:
        # Apply easing only to noisy series; leave lr/buffer/elapsed untouched.
        policy = _easing_smooth(policy)
        value = _easing_smooth(value)
        total = _easing_smooth(total)
        arena = _easing_smooth(arena, window=7)
        white = _easing_smooth(white, window=7)
        moves = _easing_smooth(moves)

    # Layout
    fig = plt.figure(figsize=(13, 20.5), facecolor=theme.bg)
    gs = fig.add_gridspec(
        nrows=6,
        ncols=4,
        height_ratios=[1.0, 1.0, 1.0, 1.05, 1.0, 0.85],
        hspace=1.0,
        wspace=0.32,
        left=0.055,
        right=0.97,
        top=0.97,
        bottom=0.04,
    )

    # ============================== Header row ==============================
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(theme.bg)
    ax_h.axis("off")
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1)

    run_name = metrics_path.parent.name
    ax_h.text(
        0.0,
        0.92,
        run_name.upper(),
        fontsize=12,
        color=theme.text_tertiary,
        fontweight="semibold",
        transform=ax_h.transAxes,
    )
    ax_h.text(
        0.0,
        0.62,
        f"Iteration {cur_iter:,}",
        fontsize=36,
        color=theme.text_primary,
        fontweight=900,
        transform=ax_h.transAxes,
        va="center",
    )
    ax_h.text(
        0.0,
        0.20,
        f"{cur_elapsed:.1f}h  ·  {avg_iter_sec:.0f}s/iter  ·  "
        f"{accepted_count} accepted",
        fontsize=14,
        color=theme.text_secondary,
        transform=ax_h.transAxes,
    )

    # KPI sidebar on the right — Apple keynote style: small qualifier, accent
    # value, one-line descriptor. 2x2 grid with a hairline divider between rows.
    kpis = [
        (
            "BEST ITER",
            f"{best_iter}",
            theme.accents["winrate"][0],
            f"of {cur_iter}",
        ),
        (
            "BEST WIN RATE",
            f"{best_wr * 100:.1f}%",
            theme.accents["white"][0],
            "peak arena",
        ),
        (
            "POLICY LOSS",
            f"{cur_policy:.2f}",
            theme.accents["policy"][0],
            f"uniform {uniform_entropy:.2f}",
        ),
        (
            "VALUE LOSS",
            f"{cur_value:.2f}",
            theme.accents["value"][0],
            "lower is better",
        ),
    ]
    # Right-edge anchors for each column (flush with chart right edge).
    col_right_xs = (0.78, 1.0)
    row_positions = (
        # (label_y, value_y, descriptor_y)
        (0.93, 0.70, 0.52),
        (0.36, 0.13, -0.05),
    )
    for i, (label, val, _accent, desc) in enumerate(kpis):
        col = i % 2
        row = i // 2
        text_x = col_right_xs[col]
        label_y, val_y, desc_y = row_positions[row]
        ax_h.text(
            text_x,
            label_y,
            label,
            fontsize=11,
            color=theme.text_tertiary,
            fontweight="semibold",
            transform=ax_h.transAxes,
            va="center",
            ha="right",
        )
        ax_h.text(
            text_x,
            val_y,
            val,
            fontsize=26,
            color=theme.text_primary,
            fontweight=900,
            transform=ax_h.transAxes,
            va="center",
            ha="right",
        )
        ax_h.text(
            text_x,
            desc_y,
            desc,
            fontsize=12,
            color=theme.text_secondary,
            transform=ax_h.transAxes,
            va="center",
            ha="right",
        )

    # ============================== Charts ==============================
    def add_chart(slot):
        ax = fig.add_subplot(slot)
        ax.set_facecolor(theme.bg)
        return ax

    n_max = cur_iter + max(5, cur_iter // 50)

    # ---- Policy Loss (wide left) ----
    ax = add_chart(gs[1, :2])
    panel_title(
        ax,
        theme,
        "Policy Loss",
        f"Now {cur_policy:.2f}  ·  Uniform ln({board_size * board_size}) = {uniform_entropy:.2f}",
    )
    style_axis(ax, theme)
    ax.set_xlim(0, n_max)
    if len(policy):
        ymin = float(np.nanmin(policy)) - 0.15
        ymax = max(uniform_entropy + 0.12, float(np.nanmax(policy)) + 0.05)
        ax.set_ylim(ymin, ymax)
    ax.axhline(
        uniform_entropy,
        color=theme.text_tertiary,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        alpha=0.7,
        zorder=2,
    )
    cs, ce = theme.accents["policy"]
    gradient_fill(ax, t_iters, policy, cs, ce, alpha=0.16)
    gradient_line(ax, t_iters, policy, cs, ce, lw=2.4, alpha=theme.line_alpha)

    # ---- Value Loss (wide right) ----
    ax = add_chart(gs[1, 2:])
    panel_title(ax, theme, "Value Loss", f"Now {cur_value:.2f}  ·  Lower is better")
    style_axis(ax, theme)
    ax.set_xlim(0, n_max)
    if len(value):
        ax.set_ylim(0, max(float(np.nanmax(value)) * 1.12, 1.0))
    cs, ce = theme.accents["value"]
    gradient_fill(ax, t_iters, value, cs, ce, alpha=0.16)
    gradient_line(ax, t_iters, value, cs, ce, lw=2.4, alpha=theme.line_alpha)

    # ---- Total Loss + Learning Rate ----
    ax = add_chart(gs[2, :2])
    panel_title(ax, theme, "Total Loss", f"Now {cur_total:.2f}  ·  policy + 1.5 × value")
    style_axis(ax, theme)
    ax.set_xlim(0, n_max)
    if len(total):
        ax.set_ylim(float(np.nanmin(total)) - 0.3, float(np.nanmax(total)) * 1.05)
    cs, ce = theme.accents["total"]
    gradient_fill(ax, t_iters, total, cs, ce, alpha=0.16)
    gradient_line(ax, t_iters, total, cs, ce, lw=2.4, alpha=theme.line_alpha)

    ax = add_chart(gs[2, 2:])
    cur_lr = float(lr[-1]) if len(lr) and np.isfinite(lr[-1]) else 0.0
    panel_title(ax, theme, "Learning Rate", f"Constant {cur_lr:.0e}")
    style_axis(ax, theme)
    ax.set_xlim(0, n_max)
    if len(lr):
        ax.set_ylim(0, float(np.nanmax(lr)) * 1.35)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_lr))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    cs, ce = theme.accents["lr"]
    gradient_line(ax, t_iters, lr, cs, ce, lw=2.4, glow=False, alpha=theme.line_alpha)

    # ---- Arena Win Rate (full width) ----
    ax = add_chart(gs[3, :])
    panel_title(
        ax,
        theme,
        "Arena Win Rate",
        f"{accepted_count} candidates accepted  ·  Threshold {accept_thresh:.0f}%",
    )
    style_axis(ax, theme, percent=True)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, 100)
    ax.axhline(
        accept_thresh,
        color=theme.text_tertiary,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        alpha=0.7,
        zorder=2,
    )
    cs, ce = theme.accents["winrate"]
    gradient_fill(ax, t_iters, arena, cs, ce, alpha=0.13)
    gradient_line(ax, t_iters, arena, cs, ce, lw=1.9, alpha=theme.line_alpha)
    if accepted.any():
        ax.scatter(
            t_iters[accepted],
            arena[accepted],
            s=22,
            color=theme.text_primary,
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
        theme,
        "Candidate White Win Rate",
        f"Now {cur_white:.0f}%  ·  Healthy band 30-70%  ·  Floor {white_thresh:.0f}%",
    )
    style_axis(ax, theme, percent=True)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, 100)
    ax.axhspan(30, 70, color=theme.text_tertiary, alpha=0.10, zorder=0)
    ax.axhline(
        white_thresh,
        color=theme.text_tertiary,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        alpha=0.7,
        zorder=2,
    )
    cs, ce = theme.accents["white"]
    gradient_line(ax, t_iters, white, cs, ce, lw=2.0, alpha=theme.line_alpha)

    ax = add_chart(gs[4, 2:])
    panel_title(
        ax,
        theme,
        "Selfplay Avg Moves",
        f"Now {cur_moves:.1f} moves per game"
        if np.isfinite(cur_moves)
        else "Selfplay length",
    )
    style_axis(ax, theme)
    ax.set_xlim(0, n_max)
    if len(moves):
        ax.set_ylim(0, max(float(np.nanmax(moves)) * 1.1, 30))
    cs, ce = theme.accents["moves"]
    gradient_fill(ax, iters, moves, cs, ce, alpha=0.12)
    gradient_line(ax, iters, moves, cs, ce, lw=2.0, alpha=theme.line_alpha)

    # ---- Replay Buffer + Elapsed ----
    ax = add_chart(gs[5, :2])
    panel_title(ax, theme, "Replay Buffer", f"{cur_buffer:,} games  ·  Capacity 200,000")
    style_axis(ax, theme)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, max(cur_buffer * 1.12, 1000))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}"
        )
    )
    cs, ce = theme.accents["buffer"]
    gradient_fill(ax, iters, buffer, cs, ce, alpha=0.16)
    gradient_line(ax, iters, buffer, cs, ce, lw=2.0, alpha=theme.line_alpha)

    ax = add_chart(gs[5, 2:])
    panel_title(
        ax,
        theme,
        "Elapsed",
        f"{cur_elapsed:.2f} hours total  ·  {avg_iter_sec:.0f}s per iteration",
    )
    style_axis(ax, theme, hours=True)
    ax.set_xlim(0, n_max)
    ax.set_ylim(0, max(cur_elapsed * 1.1, 1.0))
    cs, ce = theme.accents["elapsed"]
    gradient_fill(ax, iters, elapsed, cs, ce, alpha=0.10)
    gradient_line(ax, iters, elapsed, cs, ce, lw=2.0, alpha=theme.line_alpha)

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
    parser.add_argument(
        "--no-smooth",
        dest="smooth",
        action="store_false",
        help="Disable easing/smoothing on curves (on by default)",
    )
    parser.add_argument(
        "--theme",
        choices=sorted(THEMES),
        default="dark",
        help="Color theme (default: dark)",
    )
    parser.set_defaults(smooth=True)
    args = parser.parse_args()

    theme = THEMES[args.theme]

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

    fig = build_figure(rows, metrics_path, smooth=args.smooth, theme=theme)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=theme.bg, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {out}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
