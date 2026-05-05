"""Generic matplotlib primitives shared across plotters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker

from .theme import Theme


def _make_cmap(c0: str, c1: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("coolrl_gradient", [c0, c1], N=256)


def style_axis(
    ax,
    theme: Theme,
    *,
    percent: bool = False,
    hours: bool = False,
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
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    c0: str,
    c1: str,
    *,
    lw: float = 2.4,
    glow: bool = False,
    alpha: float = 1.0,
) -> None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return

    pts = np.array([x_arr, y_arr]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = _make_cmap(c0, c1)
    norm = Normalize(vmin=float(x_arr.min()), vmax=float(x_arr.max()))

    if glow:
        glow_lc = LineCollection(
            segs,
            cmap=cmap,
            norm=norm,
            linewidth=lw * 3.0,
            alpha=max(alpha * 0.16, 0.05),
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
        glow_lc.set_array(x_arr)
        ax.add_collection(glow_lc)

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
    lc.set_array(x_arr)
    ax.add_collection(lc)


def gradient_fill(
    ax,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    c0: str,
    c1: str,
    *,
    alpha: float = 0.18,
) -> None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return

    ymin = float(ax.get_ylim()[0])
    ymax = max(float(np.nanmax(y_arr)), ymin + 1e-6)
    cmap = _make_cmap(c0, c1)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    grad = np.tile(grad, (1, 2))
    img = ax.imshow(
        grad,
        extent=(float(x_arr.min()), float(x_arr.max()), ymin, ymax),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        zorder=1,
    )
    poly_xy = np.column_stack(
        [
            np.concatenate([x_arr, x_arr[::-1]]),
            np.concatenate([y_arr, np.full_like(x_arr, ymin)]),
        ]
    )
    img.set_clip_path(Polygon(poly_xy, closed=True, transform=ax.transData))


def moving_average_smooth(
    y: Sequence[float] | np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Smooth a 1D series while preserving NaN positions."""

    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if n < 3 or window < 3:
        return y_arr

    w = min(int(window), n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y_arr

    finite_mask = np.isfinite(y_arr)
    if not finite_mask.any():
        return y_arr

    idx = np.arange(n)
    y_clean = y_arr.copy()
    if (~finite_mask).any():
        y_clean[~finite_mask] = np.interp(
            idx[~finite_mask], idx[finite_mask], y_arr[finite_mask]
        )

    pad = w // 2
    padded = np.concatenate([y_clean[:pad][::-1], y_clean, y_clean[-pad:][::-1]])
    kernel = np.ones(w, dtype=float) / w
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[~finite_mask] = np.nan
    return smoothed


def panel_title(
    ax,
    theme: Theme,
    title: str,
    subtitle: str = "",
    caption: str = "",
) -> None:
    ax.text(
        0.0,
        1.52,
        title,
        transform=ax.transAxes,
        fontsize=17,
        color=theme.text_primary,
        fontweight=900,
    )
    if subtitle:
        ax.text(
            0.0,
            1.34,
            subtitle,
            transform=ax.transAxes,
            fontsize=13,
            color=theme.text_secondary,
        )
    if caption:
        ax.text(
            0.0,
            1.18,
            caption,
            transform=ax.transAxes,
            fontsize=11,
            color=theme.text_tertiary,
            linespacing=1.4,
        )
