from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_metrics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def trained_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "trained"]


def normalized_iteration_rows(rows: list[dict]) -> list[dict]:
    by_iteration: dict[int, dict] = {}
    without_iteration: list[dict] = []

    for row in rows:
        iteration = row.get("iteration")
        if isinstance(iteration, int):
            by_iteration[iteration] = row
        else:
            without_iteration.append(row)

    return without_iteration + [by_iteration[i] for i in sorted(by_iteration)]


def smooth_curve(x_values: list[int], y_values: list[float], points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]
    if len(unique_x) < 3:
        return unique_x, unique_y

    dense_x = np.linspace(unique_x[0], unique_x[-1], max(points, len(unique_x)))
    dense_y = np.interp(dense_x, unique_x, unique_y)
    window = max(3, min(9, len(unique_x) // 4 * 2 + 1))
    if window >= 3:
        kernel = np.ones(window, dtype=float) / window
        pad = window // 2
        padded = np.pad(dense_y, (pad, pad), mode="edge")
        dense_y = np.convolve(padded, kernel, mode="valid")
    return dense_x, dense_y


def plot_curve(ax, x_values: list[int], y_values: list[float], **kwargs) -> None:
    x, y = smooth_curve(x_values, y_values)
    ax.plot(x, y, **kwargs)


def plot(metrics_path: Path, output_path: Path | None, show: bool) -> None:
    rows = load_metrics(metrics_path)
    if not rows:
        print(f"No data in {metrics_path}")
        return
    rows = normalized_iteration_rows(rows)

    trained = trained_rows(rows)

    iters = [r["iteration"] for r in rows]
    elapsed = [r.get("elapsed_hours", 0.0) for r in rows]

    t_iters = [r["iteration"] for r in trained]
    train_loss = [r.get("train_loss", float("nan")) for r in trained]
    policy_loss = [r.get("policy_loss", float("nan")) for r in trained]
    value_loss = [r.get("value_loss", float("nan")) for r in trained]
    learning_rate = [r.get("learning_rate", float("nan")) for r in trained]
    arena_win_rate = [r.get("arena_win_rate", float("nan")) for r in trained]
    accepted_iters = [r["iteration"] for r in trained if r.get("accepted")]

    avg_moves = [r.get("selfplay_avg_moves", float("nan")) for r in rows]
    replay_games = [r.get("replay_games", 0) for r in rows]

    best_iter = rows[-1].get("best_iteration", 0)
    best_win_rate = rows[-1].get("best_arena_win_rate", 0.0)

    fig, axes = plt.subplots(8, 1, figsize=(10, 21), sharex=True)
    fig.suptitle(
        f"{metrics_path.parent.name}  —  iter {iters[-1]}  |  best iter {best_iter}  |  best win rate {best_win_rate:.3f}  |  {elapsed[-1]:.2f}h",
        fontsize=12,
        fontweight="bold",
    )

    # 1. Total loss
    ax = axes[0]
    plot_curve(ax, t_iters, train_loss, color="steelblue", linewidth=1.5, label="total")
    ax.set_title("Train Loss")
    ax.set_xlabel("iteration")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Policy loss
    ax = axes[1]
    plot_curve(ax, t_iters, policy_loss, color="darkorange", linewidth=1.5, label="policy")
    ax.set_title("Policy Loss")
    ax.set_xlabel("iteration")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Value loss
    ax = axes[2]
    plot_curve(ax, t_iters, value_loss, color="seagreen", linewidth=1.5, label="value")
    ax.set_title("Value Loss")
    ax.set_xlabel("iteration")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Learning rate
    ax = axes[3]
    plot_curve(ax, t_iters, learning_rate, color="crimson", linewidth=1.5, label="learning rate")
    ax.set_title("Learning Rate")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Arena win rate
    ax = axes[4]
    if t_iters:
        plot_curve(ax, t_iters, arena_win_rate, color="mediumpurple", linewidth=1.5, label="win rate")
        accept_wr = [arena_win_rate[t_iters.index(i)] for i in accepted_iters if i in t_iters]
        if accepted_iters:
            ax.scatter(accepted_iters, accept_wr, color="gold", zorder=5, s=40, label="accepted")
        accept_threshold = trained[0].get("arena_accept_win_rate") if trained else None
        if accept_threshold is not None:
            ax.axhline(accept_threshold, color="gray", linestyle="--", linewidth=0.8, label=f"threshold {accept_threshold}")
    ax.set_title("Arena Win Rate")
    ax.set_xlabel("iteration")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Selfplay avg moves
    ax = axes[5]
    plot_curve(ax, iters, avg_moves, color="tomato", linewidth=1.5)
    ax.set_title("Selfplay Avg Moves")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    # 7. Replay buffer games
    ax = axes[6]
    plot_curve(ax, iters, replay_games, color="teal", linewidth=1.5)
    ax.set_title("Replay Buffer Games")
    ax.set_xlabel("iteration")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, alpha=0.3)

    # 8. Elapsed hours vs iteration
    ax = axes[7]
    plot_curve(ax, iters, elapsed, color="slategray", linewidth=1.5)
    ax.set_title("Elapsed Hours")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.tick_params(axis="x", which="both", labelbottom=True)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot omok training metrics from metrics.jsonl.")
    parser.add_argument("metrics", type=str, help="Path to metrics.jsonl or checkpoint directory")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PNG path (default: <checkpoint_dir>/metrics.png)")
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window")
    args = parser.parse_args()

    path = Path(args.metrics)
    if path.is_dir():
        metrics_path = path / "metrics.jsonl"
    else:
        metrics_path = path

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = metrics_path.parent / "metrics.png"

    plot(metrics_path, output_path if not args.show or args.output else None, args.show)


if __name__ == "__main__":
    main()
