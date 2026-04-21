from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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


def infer_board_size(rows: Sequence[dict], metrics_path: Path) -> int:
    for row in reversed(rows):
        raw = row.get("board_size")
        if raw is not None:
            return int(raw)

    for sidecar_name in ("latest.json", "best.json", "iter_0000.json"):
        sidecar_path = metrics_path.parent / sidecar_name
        if not sidecar_path.exists():
            continue
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        config = payload.get("config", {})
        rules = config.get("rules", {})
        raw = rules.get("board_size")
        if raw is not None:
            return int(raw)

    return 9


def field(rows: Sequence[dict], key: str, default: float = float("nan")) -> list[float]:
    return [r.get(key, default) for r in rows]


def summed_field(rows: Sequence[dict], *keys: str) -> list[float]:
    values = []
    for row in rows:
        total = 0.0
        seen = False
        for key in keys:
            value = row.get(key)
            if value is not None:
                total += float(value)
                seen = True
        values.append(total if seen else float("nan"))
    return values


def rate(numerators: Sequence[float], denominators: Sequence[float]) -> list[float]:
    values = []
    for numerator, denominator in zip(numerators, denominators, strict=False):
        if denominator and np.isfinite(denominator) and denominator > 0:
            values.append(float(numerator) / float(denominator))
        else:
            values.append(float("nan"))
    return values


def smooth_curve(x_values: Sequence[float], y_values: Sequence[float], points: int = 300) -> tuple[np.ndarray, np.ndarray]:
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


def plot_curve(ax, x_values: Sequence[float], y_values: Sequence[float], **kwargs) -> None:
    x, y = smooth_curve(x_values, y_values)
    ax.plot(x, y, **kwargs)


def iteration_tick_steps(axes: Sequence) -> tuple[float, float]:
    max_iter = 0.0
    for ax in axes:
        right = ax.get_xlim()[1]
        if np.isfinite(right):
            max_iter = max(max_iter, right)

    if max_iter <= 20:
        return 1, 0.5
    if max_iter <= 100:
        return 10, 2
    if max_iter <= 300:
        return 25, 5
    if max_iter <= 1000:
        return 50, 10
    return 100, 25


def finish_figure(fig, axes: Sequence, output_path: Path | None, show: bool) -> None:
    major_step, minor_step = iteration_tick_steps(axes)
    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.grid(True, which="minor", axis="x", alpha=0.12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_overview(rows: list[dict], metrics_path: Path, output_path: Path | None, show: bool) -> None:
    trained = trained_rows(rows)
    iters = field(rows, "iteration")
    elapsed = field(rows, "elapsed_hours", 0.0)
    t_iters = field(trained, "iteration")
    board_size = infer_board_size(rows, metrics_path)
    uniform_policy_entropy = float(np.log(board_size * board_size))

    best_iter = rows[-1].get("best_iteration", 0)
    best_win_rate = rows[-1].get("best_arena_win_rate", 0.0)

    fig, axes = plt.subplots(9, 1, figsize=(10, 23), sharex=True)
    fig.suptitle(
        f"{metrics_path.parent.name}  |  iter {int(iters[-1])}  |  best iter {best_iter}  |  "
        f"best win rate {best_win_rate:.3f}  |  {elapsed[-1]:.2f}h",
        fontsize=12,
        fontweight="bold",
    )

    ax = axes[0]
    plot_curve(ax, t_iters, field(trained, "train_loss"), color="steelblue", linewidth=1.5, label="total")
    ax.set_title("Train Loss")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    plot_curve(ax, t_iters, field(trained, "policy_loss"), color="darkorange", linewidth=1.5, label="policy")
    ax.axhline(
        uniform_policy_entropy,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"uniform ln({board_size * board_size})",
    )
    ax.set_title("Policy Loss")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    plot_curve(ax, t_iters, field(trained, "value_loss"), color="seagreen", linewidth=1.5, label="value")
    ax.set_title("Value Loss")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    plot_curve(ax, t_iters, field(trained, "learning_rate"), color="crimson", linewidth=1.5, label="learning rate")
    ax.set_title("Learning Rate")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    arena_win_rate = field(trained, "arena_win_rate")
    accepted_iters = [r["iteration"] for r in trained if r.get("accepted")]
    plot_curve(ax, t_iters, [v * 100 for v in arena_win_rate], color="mediumpurple", linewidth=1.5, label="win rate")
    if accepted_iters:
        accept_wr = [arena_win_rate[t_iters.index(i)] * 100 for i in accepted_iters if i in t_iters]
        ax.scatter(accepted_iters, accept_wr, color="gold", zorder=5, s=40, label="accepted")
    accept_threshold = trained[0].get("arena_accept_win_rate") if trained else None
    if accept_threshold is not None:
        ax.axhline(accept_threshold * 100, color="gray", linestyle="--", linewidth=0.8, label=f"threshold {accept_threshold:.0%}")
    ax.set_title("Arena Win Rate")
    ax.set_xlabel("iteration")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    plot_curve(ax, t_iters, [v * 100 for v in field(trained, "arena_candidate_white_win_rate")], color="saddlebrown", linewidth=1.5, label="white win rate")
    white_threshold = trained[0].get("arena_white_win_rate_threshold") if trained else None
    if white_threshold is not None:
        ax.axhline(white_threshold * 100, color="gray", linestyle="--", linewidth=0.8, label=f"threshold {white_threshold:.0%}")
    ax.set_title("Arena Candidate White Win Rate")
    ax.set_xlabel("iteration")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[6]
    plot_curve(ax, iters, field(rows, "selfplay_avg_moves"), color="tomato", linewidth=1.5)
    ax.set_title("Selfplay Avg Moves")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[7]
    plot_curve(ax, iters, field(rows, "replay_games", 0.0), color="teal", linewidth=1.5)
    ax.set_title("Replay Buffer Games")
    ax.set_xlabel("iteration")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, alpha=0.3)

    ax = axes[8]
    plot_curve(ax, iters, elapsed, color="slategray", linewidth=1.5)
    ax.set_title("Elapsed Hours")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    finish_figure(fig, axes, output_path, show)


def plot_timing(rows: list[dict], metrics_path: Path, output_path: Path | None, show: bool) -> None:
    trained = trained_rows(rows)
    iters = field(rows, "iteration")
    t_iters = field(trained, "iteration")

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(f"{metrics_path.parent.name} timing", fontsize=12, fontweight="bold")

    ax = axes[0]
    plot_curve(ax, iters, field(rows, "duration_seconds"), color="black", linewidth=1.5, label="total")
    plot_curve(ax, iters, field(rows, "selfplay_seconds"), color="dodgerblue", linewidth=1.2, label="selfplay")
    plot_curve(ax, iters, field(rows, "train_seconds"), color="seagreen", linewidth=1.2, label="train")
    plot_curve(ax, iters, field(rows, "arena_seconds"), color="mediumpurple", linewidth=1.2, label="arena")
    ax.set_title("Phase Seconds")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    plot_curve(ax, t_iters, field(trained, "train_sample_seconds"), color="teal", linewidth=1.2, label="sample")
    plot_curve(ax, t_iters, field(trained, "train_forward_seconds"), color="steelblue", linewidth=1.2, label="forward")
    plot_curve(ax, t_iters, field(trained, "train_loss_seconds"), color="gray", linewidth=1.2, label="loss")
    plot_curve(ax, t_iters, field(trained, "train_backward_seconds"), color="tomato", linewidth=1.2, label="backward")
    plot_curve(ax, t_iters, field(trained, "train_optimizer_seconds"), color="darkorange", linewidth=1.2, label="optimizer")
    ax.set_title("Training Step Seconds")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    plot_curve(ax, iters, summed_field(rows, "search_selfplay_candidate_seconds", "search_selfplay_best_seconds"), color="dodgerblue", linewidth=1.5, label="selfplay")
    plot_curve(ax, iters, summed_field(rows, "search_arena_candidate_seconds", "search_arena_best_seconds"), color="mediumpurple", linewidth=1.5, label="arena")
    ax.set_title("MCTS Search Seconds")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    plot_curve(ax, iters, summed_field(rows, "eval_selfplay_candidate_seconds", "eval_selfplay_best_seconds"), color="dodgerblue", linewidth=1.5, label="selfplay")
    plot_curve(ax, iters, summed_field(rows, "eval_arena_candidate_seconds", "eval_arena_best_seconds"), color="mediumpurple", linewidth=1.5, label="arena")
    ax.set_title("Neural Evaluator Seconds")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    plot_curve(ax, iters, field(rows, "checkpoint_seconds"), color="slategray", linewidth=1.5)
    ax.set_title("Checkpoint Seconds")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    finish_figure(fig, axes, output_path, show)


def plot_throughput(rows: list[dict], metrics_path: Path, output_path: Path | None, show: bool) -> None:
    trained = trained_rows(rows)
    iters = field(rows, "iteration")
    t_iters = field(trained, "iteration")

    selfplay_eval_seconds = summed_field(rows, "eval_selfplay_candidate_seconds", "eval_selfplay_best_seconds")
    arena_eval_seconds = summed_field(rows, "eval_arena_candidate_seconds", "eval_arena_best_seconds")
    selfplay_eval_positions = summed_field(rows, "eval_selfplay_candidate_positions", "eval_selfplay_best_positions")
    arena_eval_positions = summed_field(rows, "eval_arena_candidate_positions", "eval_arena_best_positions")
    selfplay_search_seconds = summed_field(rows, "search_selfplay_candidate_seconds", "search_selfplay_best_seconds")
    arena_search_seconds = summed_field(rows, "search_arena_candidate_seconds", "search_arena_best_seconds")
    selfplay_search_states = summed_field(rows, "search_selfplay_candidate_states", "search_selfplay_best_states")
    arena_search_states = summed_field(rows, "search_arena_candidate_states", "search_arena_best_states")

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(f"{metrics_path.parent.name} throughput", fontsize=12, fontweight="bold")

    ax = axes[0]
    plot_curve(ax, iters, rate(field(rows, "selfplay_games", 0.0), field(rows, "selfplay_seconds")), color="dodgerblue", linewidth=1.5)
    ax.set_title("Selfplay Games / Second")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    plot_curve(ax, t_iters, rate(field(trained, "train_metric_samples", 0.0), field(trained, "train_seconds")), color="seagreen", linewidth=1.5, label="samples/sec")
    plot_curve(ax, t_iters, rate(field(trained, "updates_done", 0.0), field(trained, "train_seconds")), color="darkorange", linewidth=1.5, label="updates/sec")
    ax.set_title("Training Throughput")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    plot_curve(ax, iters, rate(selfplay_eval_positions, selfplay_eval_seconds), color="dodgerblue", linewidth=1.5, label="selfplay")
    plot_curve(ax, iters, rate(arena_eval_positions, arena_eval_seconds), color="mediumpurple", linewidth=1.5, label="arena")
    ax.set_title("Evaluator Positions / Second")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    plot_curve(ax, iters, rate(selfplay_search_states, selfplay_search_seconds), color="dodgerblue", linewidth=1.5, label="selfplay")
    plot_curve(ax, iters, rate(arena_search_states, arena_search_seconds), color="mediumpurple", linewidth=1.5, label="arena")
    ax.set_title("MCTS Search States / Second")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    plot_curve(ax, iters, field(rows, "eval_selfplay_candidate_avg_batch"), color="darkcyan", linewidth=1.5, label="candidate")
    plot_curve(ax, iters, field(rows, "eval_selfplay_best_avg_batch"), color="slateblue", linewidth=1.5, label="best")
    ax.set_title("Selfplay Evaluator Avg Batch")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    finish_figure(fig, axes, output_path, show)


def plot(metrics_path: Path, output_path: Path | None, show: bool, *, timing: bool = False, throughput: bool = False) -> None:
    rows = load_metrics(metrics_path)
    if not rows:
        print(f"No data in {metrics_path}")
        return
    rows = normalized_iteration_rows(rows)

    plot_overview(metrics_path=metrics_path, rows=rows, output_path=output_path, show=show)
    if timing:
        target = None if output_path is None else output_path.with_name(output_path.stem + "_timing" + output_path.suffix)
        plot_timing(metrics_path=metrics_path, rows=rows, output_path=target, show=show)
    if throughput:
        target = None if output_path is None else output_path.with_name(output_path.stem + "_throughput" + output_path.suffix)
        plot_throughput(metrics_path=metrics_path, rows=rows, output_path=target, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot omok training metrics from metrics.jsonl.")
    parser.add_argument("metrics", type=str, help="Path to metrics.jsonl or checkpoint directory")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PNG path (default: <checkpoint_dir>/metrics.png)")
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window")
    parser.add_argument("--timing", action="store_true", help="Also write timing breakdown plot")
    parser.add_argument("--throughput", action="store_true", help="Also write throughput plot")
    parser.add_argument("--all", action="store_true", help="Write overview, timing, and throughput plots")
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

    write_output = output_path if not args.show or args.output else None
    plot(
        metrics_path,
        write_output,
        args.show,
        timing=args.timing or args.all,
        throughput=args.throughput or args.all,
    )


if __name__ == "__main__":
    main()
