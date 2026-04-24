from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metrics(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    checkpoint_path = Path(checkpoint_dir)
    metrics_path = checkpoint_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found under checkpoint dir: {checkpoint_path}")
    metrics: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                metrics.append(payload)
    return sorted(
        metrics,
        key=lambda item: (
            0 if "iteration" in item else 1,
            int(item.get("iteration", 0)),
        ),
    )


def load_runtime_progress(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    progress_path = Path(checkpoint_dir) / "runtime_progress.json"
    if not progress_path.exists():
        return None
    return json.loads(progress_path.read_text(encoding="utf-8"))


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    latest = metrics[-1]
    summary: dict[str, Any] = {}
    for key in (
        "iteration",
        "elapsed_seconds",
        "total_nodes",
        "nodes_per_second",
        "avg_nodes_per_traversal",
        "cutoff_rate",
        "node_limit_cutoff_rate",
        "advantage_loss_p0",
        "advantage_loss_p1",
        "strategy_loss",
        "eval_random_win_rate",
        "eval_random_avg_diff",
        "eval_safe_heuristic_win_rate",
        "eval_safe_heuristic_avg_diff",
        "advantage_memory_size_p0",
        "advantage_memory_size_p1",
        "strategy_memory_size",
    ):
        if key in latest:
            summary[key] = latest[key]
    return summary


def _metrics_series(metrics: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for index, item in enumerate(metrics):
        if key not in item:
            continue
        xs.append(int(item.get("iteration", index + 1)))
        ys.append(float(item[key]))
    return xs, ys


def plot_metrics(checkpoint_dir: str | Path, output: str | Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = load_metrics(checkpoint_dir)
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output) if output is not None else checkpoint_path / "training_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Losses", "Iteration", [("advantage_loss_p0", "advantage_loss_p0"), ("advantage_loss_p1", "advantage_loss_p1"), ("strategy_loss", "strategy_loss")]),
        ("Evaluation Win Rates", "Iteration", [("eval_random_win_rate", "eval_random_win_rate"), ("eval_safe_heuristic_win_rate", "eval_safe_heuristic_win_rate")]),
        ("Evaluation Avg Diff", "Iteration", [("eval_random_avg_diff", "eval_random_avg_diff"), ("eval_safe_heuristic_avg_diff", "eval_safe_heuristic_avg_diff")]),
        ("Throughput", "Iteration", [("nodes_per_second", "nodes_per_second"), ("avg_nodes_per_traversal", "avg_nodes_per_traversal")]),
        ("Cutoffs", "Iteration", [("cutoff_rate", "cutoff_rate"), ("node_limit_cutoff_rate", "node_limit_cutoff_rate")]),
        ("Memory Sizes", "Iteration", [("advantage_memory_size_p0", "advantage_memory_size_p0"), ("advantage_memory_size_p1", "advantage_memory_size_p1"), ("strategy_memory_size", "strategy_memory_size")]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes_flat = list(axes.flat)
    for axis, (title, xlabel, series_specs) in zip(axes_flat, panels, strict=True):
        plotted = False
        for label, key in series_specs:
            xs, ys = _metrics_series(metrics, key)
            if not xs:
                continue
            axis.plot(xs, ys, label=label)
            plotted = True
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
    fig.suptitle(f"Lost Cities Deep CFR Metrics: {checkpoint_path}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
