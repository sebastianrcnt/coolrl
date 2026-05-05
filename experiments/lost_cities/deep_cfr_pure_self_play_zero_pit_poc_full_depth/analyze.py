#!/usr/bin/env python3
"""Lost Cities zero-pit POC 실험 전용 metrics 리포트 스크립트."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_OUTPUT = EXPERIMENT_DIR / "report.json"
DEFAULT_MARKDOWN_OUTPUT = EXPERIMENT_DIR / "report.md"
DEFAULT_PLOT_OUTPUT = EXPERIMENT_DIR / "analysis_metrics.png"
DEFAULT_HEATMAP_OUTPUT = EXPERIMENT_DIR / "analysis_latest_heatmap.png"
DEFAULT_DELTA_HEATMAP_OUTPUT = EXPERIMENT_DIR / "analysis_delta_heatmap.png"

TARGET_METRICS = (
    "win_rate",
    "avg_diff",
    "play_action_rate",
    "discard_action_rate",
    "draw_deck_rate",
    "draw_pile_rate",
    "avg_opened_colors",
    "avg_expedition_cards",
    "max_step_timeouts",
    "policy_entropy",
)

METRIC_LABELS = {
    "win_rate": "승률",
    "avg_diff": "평균 점수차",
    "play_action_rate": "play 비율",
    "discard_action_rate": "discard 비율",
    "draw_deck_rate": "덱 드로우 비율",
    "draw_pile_rate": "파일 드로우 비율",
    "avg_opened_colors": "평균 개방 색",
    "avg_expedition_cards": "평균 원정 카드",
    "max_step_timeouts": "max step timeout",
    "policy_entropy": "policy entropy",
}

HEATMAP_METRICS = (
    "win_rate",
    "avg_diff",
    "play_action_rate",
    "discard_action_rate",
    "draw_deck_rate",
    "draw_pile_rate",
    "avg_opened_colors",
    "max_step_timeouts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lost Cities zero-pit POC metrics를 opponent별로 리포트합니다."
    )
    parser.add_argument("--run", required=True, type=Path, help="metrics.jsonl이 있는 run directory")
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=None,
        help="비교 기준 metrics.jsonl이 있는 run directory",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"JSON 리포트 저장 경로. 기본값은 {DEFAULT_JSON_OUTPUT}",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help=f"Markdown 리포트 저장 경로. 기본값은 {DEFAULT_MARKDOWN_OUTPUT}",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help=f"PNG plot 저장 경로. 기본값은 {DEFAULT_PLOT_OUTPUT}",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=DEFAULT_HEATMAP_OUTPUT,
        help=f"최신 eval heatmap PNG 저장 경로. 기본값은 {DEFAULT_HEATMAP_OUTPUT}",
    )
    parser.add_argument(
        "--delta-heatmap-output",
        type=Path,
        default=DEFAULT_DELTA_HEATMAP_OUTPUT,
        help=f"baseline delta heatmap PNG 저장 경로. 기본값은 {DEFAULT_DELTA_HEATMAP_OUTPUT}",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="report.md와 report.json을 갱신함. 기본 실행은 기록 파일을 덮어쓰지 않음",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="plot에만 적용할 이동 평균 window. 1이면 raw metrics를 그대로 그림",
    )
    parser.add_argument("--no-plot", action="store_true", help="plot 생성을 건너뜀")
    return parser.parse_args()


def read_metrics(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl을 찾을 수 없습니다: {metrics_path}")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{metrics_path}:{line_number}: JSON 파싱 실패 ({exc.msg})")
                continue
            if not isinstance(row, dict):
                warnings.append(f"{metrics_path}:{line_number}: object가 아닌 row를 건너뜀")
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"유효한 metric row가 없습니다: {metrics_path}")
    return rows, warnings


def metric_series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for index, row in enumerate(rows):
        value = numeric(row.get(key))
        if value is None:
            continue
        xs.append(int(row.get("iteration", index + 1)))
        ys.append(value)
    return xs, ys


def has_eval_win_rate(row: dict[str, Any]) -> bool:
    return any(key.startswith("eval_") and key.endswith("_win_rate") for key in row)


def find_latest_eval_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        if has_eval_win_rate(row):
            return row
    return None


def metric_key(opponent: str, metric: str) -> str:
    return f"eval_{opponent}_{metric}"


def discover_opponents(eval_row: dict[str, Any] | None) -> list[str]:
    if eval_row is None:
        return []

    opponents: set[str] = set()
    suffixes = sorted((f"_{metric}" for metric in TARGET_METRICS), key=len, reverse=True)
    for key in eval_row:
        if not key.startswith("eval_"):
            continue
        body = key[len("eval_") :]
        for suffix in suffixes:
            if body.endswith(suffix):
                opponent = body[: -len(suffix)]
                if opponent:
                    opponents.add(opponent)
                break
    return sorted(opponents)


def extract_opponents(eval_row: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    opponents: dict[str, dict[str, Any]] = {}
    if eval_row is None:
        return opponents

    for opponent in discover_opponents(eval_row):
        opponents[opponent] = {
            metric: normalize_value(eval_row.get(metric_key(opponent, metric)))
            for metric in TARGET_METRICS
        }
    return opponents


def normalize_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize_run(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    rows, warnings = read_metrics(run_dir)
    latest_row = rows[-1]
    latest_eval_row = find_latest_eval_row(rows)

    summary = {
        "run": {
            "path": str(run_dir),
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "row_count": len(rows),
        },
        "latest_iteration": latest_row.get("iteration"),
        "latest_eval_iteration": latest_eval_row.get("iteration") if latest_eval_row else None,
        "opponents": extract_opponents(latest_eval_row),
    }
    return summary, warnings


def compute_deltas(
    current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for opponent in sorted(set(current) & set(baseline)):
        metric_deltas: dict[str, float] = {}
        for metric in TARGET_METRICS:
            current_value = numeric(current[opponent].get(metric))
            baseline_value = numeric(baseline[opponent].get(metric))
            if current_value is None or baseline_value is None:
                continue
            metric_deltas[metric] = current_value - baseline_value
        if metric_deltas:
            deltas[opponent] = metric_deltas
    return deltas


def format_value(value: Any) -> str:
    number = numeric(value)
    if number is None:
        return "-"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def format_delta(value: Any) -> str:
    number = numeric(value)
    if number is None:
        return "-"
    sign = "+" if number > 0 else ""
    return f"{sign}{format_value(number)}"


def render_stdout(summary: dict[str, Any], deltas: dict[str, dict[str, float]] | None) -> str:
    lines = [
        "Lost Cities zero-pit 분석 리포트",
        f"- run: {summary['run']['path']}",
        f"- 최신 row iteration: {summary['latest_iteration']}",
        f"- 최신 eval iteration: {summary['latest_eval_iteration']}",
    ]

    opponents = summary["opponents"]
    if not opponents:
        lines.append("- 최신 eval row를 찾지 못했습니다.")
        return "\n".join(lines)

    lines.append("- opponent별 핵심 지표:")
    for opponent, metrics in opponents.items():
        delta_part = ""
        if deltas and opponent in deltas and "win_rate" in deltas[opponent]:
            delta_part = f", 승률 delta {format_delta(deltas[opponent]['win_rate'])}"
        lines.append(
            f"  - {opponent}: 승률 {format_value(metrics.get('win_rate'))}, "
            f"평균 점수차 {format_value(metrics.get('avg_diff'))}, "
            f"play 비율 {format_value(metrics.get('play_action_rate'))}, "
            f"timeout {format_value(metrics.get('max_step_timeouts'))}{delta_part}"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any], payload: dict[str, Any]) -> str:
    baseline = payload.get("baseline")
    deltas = payload.get("deltas", {})
    lines = [
        "# Lost Cities zero-pit 분석 리포트",
        "",
        f"- run: `{summary['run']['path']}`",
        f"- 최신 row iteration: `{summary['latest_iteration']}`",
        f"- 최신 eval iteration: `{summary['latest_eval_iteration']}`",
    ]
    if baseline:
        lines.extend(
            [
                f"- baseline run: `{baseline['run']['path']}`",
                f"- baseline 최신 eval iteration: `{baseline['latest_eval_iteration']}`",
            ]
        )
    lines.append("")

    opponents = summary["opponents"]
    if not opponents:
        lines.append("최신 eval row를 찾지 못했습니다.")
        lines.append("")
        return "\n".join(lines)

    headers = ["opponent", *[METRIC_LABELS[metric] for metric in TARGET_METRICS]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for opponent, metrics in opponents.items():
        row = [opponent, *[format_value(metrics.get(metric)) for metric in TARGET_METRICS]]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if deltas:
        delta_headers = ["opponent", *[f"{METRIC_LABELS[metric]} delta" for metric in TARGET_METRICS]]
        lines.append("## baseline 대비 delta")
        lines.append("")
        lines.append("| " + " | ".join(delta_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(delta_headers)) + " |")
        for opponent in sorted(deltas):
            row = [
                opponent,
                *[format_delta(deltas[opponent].get(metric)) for metric in TARGET_METRICS],
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def smooth_values(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return values
    half_window = max(1, window // 2)
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        window_values = values[start:end]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed


def plot_records(
    rows: list[dict[str, Any]],
    series_specs: list[tuple[str, str]],
    *,
    percent: bool = False,
    smooth_window: int = 1,
) -> dict[str, list[Any]]:
    records: dict[str, list[Any]] = {"iteration": [], "value": [], "series": []}
    for label, key in series_specs:
        xs, ys = metric_series(rows, key)
        if not xs:
            continue
        values = [value * 100 for value in ys] if percent else ys
        values = smooth_values(values, smooth_window)
        for iteration, value in zip(xs, values, strict=True):
            records["iteration"].append(iteration)
            records["value"].append(value)
            records["series"].append(label)
    return records


def draw_panel(
    sns: Any,
    axis: Any,
    rows: list[dict[str, Any]],
    title: str,
    series_specs: list[tuple[str, str]],
    *,
    ylabel: str,
    percent: bool = False,
    smooth_window: int = 1,
) -> None:
    records = plot_records(rows, series_specs, percent=percent, smooth_window=smooth_window)
    axis.set_title(title, fontsize=11, fontweight="bold")
    axis.set_xlabel("iteration")
    axis.set_ylabel(ylabel)
    if not records["iteration"]:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        return
    sns.lineplot(
        data=records,
        x="iteration",
        y="value",
        hue="series",
        ax=axis,
        linewidth=1.5,
        alpha=0.85,
        marker="o",
        markersize=2.5,
        errorbar=None,
    )
    axis.grid(True, alpha=0.35)
    legend = axis.legend(
        title=None,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
        fontsize=8,
    )
    if legend is not None:
        for line in legend.get_lines():
            line.set_linewidth(1.8)


def endpoint_bucket_sort_key(key: str) -> tuple[int, int]:
    label = key.removeprefix("endpoint_depth_bucket_")
    start_token = label.split("_", maxsplit=1)[0]
    try:
        start = int(start_token)
    except ValueError:
        start = 10**9
    return start, 1 if label.endswith("_plus") else 0


def latest_endpoint_bucket_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        if any(key.startswith("endpoint_depth_bucket_") for key in row):
            return row
    return None


def draw_latest_endpoint_bucket_panel(sns: Any, axis: Any, rows: list[dict[str, Any]]) -> None:
    axis.set_title("Latest Endpoint Depth Buckets", fontsize=11, fontweight="bold")
    axis.set_xlabel("endpoint depth")
    axis.set_ylabel("traversals")

    row = latest_endpoint_bucket_row(rows)
    if row is None:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        return

    keys = sorted(
        (key for key in row if key.startswith("endpoint_depth_bucket_")),
        key=endpoint_bucket_sort_key,
    )
    labels = [key.removeprefix("endpoint_depth_bucket_").replace("_", "-") for key in keys]
    values = [numeric(row.get(key)) or 0.0 for key in keys]
    records = {"bucket": labels, "traversals": values}
    sns.barplot(data=records, x="bucket", y="traversals", ax=axis, color="#4C78A8")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(True, axis="y", alpha=0.35)


def plot_run(run_dir: Path, output: Path | None = None, *, smooth_window: int = 1) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows, _warnings = read_metrics(run_dir)
    output_path = output if output is not None else DEFAULT_PLOT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latest_eval_row = find_latest_eval_row(rows)
    opponents = discover_opponents(latest_eval_row)
    sns.set_theme(style="darkgrid", context="notebook")
    fig, axes = plt.subplots(5, 3, figsize=(21, 20), sharex=False, sharey=False)
    fig.suptitle(
        f"Lost Cities zero-pit metrics: {run_dir.name}",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    if smooth_window > 1:
        fig.text(0.01, 0.985, f"smooth window: {smooth_window}", fontsize=9, va="top")

    panels = [
        (
            "Advantage Loss",
            [("p0", "advantage_loss_p0"), ("p1", "advantage_loss_p1")],
            "loss",
            False,
        ),
        (
            "Strategy Loss",
            [("strategy", "strategy_loss")],
            "loss",
            False,
        ),
        (
            "Win Rate",
            [(opponent, metric_key(opponent, "win_rate")) for opponent in opponents],
            "win rate (%)",
            True,
        ),
        (
            "Avg Diff",
            [(opponent, metric_key(opponent, "avg_diff")) for opponent in opponents],
            "score diff",
            False,
        ),
        (
            "Play Action Rate",
            [(opponent, metric_key(opponent, "play_action_rate")) for opponent in opponents],
            "rate (%)",
            True,
        ),
        (
            "Discard Action Rate",
            [(opponent, metric_key(opponent, "discard_action_rate")) for opponent in opponents],
            "rate (%)",
            True,
        ),
        (
            "Draw Deck Rate",
            [(opponent, metric_key(opponent, "draw_deck_rate")) for opponent in opponents],
            "rate (%)",
            True,
        ),
        (
            "Draw Pile Rate",
            [(opponent, metric_key(opponent, "draw_pile_rate")) for opponent in opponents],
            "rate (%)",
            True,
        ),
        (
            "Opened Colors",
            [(opponent, metric_key(opponent, "avg_opened_colors")) for opponent in opponents],
            "colors",
            False,
        ),
        (
            "Expedition Cards",
            [(opponent, metric_key(opponent, "avg_expedition_cards")) for opponent in opponents],
            "cards",
            False,
        ),
        (
            "Max Step Timeouts",
            [(opponent, metric_key(opponent, "max_step_timeouts")) for opponent in opponents],
            "timeouts",
            False,
        ),
        (
            "Policy Entropy",
            [(opponent, metric_key(opponent, "policy_entropy")) for opponent in opponents],
            "entropy",
            False,
        ),
        (
            "Traversal Depth",
            [
                ("avg endpoint depth", "avg_endpoint_depth"),
                ("max depth reached", "max_depth_reached"),
                ("avg nodes/traversal", "avg_nodes_per_traversal"),
            ],
            "depth / nodes",
            False,
        ),
        (
            "Traversal Endpoint Rates",
            [
                ("terminal", "terminal_traversal_rate"),
                ("node limit cutoff", "node_limit_cutoff_traversal_rate"),
                ("depth cutoff", "depth_cutoff_traversal_rate"),
            ],
            "rate (%)",
            True,
        ),
    ]

    flat_axes = list(axes.flat)
    for axis, (title, specs, ylabel, percent) in zip(flat_axes[: len(panels)], panels, strict=True):
        draw_panel(
            sns,
            axis,
            rows,
            title,
            specs,
            ylabel=ylabel,
            percent=percent,
            smooth_window=smooth_window,
        )
    draw_latest_endpoint_bucket_panel(sns, flat_axes[len(panels)], rows)

    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=5.0, h_pad=2.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return output_path


def heatmap_matrix(
    opponents: dict[str, dict[str, Any]],
    metrics: tuple[str, ...] = HEATMAP_METRICS,
) -> tuple[list[str], list[str], list[list[float]], list[list[str]]]:
    names = sorted(opponents)
    labels = list(metrics)
    values: list[list[float]] = []
    annotations: list[list[str]] = []
    for opponent in names:
        value_row: list[float] = []
        annotation_row: list[str] = []
        for metric in metrics:
            value = numeric(opponents[opponent].get(metric))
            value_row.append(float("nan") if value is None else value)
            annotation_row.append(format_value(value))
        values.append(value_row)
        annotations.append(annotation_row)
    return names, labels, values, annotations


def delta_heatmap_matrix(
    deltas: dict[str, dict[str, float]],
    metrics: tuple[str, ...] = HEATMAP_METRICS,
) -> tuple[list[str], list[str], list[list[float]], list[list[str]]]:
    names = sorted(deltas)
    labels = [f"{metric} delta" for metric in metrics]
    values: list[list[float]] = []
    annotations: list[list[str]] = []
    for opponent in names:
        value_row: list[float] = []
        annotation_row: list[str] = []
        for metric in metrics:
            value = numeric(deltas[opponent].get(metric))
            value_row.append(float("nan") if value is None else value)
            annotation_row.append(format_delta(value))
        values.append(value_row)
        annotations.append(annotation_row)
    return names, labels, values, annotations


def column_normalized_values(values: list[list[float]]) -> list[list[float]]:
    if not values:
        return values

    column_count = len(values[0])
    normalized = [[float("nan") for _ in range(column_count)] for _ in values]
    for column in range(column_count):
        column_values = [
            row[column]
            for row in values
            if isinstance(row[column], float) and math.isfinite(row[column])
        ]
        if not column_values:
            continue
        minimum = min(column_values)
        maximum = max(column_values)
        span = maximum - minimum
        for row_index, row in enumerate(values):
            value = row[column]
            if not math.isfinite(value):
                continue
            normalized[row_index][column] = 0.5 if span == 0 else (value - minimum) / span
    return normalized


def column_scaled_delta_values(values: list[list[float]]) -> list[list[float]]:
    if not values:
        return values

    column_count = len(values[0])
    scaled = [[float("nan") for _ in range(column_count)] for _ in values]
    for column in range(column_count):
        column_values = [
            abs(row[column])
            for row in values
            if isinstance(row[column], float) and math.isfinite(row[column])
        ]
        maximum = max(column_values) if column_values else 0.0
        if maximum == 0:
            maximum = 1.0
        for row_index, row in enumerate(values):
            value = row[column]
            if math.isfinite(value):
                scaled[row_index][column] = value / maximum
    return scaled


def plot_latest_heatmap(summary: dict[str, Any], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output.parent.mkdir(parents=True, exist_ok=True)
    names, labels, values, annotations = heatmap_matrix(summary["opponents"])
    colors = column_normalized_values(values)

    height = max(4.0, 0.5 * max(1, len(names)) + 1.8)
    fig, axis = plt.subplots(figsize=(13, height))
    axis.set_title(
        f"Latest eval metrics: {Path(summary['run']['path']).name}",
        fontsize=14,
        fontweight="bold",
    )
    if names:
        sns.heatmap(
            colors,
            annot=annotations,
            fmt="",
            cmap="viridis",
            xticklabels=labels,
            yticklabels=names,
            linewidths=0.5,
            linecolor="white",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "column-normalized value"},
            ax=axis,
        )
    else:
        axis.text(0.5, 0.5, "No eval data", ha="center", va="center")
        axis.set_axis_off()
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return output


def plot_delta_heatmap(
    deltas: dict[str, dict[str, float]],
    run_name: str,
    baseline_name: str,
    output: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output.parent.mkdir(parents=True, exist_ok=True)
    names, labels, values, annotations = delta_heatmap_matrix(deltas)
    colors = column_scaled_delta_values(values)

    height = max(4.0, 0.5 * max(1, len(names)) + 1.8)
    fig, axis = plt.subplots(figsize=(13, height))
    axis.set_title(
        f"Eval delta: {run_name} - {baseline_name}",
        fontsize=14,
        fontweight="bold",
    )
    if names:
        sns.heatmap(
            colors,
            annot=annotations,
            fmt="",
            cmap="vlag",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=names,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "column-normalized delta"},
            ax=axis,
        )
    else:
        axis.text(0.5, 0.5, "No shared eval metrics", ha="center", va="center")
        axis.set_axis_off()
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    summary, warnings = summarize_run(args.run)

    payload: dict[str, Any] = dict(summary)
    deltas = None
    if args.baseline_run is not None:
        baseline_summary, baseline_warnings = summarize_run(args.baseline_run)
        warnings.extend(baseline_warnings)
        payload["baseline"] = baseline_summary
        deltas = compute_deltas(summary["opponents"], baseline_summary["opponents"])
        payload["deltas"] = deltas

    for warning in warnings:
        print(f"경고: {warning}", file=sys.stderr)

    stdout_summary = render_stdout(summary, deltas)
    print(stdout_summary)

    if args.write_report:
        write_text(args.json_output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        write_text(args.markdown_output, render_markdown(summary, payload))
        print(f"- json report: {args.json_output}")
        print(f"- markdown report: {args.markdown_output}")
    if not args.no_plot:
        plot_path = plot_run(args.run, output=args.plot_output, smooth_window=args.smooth_window)
        print(f"- plot: {plot_path}")
        heatmap_path = plot_latest_heatmap(summary, args.heatmap_output)
        print(f"- latest heatmap: {heatmap_path}")
        if deltas:
            delta_heatmap_path = plot_delta_heatmap(
                deltas,
                args.run.name,
                args.baseline_run.name if args.baseline_run is not None else "baseline",
                args.delta_heatmap_output,
            )
            print(f"- delta heatmap: {delta_heatmap_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
