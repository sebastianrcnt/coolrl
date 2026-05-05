#!/usr/bin/env python3
"""Lost Cities zero-pit POC 실험 전용 metrics 리포트 스크립트."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


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
    parser.add_argument("--json-output", type=Path, default=None, help="JSON 리포트 저장 경로")
    parser.add_argument("--markdown-output", type=Path, default=None, help="Markdown 리포트 저장 경로")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="PNG plot 저장 경로. 기본값은 <run>/analysis_metrics.png",
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


def plot_run(run_dir: Path, output: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from coolrl.plotting import (
        DARK,
        configure_fonts,
        gradient_line,
        moving_average_smooth,
        panel_title,
        style_axis,
    )

    rows, _warnings = read_metrics(run_dir)
    output_path = output if output is not None else run_dir / "analysis_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    theme = DARK
    latest_eval_row = find_latest_eval_row(rows)
    opponents = discover_opponents(latest_eval_row)
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), facecolor=theme.bg)
    fig.suptitle(
        f"Lost Cities zero-pit metrics: {run_dir.name}",
        color=theme.text_primary,
        fontsize=22,
        fontweight=900,
        y=0.985,
    )

    panels = [
        (
            "Training Loss",
            [("advantage p0", "advantage_loss_p0"), ("advantage p1", "advantage_loss_p1"), ("strategy", "strategy_loss")],
            False,
        ),
        ("Win Rate", [(opponent, metric_key(opponent, "win_rate")) for opponent in opponents], True),
        ("Avg Diff", [(opponent, metric_key(opponent, "avg_diff")) for opponent in opponents], False),
        (
            "Action Rates",
            [
                (f"{opponent} play", metric_key(opponent, "play_action_rate"))
                for opponent in opponents
            ]
            + [
                (f"{opponent} discard", metric_key(opponent, "discard_action_rate"))
                for opponent in opponents
            ],
            True,
        ),
        (
            "Expedition Shape",
            [
                (f"{opponent} colors", metric_key(opponent, "avg_opened_colors"))
                for opponent in opponents
            ]
            + [
                (f"{opponent} cards", metric_key(opponent, "avg_expedition_cards"))
                for opponent in opponents
            ],
            False,
        ),
        (
            "Timeouts / Entropy",
            [
                (f"{opponent} timeout", metric_key(opponent, "max_step_timeouts"))
                for opponent in opponents
            ]
            + [
                (f"{opponent} entropy", metric_key(opponent, "policy_entropy"))
                for opponent in opponents
            ],
            False,
        ),
    ]
    accent_pairs = list(theme.accents.values())

    for axis, (title, specs, percent) in zip(axes.flat, panels, strict=True):
        axis.set_facecolor(theme.bg)
        panel_title(axis, theme, title)
        style_axis(axis, theme, percent=percent)
        plotted = False
        for index, (label, key) in enumerate(specs):
            xs, ys = metric_series(rows, key)
            if not xs:
                continue
            y_values = [value * 100 for value in ys] if percent else ys
            y_arr = moving_average_smooth(y_values, window=5) if len(y_values) >= 5 else y_values
            c0, c1 = accent_pairs[index % len(accent_pairs)]
            gradient_line(axis, xs, y_arr, c0, c1, lw=2.0, alpha=theme.line_alpha)
            axis.plot([], [], color=c1, label=label)
            plotted = True
        if plotted:
            axis.legend(frameon=False, labelcolor=theme.text_secondary, fontsize=8)
            axis.autoscale_view()
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", color=theme.text_tertiary, transform=axis.transAxes)
        axis.set_xlabel("iteration", color=theme.text_secondary)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=170, facecolor=theme.bg, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return output_path


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

    if args.json_output is not None:
        write_text(args.json_output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.markdown_output is not None:
        write_text(args.markdown_output, render_markdown(summary, payload))
    if not args.no_plot:
        plot_path = plot_run(args.run, output=args.plot_output)
        print(f"- plot: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
