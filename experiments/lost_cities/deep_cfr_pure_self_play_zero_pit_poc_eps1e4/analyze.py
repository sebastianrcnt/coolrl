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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
