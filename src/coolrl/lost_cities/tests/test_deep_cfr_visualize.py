from __future__ import annotations

import json
from pathlib import Path

import pytest

from coolrl.lost_cities.deep_cfr.visualize import (
    load_metrics,
    load_runtime_progress,
    plot_metrics,
    summarize_metrics,
)


def test_load_metrics_reads_and_sorts_jsonl(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"iteration": 2, "strategy_loss": 0.4}),
                "",
                json.dumps({"iteration": 1, "strategy_loss": 0.6}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = load_metrics(tmp_path)

    assert [item["iteration"] for item in metrics] == [1, 2]


def test_summarize_metrics_returns_latest_values() -> None:
    summary = summarize_metrics(
        [
            {"iteration": 1, "strategy_loss": 1.2, "nodes_per_second": 10.0},
            {
                "iteration": 2,
                "elapsed_seconds": 5.0,
                "total_nodes": 123,
                "nodes_per_second": 20.0,
                "avg_nodes_per_traversal": 4.0,
                "cutoff_rate": 0.1,
                "node_limit_cutoff_rate": 0.2,
                "eval_random_win_rate": 0.7,
                "eval_random_avg_diff": 3.0,
                "eval_random_avg_final_score": 12.0,
                "eval_random_avg_opened_colors": 2.0,
                "eval_random_play_action_rate": 0.4,
                "eval_safe_heuristic_win_rate": 0.6,
                "eval_safe_heuristic_avg_diff": 2.0,
                "eval_safe_heuristic_avg_final_score": 8.0,
                "eval_safe_heuristic_avg_opened_colors": 1.0,
                "eval_safe_heuristic_play_action_rate": 0.3,
                "advantage_loss_p0": 0.4,
                "advantage_loss_p1": 0.5,
                "strategy_loss": 0.3,
            },
        ]
    )

    assert summary["iteration"] == 2
    assert summary["elapsed_seconds"] == 5.0
    assert summary["total_nodes"] == 123
    assert summary["nodes_per_second"] == 20.0
    assert summary["avg_nodes_per_traversal"] == 4.0
    assert summary["cutoff_rate"] == 0.1
    assert summary["node_limit_cutoff_rate"] == 0.2
    assert summary["eval_random_win_rate"] == 0.7
    assert summary["eval_random_avg_final_score"] == 12.0
    assert summary["eval_random_avg_opened_colors"] == 2.0
    assert summary["eval_random_play_action_rate"] == 0.4
    assert summary["eval_safe_heuristic_win_rate"] == 0.6
    assert summary["eval_safe_heuristic_avg_final_score"] == 8.0
    assert summary["eval_safe_heuristic_avg_opened_colors"] == 1.0
    assert summary["eval_safe_heuristic_play_action_rate"] == 0.3
    assert summary["advantage_loss_p0"] == 0.4
    assert summary["advantage_loss_p1"] == 0.5
    assert summary["strategy_loss"] == 0.3


def test_plot_metrics_writes_png(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "iteration": 1,
                        "advantage_loss_p0": 0.7,
                        "advantage_loss_p1": 0.6,
                        "strategy_loss": 0.5,
                        "eval_random_win_rate": 0.4,
                        "eval_safe_heuristic_win_rate": 0.3,
                        "eval_random_avg_diff": -1.0,
                        "eval_safe_heuristic_avg_diff": -2.0,
                        "nodes_per_second": 10.0,
                        "avg_nodes_per_traversal": 4.0,
                        "cutoff_rate": 0.1,
                        "node_limit_cutoff_rate": 0.05,
                        "advantage_memory_size_p0": 11,
                        "advantage_memory_size_p1": 12,
                        "strategy_memory_size": 13,
                    }
                ),
                json.dumps(
                    {
                        "iteration": 2,
                        "advantage_loss_p0": 0.4,
                        "advantage_loss_p1": 0.3,
                        "strategy_loss": 0.2,
                        "eval_random_win_rate": 0.6,
                        "eval_safe_heuristic_win_rate": 0.5,
                        "eval_random_avg_diff": 1.5,
                        "eval_safe_heuristic_avg_diff": 0.5,
                        "nodes_per_second": 20.0,
                        "avg_nodes_per_traversal": 5.0,
                        "cutoff_rate": 0.08,
                        "node_limit_cutoff_rate": 0.02,
                        "advantage_memory_size_p0": 21,
                        "advantage_memory_size_p1": 22,
                        "strategy_memory_size": 23,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "runtime_progress.json").write_text(json.dumps({"iteration": 2}), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="legacy compatibility"):
        output_path = plot_metrics(tmp_path)

    assert load_runtime_progress(tmp_path) == {"iteration": 2}
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0
