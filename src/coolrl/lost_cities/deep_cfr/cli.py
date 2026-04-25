"""Lost Cities Deep CFR command-line tools.

Examples:
  uv run python -m coolrl.lost_cities.deep_cfr.cli status --checkpoint-dir checkpoints/lost_cities_deep_cfr_overnight
  uv run python -m coolrl.lost_cities.deep_cfr.cli plot --checkpoint-dir checkpoints/lost_cities_deep_cfr_overnight
  uv run python -m coolrl.lost_cities.deep_cfr.cli eval --checkpoint checkpoints/lost_cities_deep_cfr_overnight/latest.pt --games 500 --opponent safe_heuristic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from loguru import logger

from .benchmark import benchmark_traversal_modes
from .config import config_from_dict, load_config
from .evaluate import evaluate_against_bot, make_opponent
from .networks import StrategyNet
from .trainer import DeepCFRTrainer, _torch_device
from .visualize import load_metrics, load_runtime_progress, plot_metrics, summarize_metrics


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")


def train_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    trainer = DeepCFRTrainer(config, resume_path=args.resume)
    trainer.run()


def eval_command(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else config_from_dict(checkpoint["config"])
    device = _torch_device(config.device)
    strategy_net = StrategyNet(
        int(checkpoint["input_dim"]),
        int(checkpoint["action_size"]),
        config.network,
    ).to(device)
    strategy_net.load_state_dict(checkpoint["strategy_net"])
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    opponent = make_opponent(args.opponent, seed=config.seed + 99)
    result = evaluate_against_bot(
        strategy_net,
        opponent,
        lc_config,
        args.games,
        config.seed + 123_000,
        device=device,
        max_steps=config.evaluation.max_steps,
        on_max_steps=config.evaluation.on_max_steps,
    )
    logger.info(
        "Evaluation vs {}: games={} win_rate={:.3f} avg_diff={:.2f} avg_final_score={:.2f} avg_opponent_score={:.2f} avg_opened_colors={:.2f} play_action_rate={:.3f} discard_action_rate={:.3f} wins={} losses={} draws={} max_step_timeouts={}",
        args.opponent,
        result["games"],
        result["win_rate"],
        result["avg_diff"],
        result["avg_final_score"],
        result["avg_opponent_score"],
        result["avg_opened_colors"],
        result["play_action_rate"],
        result["discard_action_rate"],
        result["wins"],
        result["losses"],
        result["draws"],
        result["max_step_timeouts"],
    )


def _log_traversal_result(label: str, r: dict) -> None:
    logger.info(
        "{}: requested_workers={} effective_workers={} batches={} traversal_seconds={:.4f} total_nodes={} traversals={} avg_nodes_per_traversal={:.1f} nodes_per_second={:.1f} cutoffs={} cutoff_rate={:.4f} node_limit_cutoffs={} node_limit_cutoff_rate={:.4f} cutoff_rollouts={} rollout_steps={} rollout_timeouts={}",
        label,
        r["requested_workers"],
        r["effective_workers"],
        r["num_batches"],
        r["traversal_seconds"],
        r["total_nodes"],
        r["traversals"],
        r["avg_nodes_per_traversal"],
        r["nodes_per_second"],
        r["cutoffs"],
        r["cutoff_rate"],
        r["node_limit_cutoffs"],
        r["node_limit_cutoff_rate"],
        r["cutoff_rollouts"],
        r["cutoff_rollout_steps"],
        r["cutoff_rollout_max_step_timeouts"],
    )


def benchmark_traversal_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = benchmark_traversal_modes(
        config,
        mp_workers=args.mp_workers,
        iteration=args.iteration,
        mode=args.mode,
    )
    logger.info(
        "Traversal benchmark uses device={} for both modes so the comparison reflects traversal multiprocessing overhead/speedup.",
        result["device_used"],
    )
    if "single_process" in result:
        _log_traversal_result("Single-process", result["single_process"])
    if "multiprocessing" in result:
        _log_traversal_result("Multiprocessing", result["multiprocessing"])
    speedup = result["speedup_vs_single_process"]
    if speedup == "n/a":
        logger.info("Traversal benchmark speedup vs single-process: n/a")
    else:
        logger.info("Traversal benchmark speedup vs single-process: {:.3f}x", speedup)


def status_command(args: argparse.Namespace) -> None:
    metrics = load_metrics(args.checkpoint_dir)
    runtime_progress = load_runtime_progress(args.checkpoint_dir)
    summary = summarize_metrics(metrics)
    latest = runtime_progress or summary

    logger.info("Checkpoint dir: {}", Path(args.checkpoint_dir))
    logger.info(
        "Iteration={} elapsed_seconds={} nodes_per_second={} avg_nodes_per_traversal={}",
        latest.get("iteration", "n/a"),
        latest.get("elapsed_seconds", "n/a"),
        latest.get("nodes_per_second", "n/a"),
        latest.get("avg_nodes_per_traversal", "n/a"),
    )
    if "cutoff_rate" in latest or "node_limit_cutoff_rate" in latest:
        logger.info(
            "Cutoffs: cutoff_rate={} node_limit_cutoff_rate={} cutoff_rollouts={} cutoff_rollout_steps={} avg_cutoff_rollout_steps={} cutoff_rollout_max_step_timeouts={}",
            latest.get("cutoff_rate", "n/a"),
            latest.get("node_limit_cutoff_rate", "n/a"),
            latest.get("cutoff_rollouts", "n/a"),
            latest.get("cutoff_rollout_steps", "n/a"),
            latest.get("avg_cutoff_rollout_steps", "n/a"),
            latest.get("cutoff_rollout_max_step_timeouts", "n/a"),
        )
    if "advantage_loss_p0" in latest or "advantage_loss_p1" in latest or "strategy_loss" in latest:
        logger.info(
            "Losses: advantage_p0={} advantage_p1={} strategy={}",
            latest.get("advantage_loss_p0", "n/a"),
            latest.get("advantage_loss_p1", "n/a"),
            latest.get("strategy_loss", "n/a"),
        )
    if "advantage_memory_size_p0" in latest or "advantage_memory_size_p1" in latest or "strategy_memory_size" in latest:
        logger.info(
            "Memory: advantage_p0={} advantage_p1={} strategy={}",
            latest.get("advantage_memory_size_p0", "n/a"),
            latest.get("advantage_memory_size_p1", "n/a"),
            latest.get("strategy_memory_size", "n/a"),
        )
    eval_opponents = sorted(
        {
            key.removeprefix("eval_").removesuffix("_win_rate")
            for key in latest
            if key.startswith("eval_") and key.endswith("_win_rate")
        }
    )
    for opponent_name in eval_opponents:
        prefix = f"eval_{opponent_name}"
        logger.info(
            "Eval {}: win_rate={} avg_diff={} avg_final_score={} avg_opponent_score={} avg_opened_colors={} avg_opponent_opened_colors={} avg_expedition_cards={} avg_play_actions={} avg_discard_actions={} play_action_rate={} discard_action_rate={} max_step_timeouts={}",
            opponent_name,
            latest.get(f"{prefix}_win_rate", "n/a"),
            latest.get(f"{prefix}_avg_diff", "n/a"),
            latest.get(f"{prefix}_avg_final_score", "n/a"),
            latest.get(f"{prefix}_avg_opponent_score", "n/a"),
            latest.get(f"{prefix}_avg_opened_colors", "n/a"),
            latest.get(f"{prefix}_avg_opponent_opened_colors", "n/a"),
            latest.get(f"{prefix}_avg_expedition_cards", "n/a"),
            latest.get(f"{prefix}_avg_play_actions", "n/a"),
            latest.get(f"{prefix}_avg_discard_actions", "n/a"),
            latest.get(f"{prefix}_play_action_rate", "n/a"),
            latest.get(f"{prefix}_discard_action_rate", "n/a"),
            latest.get(f"{prefix}_max_step_timeouts", "n/a"),
        )


def plot_command(args: argparse.Namespace) -> None:
    output_path = plot_metrics(args.checkpoint_dir, output=args.output)
    logger.info("Saved training metrics plot to {}", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m coolrl.lost_cities.deep_cfr.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, default=Path("configs/lost_cities_deep_cfr_tier3.yaml"))
    train.add_argument("--resume", type=str, default=None)
    train.set_defaults(func=train_command)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--config", type=Path, default=None)
    eval_parser.add_argument("--games", type=int, default=100)
    eval_parser.add_argument("--opponent", choices=["random", "safe_heuristic", "passive_discard"], default="random")
    eval_parser.set_defaults(func=eval_command)

    benchmark = subparsers.add_parser("benchmark-traversal")
    benchmark.add_argument("--config", type=Path, default=Path("configs/lost_cities_deep_cfr_probe.yaml"))
    benchmark.add_argument("--mp-workers", type=int, default=2)
    benchmark.add_argument("--iteration", type=int, default=1)
    benchmark.add_argument("--mode", choices=["compare", "single", "mp"], default="compare")
    benchmark.set_defaults(func=benchmark_traversal_command)

    status = subparsers.add_parser("status")
    status.add_argument("--checkpoint-dir", type=Path, required=True)
    status.set_defaults(func=status_command)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--checkpoint-dir", type=Path, required=True)
    plot.add_argument("--output", type=Path, default=None)
    plot.set_defaults(func=plot_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
