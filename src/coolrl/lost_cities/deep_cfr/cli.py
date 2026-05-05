"""Lost Cities Deep CFR command-line tools.

Examples:
  uv run python -m coolrl.lost_cities.deep_cfr.cli train --config configs/lost_cities_deep_cfr_tier3.yaml
  uv run python -m coolrl.lost_cities.deep_cfr.cli status --checkpoint-dir checkpoints/lost_cities_deep_cfr_tier3
  uv run python -m coolrl.lost_cities.deep_cfr.cli eval --checkpoint checkpoints/lost_cities_deep_cfr_tier3/latest.pt --games 500 --opponent safe_heuristic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import time

import torch
from loguru import logger

from .benchmark import benchmark_traversal_modes
from .config import RunConfig, config_from_dict, load_config
from .evaluate import SUPPORTED_OPPONENTS, evaluate_against_bot, make_opponent
from .imitation import pretrain_safe_heuristic_checkpoint
from .networks import StrategyNet
from .policy_gradient import fine_tune_policy_gradient_checkpoint
from .trainer import DeepCFRTrainer, _torch_device
from .visualize import load_metrics, load_runtime_progress, plot_metrics, summarize_metrics

_RESUME_LATEST = "__latest__"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")


def _configure_train_file_logging(config: RunConfig, *, resume_path: str | None) -> Path:
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "train.log"
    if resume_path is None and log_path.exists():
        archive = log_path.with_name(f"train.{time.strftime('%Y%m%d-%H%M%S')}.log")
        shutil.move(log_path, archive)
    logger.add(log_path, level="INFO", encoding="utf-8")
    return log_path


def _resolve_resume_path(config: RunConfig, resume: str | None) -> str | None:
    if resume != _RESUME_LATEST:
        return resume
    latest_path = config.checkpoint_dir / "latest.pt"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"--resume was used without a path, but latest checkpoint does not exist: {latest_path}"
        )
    return str(latest_path)


def train_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    resume_path = _resolve_resume_path(config, args.resume)
    log_path = _configure_train_file_logging(config, resume_path=resume_path)
    logger.info("Training log file: {}", log_path)
    trainer = DeepCFRTrainer(config, resume_path=resume_path)
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
        max_steps=args.max_steps if args.max_steps is not None else config.evaluation.max_steps,
        on_max_steps=config.evaluation.on_max_steps,
        sample=args.sample,
    )
    logger.info(
        "Evaluation vs {}: games={} sample={} max_steps={} win_rate={:.3f} avg_diff={:.2f} avg_final_score={:.2f} avg_opponent_score={:.2f} avg_opened_colors={:.2f} play_action_rate={:.3f} discard_action_rate={:.3f} wins={} losses={} draws={} max_step_timeouts={}",
        args.opponent,
        result["games"],
        args.sample,
        args.max_steps if args.max_steps is not None else config.evaluation.max_steps,
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


def pretrain_heuristic_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = args.output or (config.checkpoint_dir / "safe_heuristic_pretrain.pt")
    result = pretrain_safe_heuristic_checkpoint(
        config,
        output_path=output,
        games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
        dataset_mode=args.dataset_mode,
        base_checkpoint=args.base_checkpoint,
        init_checkpoint=args.init_checkpoint,
        policy_sample=args.policy_sample,
        improvement_rollouts=args.improvement_rollouts,
        improvement_rollout_max_steps=args.improvement_rollout_max_steps,
        improvement_max_examples=args.improvement_max_examples,
        improvement_top_k=args.improvement_top_k,
        improvement_progress_every=args.improvement_progress_every,
        learning_rate=args.learning_rate,
    )
    logger.info(
        "Safe heuristic pretrain complete: output={} games={} states={} strategy_loss={:.4f} strategy_accuracy={:.4f} advantage_loss=({:.4f},{:.4f})",
        result.output_path,
        result.games,
        result.states,
        result.strategy_loss,
        result.strategy_accuracy,
        result.advantage_loss_p0,
        result.advantage_loss_p1,
    )


def fine_tune_policy_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = fine_tune_policy_gradient_checkpoint(
        config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        games=args.games,
        opponent=args.opponent,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        batch_games=args.batch_games,
        normalize_advantages=args.normalize_advantages,
        baseline_decay=args.baseline_decay,
        seed=args.seed,
    )
    logger.info(
        "Policy-gradient fine-tune complete: output={} games={} win_rate={:.3f} avg_diff={:.2f} avg_loss={:.4f} avg_kl={:.4f} avg_entropy={:.4f}",
        result.output_path,
        result.games,
        result.wins / max(1, result.games),
        result.avg_diff,
        result.avg_loss,
        result.avg_kl,
        result.avg_entropy,
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
    train.add_argument("--resume", nargs="?", const=_RESUME_LATEST, default=None)
    train.set_defaults(func=train_command)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--config", type=Path, default=None)
    eval_parser.add_argument("--games", type=int, default=100)
    eval_parser.add_argument("--opponent", choices=SUPPORTED_OPPONENTS, default="random")
    eval_parser.add_argument("--sample", action="store_true", help="Sample from the strategy policy instead of using argmax.")
    eval_parser.add_argument("--max-steps", type=int, default=None, help="Override evaluation.max_steps from the config/checkpoint.")
    eval_parser.set_defaults(func=eval_command)

    pretrain = subparsers.add_parser("pretrain-heuristic")
    pretrain.add_argument("--config", type=Path, default=Path("configs/lost_cities_deep_cfr_tier3.yaml"))
    pretrain.add_argument("--output", type=Path, default=None)
    pretrain.add_argument("--games", type=int, default=1000)
    pretrain.add_argument("--epochs", type=int, default=8)
    pretrain.add_argument("--batch-size", type=int, default=2048)
    pretrain.add_argument("--max-steps", type=int, default=1000)
    pretrain.add_argument("--seed", type=int, default=None)
    pretrain.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override optimization.learning_rate for this pretrain run.",
    )
    pretrain.add_argument(
        "--dataset-mode",
        choices=["safe_self_play", "aggregated", "successful_policy_vs_safe", "safe_action_rollout"],
        default="safe_self_play",
        help=(
            "Use safe self-play states, aggregate model-induced states from "
            "--base-checkpoint, replay successful policy-vs-safe actions, "
            "or label actions by short safe-rollout improvement."
        ),
    )
    pretrain.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used as the behavior policy for aggregated imitation collection.",
    )
    pretrain.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize the pretrain networks before imitation updates.",
    )
    pretrain.add_argument(
        "--policy-sample",
        action="store_true",
        help="Sample the behavior policy when collecting aggregated imitation states.",
    )
    pretrain.add_argument(
        "--improvement-rollouts",
        type=int,
        default=1,
        help="Rollouts per legal action for safe_action_rollout dataset labeling.",
    )
    pretrain.add_argument(
        "--improvement-rollout-max-steps",
        type=int,
        default=300,
        help="Maximum rollout steps after each candidate action for safe_action_rollout labels.",
    )
    pretrain.add_argument(
        "--improvement-max-examples",
        type=int,
        default=None,
        help="Optional cap on safe_action_rollout labeled states.",
    )
    pretrain.add_argument(
        "--improvement-top-k",
        type=int,
        default=None,
        help="Limit safe_action_rollout candidate actions to policy top-k plus the safe action.",
    )
    pretrain.add_argument(
        "--improvement-progress-every",
        type=int,
        default=100,
        help="Log safe_action_rollout collection progress every N labeled states; 0 disables progress logs.",
    )
    pretrain.set_defaults(func=pretrain_heuristic_command)

    fine_tune = subparsers.add_parser("fine-tune-policy")
    fine_tune.add_argument("--config", type=Path, default=Path("configs/lost_cities_deep_cfr_safe_dagger_256.yaml"))
    fine_tune.add_argument("--checkpoint", type=Path, required=True)
    fine_tune.add_argument("--output", type=Path, required=True)
    fine_tune.add_argument("--games", type=int, default=1000)
    fine_tune.add_argument("--opponent", choices=SUPPORTED_OPPONENTS, default="safe_heuristic")
    fine_tune.add_argument("--max-steps", type=int, default=1000)
    fine_tune.add_argument("--learning-rate", type=float, default=1.0e-6)
    fine_tune.add_argument("--reward-scale", type=float, default=100.0)
    fine_tune.add_argument("--reward-clip", type=float, default=2.0)
    fine_tune.add_argument("--kl-coef", type=float, default=0.05)
    fine_tune.add_argument("--entropy-coef", type=float, default=0.001)
    fine_tune.add_argument("--grad-clip", type=float, default=0.5)
    fine_tune.add_argument("--batch-games", type=int, default=1)
    fine_tune.add_argument("--normalize-advantages", action="store_true")
    fine_tune.add_argument("--baseline-decay", type=float, default=None)
    fine_tune.add_argument("--seed", type=int, default=None)
    fine_tune.set_defaults(func=fine_tune_policy_command)

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
