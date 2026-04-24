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
    )
    logger.info(
        "Evaluation vs {}: games={} win_rate={:.3f} avg_diff={:.2f} wins={} losses={} draws={}",
        args.opponent,
        result["games"],
        result["win_rate"],
        result["avg_diff"],
        result["wins"],
        result["losses"],
        result["draws"],
    )


def benchmark_traversal_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = benchmark_traversal_modes(
        config,
        mp_workers=args.mp_workers,
        iteration=args.iteration,
    )
    single = result["single_process"]
    multiprocessing = result["multiprocessing"]
    logger.info(
        "Traversal benchmark uses device={} for both modes so the comparison reflects traversal multiprocessing overhead/speedup.",
        result["device_used"],
    )
    logger.info(
        "Single-process: workers={} traversal_seconds={:.4f} total_nodes={} traversals={} avg_nodes_per_traversal={:.1f} nodes_per_second={:.1f}",
        single["num_workers"],
        single["traversal_seconds"],
        single["total_nodes"],
        single["traversals"],
        single["avg_nodes_per_traversal"],
        single["nodes_per_second"],
    )
    logger.info(
        "Multiprocessing: workers={} traversal_seconds={:.4f} total_nodes={} traversals={} avg_nodes_per_traversal={:.1f} nodes_per_second={:.1f}",
        multiprocessing["num_workers"],
        multiprocessing["traversal_seconds"],
        multiprocessing["total_nodes"],
        multiprocessing["traversals"],
        multiprocessing["avg_nodes_per_traversal"],
        multiprocessing["nodes_per_second"],
    )
    logger.info(
        "Traversal benchmark speedup vs single-process: {:.3f}x",
        result["speedup_vs_single_process"],
    )


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
    eval_parser.add_argument("--opponent", choices=["random", "safe_heuristic"], default="random")
    eval_parser.set_defaults(func=eval_command)

    benchmark = subparsers.add_parser("benchmark-traversal")
    benchmark.add_argument("--config", type=Path, default=Path("configs/lost_cities_deep_cfr_probe.yaml"))
    benchmark.add_argument("--mp-workers", type=int, default=2)
    benchmark.add_argument("--iteration", type=int, default=1)
    benchmark.set_defaults(func=benchmark_traversal_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
