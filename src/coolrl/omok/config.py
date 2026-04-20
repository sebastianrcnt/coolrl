from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RulesConfig:
    board_size: int = 9
    exactly_five: bool = False

    def __post_init__(self) -> None:
        if self.board_size != 9:
            raise ValueError("only 9x9 Omok is supported in coolrl.omok")


@dataclass(slots=True)
class NetworkConfig:
    input_planes: int = 4
    channels: int = 32
    blocks: int = 2
    value_hidden: int = 64
    se_reduction: int = 8


@dataclass(slots=True)
class SelfPlayConfig:
    mcts_backend: str = "python"
    evaluator_backend: str = "torch"
    games_per_iteration: int = 4
    batch_size: int = 2
    num_workers: int | str = 0
    search_threads: int | str = 1
    inference_batch_size: int = 256
    inference_wait_ms: float = 1.0
    temperature_moves: int = 6
    temperature_end: float = 0.0
    dirichlet_alpha: float = 0.25
    dirichlet_epsilon: float = 0.20
    c_puct: float = 1.6
    virtual_loss: float = 1.0
    leaves_per_batch: int = 1
    simulation_ramp_iterations: int | None = None
    candidate_mix_fraction: float = 0.0
    mixed_iterations_after_promotion: int = 0
    simulation_schedule: list[dict[str, float | int]] = field(
        default_factory=lambda: [{"fraction": 0.0, "simulations": 16}]
    )

    def resolved_num_workers(self) -> tuple[int, bool]:
        value = self.num_workers
        if isinstance(value, str):
            token = value.strip().lower()
            if token == "auto":
                return max(1, os.cpu_count() or 1), True
            try:
                return max(0, int(token)), False
            except ValueError as exc:
                raise ValueError(f"unsupported selfplay.num_workers: {value!r}") from exc
        return max(0, int(value)), False

    def resolved_search_threads(self) -> tuple[int, bool]:
        value = self.search_threads
        if isinstance(value, str):
            token = value.strip().lower()
            if token == "auto":
                return max(1, os.cpu_count() or 1), True
            try:
                return max(1, int(token)), False
            except ValueError as exc:
                raise ValueError(f"unsupported selfplay.search_threads: {value!r}") from exc
        return max(1, int(value)), False


@dataclass(slots=True)
class OptimizationConfig:
    batch_size: int = 32
    updates_per_iteration: int = 8
    learning_rate: float = 1.0e-3
    min_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 0.0
    replay_capacity: int = 4096
    warmup_games: int = 4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    value_discount: float = 1.0
    recency_temperature: float = 0.0


@dataclass(slots=True)
class ArenaConfig:
    games: int = 4
    simulations: int = 16
    accept_win_rate: float = 0.55
    min_white_win_rate: float = 0.0
    bootstrap_games: int = 2
    bootstrap_simulations: int = 16
    bootstrap_accept_win_rate: float = 0.0
    bootstrap_min_white_win_rate: float = 0.0


@dataclass(slots=True)
class CheckpointConfig:
    directory: str = "checkpoints/omok_default"
    save_every_iteration: bool = True
    save_iteration_interval: int = 1
    progress_interval_batches: int = 4
    progress_interval_seconds: float = 30.0


@dataclass(slots=True)
class RunConfig:
    experiment_name: str = "omok_9x9"
    seed: int = 17
    max_hours: float | None = None
    max_iterations: int | None = None
    device: str = "auto"
    use_amp: bool = False
    rules: RulesConfig = field(default_factory=RulesConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoint.directory)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _to_dataclass(cfg: dict[str, Any]) -> RunConfig:
    raw_max_hours = cfg.get("max_hours", None)
    raw_max_iterations = cfg.get("max_iterations", None)
    return RunConfig(
        experiment_name=str(cfg.get("experiment_name", "omok_9x9")),
        seed=int(cfg.get("seed", 17)),
        max_hours=None if raw_max_hours is None else float(raw_max_hours),
        max_iterations=None if raw_max_iterations is None else int(raw_max_iterations),
        device=str(cfg.get("device", "auto")),
        use_amp=bool(cfg.get("use_amp", False)),
        rules=RulesConfig(**cfg.get("rules", {})),
        network=NetworkConfig(**cfg.get("network", {})),
        selfplay=SelfPlayConfig(**cfg.get("selfplay", {})),
        optimization=OptimizationConfig(**cfg.get("optimization", {})),
        arena=ArenaConfig(**cfg.get("arena", {})),
        checkpoint=CheckpointConfig(**cfg.get("checkpoint", {})),
    )


def load_config(path: str | Path | None) -> RunConfig:
    default = RunConfig()
    if path is None:
        return default
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    merged = _merge_dict(default.to_dict(), raw)
    return _to_dataclass(merged)


def config_from_dict(payload: dict[str, Any]) -> RunConfig:
    return _to_dataclass(_merge_dict(RunConfig().to_dict(), payload))
