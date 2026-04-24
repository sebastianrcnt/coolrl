from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ..game import LostCitiesConfig, tier_config


@dataclass(slots=True)
class RulesConfig:
    tier: str = "tier3"
    n_colors: int | None = None
    n_ranks: int | None = None
    min_rank: int | None = None
    n_handshakes: int | None = None
    hand_size: int | None = None
    expedition_penalty: int | None = None
    bonus_threshold: int | None = None
    bonus_amount: int | None = None

    def to_lost_cities_config(self, seed: int | None = None) -> LostCitiesConfig:
        values = tier_config(self.tier).to_snapshot()
        for field_info in fields(LostCitiesConfig):
            if field_info.name == "seed":
                continue
            if hasattr(self, field_info.name):
                value = getattr(self, field_info.name)
                if value is not None:
                    values[field_info.name] = value
        if seed is not None:
            values["seed"] = seed
        config = LostCitiesConfig(**values)
        config.validate()
        return config


@dataclass(slots=True)
class NetworkConfig:
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = "relu"


@dataclass(slots=True)
class TraversalConfig:
    backend: str = "python"
    traversals_per_player: int = 100
    strategy_sample_interval: int = 1
    store_strategy_on_opponent_nodes: bool = True
    store_strategy_on_traverser_nodes: bool = True
    max_depth: int | None = 8
    max_nodes_per_traversal: int | None = 10_000
    progress_every_traversals: int = 10
    num_workers: int | str = 0
    traversal_worker_chunk_size: int = 4
    profile_hotspots: bool = False
    regret_matching_epsilon: float = 1.0e-8

    def resolved_num_workers(self) -> tuple[int, bool]:
        value = self.num_workers
        if isinstance(value, str):
            token = value.strip().lower()
            if token == "auto":
                return max(1, os.cpu_count() or 1), True
            try:
                return max(0, int(token)), False
            except ValueError as exc:
                raise ValueError(f"unsupported traversal.num_workers: {value!r}") from exc
        return max(0, int(value)), False


@dataclass(slots=True)
class OptimizationConfig:
    advantage_batch_size: int = 1024
    strategy_batch_size: int = 1024
    advantage_updates_per_iteration: int = 256
    strategy_updates_per_iteration: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0


@dataclass(slots=True)
class MemoryConfig:
    advantage_capacity: int = 2_000_000
    strategy_capacity: int = 2_000_000


@dataclass(slots=True)
class EvaluationConfig:
    eval_every: int = 5
    games: int = 100
    opponents: list[str] = field(default_factory=lambda: ["random", "safe_heuristic"])
    max_steps: int = 10_000
    on_max_steps: str = "score_diff"

    def __post_init__(self) -> None:
        self.max_steps = int(self.max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"evaluation.max_steps must be positive, got {self.max_steps}")
        token = str(self.on_max_steps).strip().lower()
        if token not in {"score_diff", "loss", "draw"}:
            raise ValueError(
                "evaluation.on_max_steps must be one of 'score_diff', 'loss', or 'draw'"
            )
        self.on_max_steps = token


@dataclass(slots=True)
class CheckpointConfig:
    directory: str = "checkpoints/lost_cities_deep_cfr_tier3"
    save_every_iteration: bool = True
    save_iteration_interval: int = 1
    save_latest_only: bool = False
    progress_interval_seconds: float = 20.0


@dataclass(slots=True)
class RunConfig:
    experiment_name: str = "lost_cities_deep_cfr_tier3"
    seed: int = 29
    max_hours: float | None = None
    max_iterations: int | None = 100
    device: str = "auto"
    use_amp: bool = False
    rules: RulesConfig = field(default_factory=RulesConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    traversal: TraversalConfig = field(default_factory=TraversalConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
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
        experiment_name=str(cfg.get("experiment_name", "lost_cities_deep_cfr_tier3")),
        seed=int(cfg.get("seed", 29)),
        max_hours=None if raw_max_hours is None else float(raw_max_hours),
        max_iterations=None if raw_max_iterations is None else int(raw_max_iterations),
        device=str(cfg.get("device", "auto")),
        use_amp=bool(cfg.get("use_amp", False)),
        rules=RulesConfig(**cfg.get("rules", {})),
        network=NetworkConfig(**cfg.get("network", {})),
        traversal=TraversalConfig(**cfg.get("traversal", {})),
        optimization=OptimizationConfig(**cfg.get("optimization", {})),
        memory=MemoryConfig(**cfg.get("memory", {})),
        evaluation=EvaluationConfig(**cfg.get("evaluation", {})),
        checkpoint=CheckpointConfig(**cfg.get("checkpoint", {})),
    )


def load_config(path: str | Path | None) -> RunConfig:
    default = RunConfig()
    if path is None:
        return default
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping in config file: {path}")
    return _to_dataclass(_merge_dict(default.to_dict(), raw))


def config_from_dict(payload: dict[str, Any]) -> RunConfig:
    return _to_dataclass(_merge_dict(RunConfig().to_dict(), payload))
