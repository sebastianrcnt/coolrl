from __future__ import annotations

import os
import sys
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
class EncodingConfig:
    derived_playability: bool = False


@dataclass(slots=True)
class SelfPlayLeagueConfig:
    current_weight: float = 0.5
    recent_weight: float = 0.3
    older_weight: float = 0.2
    anchor_weight: float = 0.0
    anchor_policy: str = "safe_heuristic"
    recent_window: int = 5
    max_snapshots: int = 20
    snapshot_every: int = 1

    def __post_init__(self) -> None:
        self.current_weight = float(self.current_weight)
        self.recent_weight = float(self.recent_weight)
        self.older_weight = float(self.older_weight)
        self.anchor_weight = float(self.anchor_weight)
        weights = (self.current_weight, self.recent_weight, self.older_weight, self.anchor_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("traversal.self_play_league weights must be nonnegative")
        if sum(weights) <= 0.0:
            raise ValueError("traversal.self_play_league weights must sum to a positive value")
        self.anchor_policy = str(self.anchor_policy).strip().lower()
        if self.anchor_policy not in {"safe_heuristic"}:
            raise ValueError("traversal.self_play_league.anchor_policy must be 'safe_heuristic'")
        self.recent_window = int(self.recent_window)
        if self.recent_window < 0:
            raise ValueError("traversal.self_play_league.recent_window must be nonnegative")
        self.max_snapshots = int(self.max_snapshots)
        if self.max_snapshots < 0:
            raise ValueError("traversal.self_play_league.max_snapshots must be nonnegative")
        self.snapshot_every = int(self.snapshot_every)
        if self.snapshot_every <= 0:
            raise ValueError("traversal.self_play_league.snapshot_every must be positive")


@dataclass(slots=True)
class TraversalConfig:
    backend: str = "python"
    traversals_per_player: int = 100
    strategy_sample_interval: int = 1
    store_strategy_on_opponent_nodes: bool = True
    store_strategy_on_traverser_nodes: bool = True
    max_depth: int | None = 8
    max_nodes_per_traversal: int | None = 10_000
    cutoff_value_mode: str = "score_diff"
    cutoff_rollouts: int = 0
    cutoff_rollout_policy: str = "random"
    cutoff_rollout_max_steps: int = 10_000
    opponent_policy: str = "network"
    progress_every_traversals: int = 10
    num_workers: int | str = 0
    traversal_worker_chunk_size: int = 4
    profile_hotspots: bool = False
    regret_matching_epsilon: float = 1.0e-8
    outcome_sampling_epsilon: float = 0.0
    outcome_sampling_value_clip: float | None = None
    outcome_unsampled_regret: str = "negative_node_value"
    endpoint_depth_bucket_width: int = 100
    endpoint_depth_bucket_max: int = 1000
    self_play_league: SelfPlayLeagueConfig = field(default_factory=SelfPlayLeagueConfig)

    def __post_init__(self) -> None:
        if isinstance(self.self_play_league, dict):
            self.self_play_league = SelfPlayLeagueConfig(**self.self_play_league)
        mode = str(self.cutoff_value_mode).strip().lower()
        if mode not in {"score_diff", "random_rollout"}:
            raise ValueError("traversal.cutoff_value_mode must be one of 'score_diff' or 'random_rollout'")
        self.cutoff_value_mode = mode
        policy = str(self.cutoff_rollout_policy).strip().lower()
        if policy not in {"random", "safe_heuristic"}:
            raise ValueError(
                "traversal.cutoff_rollout_policy must be one of 'random' or 'safe_heuristic'"
            )
        self.cutoff_rollout_policy = policy
        opponent_policy = str(self.opponent_policy).strip().lower()
        if opponent_policy not in {"network", "safe_heuristic", "self_play_league"}:
            raise ValueError(
                "traversal.opponent_policy must be one of "
                "'network', 'safe_heuristic', or 'self_play_league'"
            )
        self.opponent_policy = opponent_policy
        self.cutoff_rollouts = int(self.cutoff_rollouts)
        if self.cutoff_rollouts < 0:
            raise ValueError("traversal.cutoff_rollouts must be nonnegative")
        self.cutoff_rollout_max_steps = int(self.cutoff_rollout_max_steps)
        if self.cutoff_rollout_max_steps <= 0:
            raise ValueError("traversal.cutoff_rollout_max_steps must be positive")
        self.outcome_sampling_epsilon = float(self.outcome_sampling_epsilon)
        if not 0.0 <= self.outcome_sampling_epsilon <= 1.0:
            raise ValueError("traversal.outcome_sampling_epsilon must be between 0 and 1")
        if self.outcome_sampling_value_clip is not None:
            self.outcome_sampling_value_clip = float(self.outcome_sampling_value_clip)
            if self.outcome_sampling_value_clip <= 0.0:
                raise ValueError("traversal.outcome_sampling_value_clip must be positive when set")
        unsampled_regret = str(self.outcome_unsampled_regret).strip().lower()
        if unsampled_regret not in {"negative_node_value", "zero"}:
            raise ValueError(
                "traversal.outcome_unsampled_regret must be one of "
                "'negative_node_value' or 'zero'"
            )
        self.outcome_unsampled_regret = unsampled_regret
        self.endpoint_depth_bucket_width = int(self.endpoint_depth_bucket_width)
        if self.endpoint_depth_bucket_width <= 0:
            raise ValueError("traversal.endpoint_depth_bucket_width must be positive")
        self.endpoint_depth_bucket_max = int(self.endpoint_depth_bucket_max)
        if self.endpoint_depth_bucket_max <= 0:
            raise ValueError("traversal.endpoint_depth_bucket_max must be positive")

    def _cpu_worker_guess(self) -> int:
        logical = max(1, os.cpu_count() or 1)
        physical = None
        try:
            psutil = sys.modules.get("psutil")
            if psutil is None:
                import psutil as imported_psutil

                psutil = imported_psutil
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
        return max(1, int(physical) if physical else logical // 2)

    def estimated_num_batches(self) -> int:
        traversals_per_player = max(0, int(self.traversals_per_player))
        if traversals_per_player <= 0:
            return 0
        chunk_size = max(1, int(self.traversal_worker_chunk_size))
        chunks_per_player = (traversals_per_player + chunk_size - 1) // chunk_size
        return 2 * chunks_per_player

    def resolved_num_workers(self) -> tuple[int, bool]:
        value = self.num_workers
        if isinstance(value, str):
            token = value.strip().lower()
            if token == "auto":
                return self._cpu_worker_guess(), True
            try:
                return max(0, int(token)), False
            except ValueError as exc:
                raise ValueError(f"unsupported traversal.num_workers: {value!r}") from exc
        return max(0, int(value)), False

    def resolved_num_workers_for_traversal(self) -> tuple[int, bool, int | None, int | None]:
        requested_workers, is_auto = self.resolved_num_workers()
        if not is_auto:
            return requested_workers, False, None, None
        cpu_guess = requested_workers
        num_batches = self.estimated_num_batches()
        resolved = max(1, min(cpu_guess, num_batches)) if num_batches > 0 else 1
        return resolved, True, cpu_guess, num_batches


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
    opponents: list[str] = field(default_factory=lambda: ["random", "safe_heuristic", "passive_discard"])
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
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
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
        encoding=EncodingConfig(**cfg.get("encoding", {})),
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
