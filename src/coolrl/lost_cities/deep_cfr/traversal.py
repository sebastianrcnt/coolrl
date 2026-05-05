from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch

from ..game import GameState
from ..bots.heuristic import SafeHeuristicBot
from .config import SelfPlayLeagueConfig
from .encoding import encode_information_state, legal_mask_array
from .memory import AdvantageMemory, StrategyMemory
from .networks import AdvantageNet, regret_matching

DEFAULT_ENDPOINT_DEPTH_BUCKET_WIDTH = 100
DEFAULT_ENDPOINT_DEPTH_BUCKET_MAX = 1000


def endpoint_depth_bucket_keys(width: int, max_depth: int) -> tuple[str, ...]:
    return tuple(f"{start}_{start + width - 1}" for start in range(0, max_depth, width)) + (
        f"{max_depth}_plus",
    )


def endpoint_depth_bucket_key(depth: int, width: int, max_depth: int) -> str:
    if depth >= max_depth:
        return f"{max_depth}_plus"
    start = max(0, depth // width * width)
    end = start + width - 1
    return f"{start}_{end}"


@dataclass(slots=True)
class TraversalStats:
    nodes: int = 0
    terminals: int = 0
    cutoffs: int = 0
    node_limit_cutoffs: int = 0
    max_depth_reached: int = 0
    endpoint_traversals: int = 0
    endpoint_depth_sum: int = 0
    terminal_depth_sum: int = 0
    cutoff_depth_sum: int = 0
    node_limit_cutoff_depth_sum: int = 0
    endpoint_depth_bucket_width: int = DEFAULT_ENDPOINT_DEPTH_BUCKET_WIDTH
    endpoint_depth_bucket_max: int = DEFAULT_ENDPOINT_DEPTH_BUCKET_MAX
    endpoint_depth_buckets: dict[str, int] = field(default_factory=dict)
    league_current_traversals: int = 0
    league_recent_traversals: int = 0
    league_older_traversals: int = 0
    league_anchor_traversals: int = 0
    cutoff_rollouts: int = 0
    cutoff_rollout_steps: int = 0
    cutoff_rollout_max_step_timeouts: int = 0

    def __post_init__(self) -> None:
        self.endpoint_depth_bucket_width = int(self.endpoint_depth_bucket_width)
        self.endpoint_depth_bucket_max = int(self.endpoint_depth_bucket_max)
        if self.endpoint_depth_bucket_width <= 0:
            raise ValueError("endpoint_depth_bucket_width must be positive")
        if self.endpoint_depth_bucket_max <= 0:
            raise ValueError("endpoint_depth_bucket_max must be positive")
        if not self.endpoint_depth_buckets:
            self.endpoint_depth_buckets = dict.fromkeys(
                endpoint_depth_bucket_keys(
                    self.endpoint_depth_bucket_width,
                    self.endpoint_depth_bucket_max,
                ),
                0,
            )

    def accumulate(self, other: TraversalStats) -> None:
        self.nodes += other.nodes
        self.terminals += other.terminals
        self.cutoffs += other.cutoffs
        self.node_limit_cutoffs += other.node_limit_cutoffs
        self.max_depth_reached = max(self.max_depth_reached, other.max_depth_reached)
        self.endpoint_traversals += other.endpoint_traversals
        self.endpoint_depth_sum += other.endpoint_depth_sum
        self.terminal_depth_sum += other.terminal_depth_sum
        self.cutoff_depth_sum += other.cutoff_depth_sum
        self.node_limit_cutoff_depth_sum += other.node_limit_cutoff_depth_sum
        for key, value in other.endpoint_depth_buckets.items():
            self.endpoint_depth_buckets[key] = self.endpoint_depth_buckets.get(key, 0) + value
        self.league_current_traversals += other.league_current_traversals
        self.league_recent_traversals += other.league_recent_traversals
        self.league_older_traversals += other.league_older_traversals
        self.league_anchor_traversals += other.league_anchor_traversals
        self.cutoff_rollouts += other.cutoff_rollouts
        self.cutoff_rollout_steps += other.cutoff_rollout_steps
        self.cutoff_rollout_max_step_timeouts += other.cutoff_rollout_max_step_timeouts


@dataclass(slots=True)
class TraversalTimingStats:
    traversal_wall_seconds: float = 0.0
    encode_information_state_seconds: float = 0.0
    advantage_forward_seconds: float = 0.0
    regret_matching_seconds: float = 0.0
    clone_apply_seconds: float = 0.0
    memory_add_seconds: float = 0.0
    policy_calls: int = 0
    clone_apply_calls: int = 0
    memory_add_calls: int = 0

    def accumulate(self, other: TraversalTimingStats | None) -> None:
        if other is None:
            return
        self.traversal_wall_seconds += other.traversal_wall_seconds
        self.encode_information_state_seconds += other.encode_information_state_seconds
        self.advantage_forward_seconds += other.advantage_forward_seconds
        self.regret_matching_seconds += other.regret_matching_seconds
        self.clone_apply_seconds += other.clone_apply_seconds
        self.memory_add_seconds += other.memory_add_seconds
        self.policy_calls += other.policy_calls
        self.clone_apply_calls += other.clone_apply_calls
        self.memory_add_calls += other.memory_add_calls


def clone_state(state: GameState) -> GameState:
    return state.clone()


class DeepCFRTraverser:
    def __init__(
        self,
        advantage_nets: list[AdvantageNet],
        advantage_memories: list[AdvantageMemory],
        strategy_memory: StrategyMemory,
        *,
        device: torch.device,
        epsilon: float = 1.0e-8,
        strategy_sample_interval: int = 1,
        store_strategy_on_opponent_nodes: bool = True,
        store_strategy_on_traverser_nodes: bool = True,
        max_depth: int | None = None,
        max_nodes_per_traversal: int | None = None,
        cutoff_value_mode: str = "score_diff",
        cutoff_rollouts: int = 0,
        cutoff_rollout_policy: str = "random",
        cutoff_rollout_max_steps: int = 10_000,
        opponent_policy: str = "network",
        league_advantage_nets: list[list[AdvantageNet]] | None = None,
        self_play_league: SelfPlayLeagueConfig | None = None,
        outcome_sampling_epsilon: float = 0.0,
        outcome_sampling_value_clip: float | None = None,
        outcome_unsampled_regret: str = "negative_node_value",
        endpoint_depth_bucket_width: int = DEFAULT_ENDPOINT_DEPTH_BUCKET_WIDTH,
        endpoint_depth_bucket_max: int = DEFAULT_ENDPOINT_DEPTH_BUCKET_MAX,
        rng: np.random.Generator | None = None,
        timing_stats: TraversalTimingStats | None = None,
    ) -> None:
        self.advantage_nets = advantage_nets
        self.advantage_memories = advantage_memories
        self.strategy_memory = strategy_memory
        self.device = device
        self.epsilon = float(epsilon)
        self.strategy_sample_interval = max(1, int(strategy_sample_interval))
        self.store_strategy_on_opponent_nodes = bool(store_strategy_on_opponent_nodes)
        self.store_strategy_on_traverser_nodes = bool(store_strategy_on_traverser_nodes)
        self.max_depth = max_depth
        self.max_nodes_per_traversal = max_nodes_per_traversal
        self.cutoff_value_mode = cutoff_value_mode
        self.cutoff_rollouts = max(0, int(cutoff_rollouts))
        self.cutoff_rollout_policy = cutoff_rollout_policy
        self.cutoff_rollout_max_steps = max(1, int(cutoff_rollout_max_steps))
        self.opponent_policy = opponent_policy
        self.league_advantage_nets = league_advantage_nets or []
        self.self_play_league = self_play_league or SelfPlayLeagueConfig()
        self.outcome_sampling_epsilon = float(outcome_sampling_epsilon)
        self.outcome_sampling_value_clip = (
            None if outcome_sampling_value_clip is None else max(1.0e-9, float(outcome_sampling_value_clip))
        )
        self.outcome_unsampled_regret = outcome_unsampled_regret
        self.endpoint_depth_bucket_width = max(1, int(endpoint_depth_bucket_width))
        self.endpoint_depth_bucket_max = max(1, int(endpoint_depth_bucket_max))
        self.sampling_probability_floor = 1.0e-12
        self.rng = rng or np.random.default_rng()
        self.timing_stats = timing_stats
        self._active_league_snapshot: list[AdvantageNet] | None = None
        self._active_league_bucket: str | None = None
        self._safe_heuristic_rollout_bot = (
            SafeHeuristicBot() if self.cutoff_rollout_policy == "safe_heuristic" else None
        )

    def _node_budget_reached(self, stats: TraversalStats) -> bool:
        return self.max_nodes_per_traversal is not None and stats.nodes >= self.max_nodes_per_traversal

    def _record_endpoint_depth(self, stats: TraversalStats, depth: int, reason: str) -> None:
        stats.endpoint_traversals += 1
        stats.endpoint_depth_sum += depth
        if reason == "terminal":
            stats.terminal_depth_sum += depth
        elif reason == "cutoff":
            stats.cutoff_depth_sum += depth
        elif reason == "node_limit_cutoff":
            stats.node_limit_cutoff_depth_sum += depth

        key = endpoint_depth_bucket_key(
            depth,
            self.endpoint_depth_bucket_width,
            self.endpoint_depth_bucket_max,
        )
        stats.endpoint_depth_buckets[key] = stats.endpoint_depth_buckets.get(key, 0) + 1

    def traverse(self, state: GameState, traverser: int, iteration: int) -> tuple[float, TraversalStats]:
        stats = TraversalStats(
            endpoint_depth_bucket_width=self.endpoint_depth_bucket_width,
            endpoint_depth_bucket_max=self.endpoint_depth_bucket_max,
        )
        if self.opponent_policy == "self_play_league":
            self._active_league_bucket = self._league_bucket()
            self._record_league_bucket(stats, self._active_league_bucket)
            self._active_league_snapshot = self._select_league_snapshot()
        try:
            if self.timing_stats is None:
                value = self._traverse(state, traverser, iteration, 0, stats)
                return value, stats
            started = time.perf_counter()
            value = self._traverse(state, traverser, iteration, 0, stats)
            self.timing_stats.traversal_wall_seconds += time.perf_counter() - started
            return value, stats
        finally:
            self._active_league_snapshot = None
            self._active_league_bucket = None

    def _policy_from_net(
        self,
        net: AdvantageNet,
        state: GameState,
        player: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        info = encode_information_state(state, player)
        legal = legal_mask_array(state)
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = net(x).squeeze(0).detach().cpu().numpy()
        policy = regret_matching(advantages, legal, self.epsilon)
        return info, legal, np.asarray(policy, dtype=np.float32)

    def _policy(self, state: GameState, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.timing_stats is None:
            return self._policy_from_net(self.advantage_nets[player], state, player)

        self.timing_stats.policy_calls += 1
        started = time.perf_counter()
        info = encode_information_state(state, player)
        self.timing_stats.encode_information_state_seconds += time.perf_counter() - started
        legal = legal_mask_array(state)
        started = time.perf_counter()
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = self.advantage_nets[player](x).squeeze(0).detach().cpu().numpy()
        self.timing_stats.advantage_forward_seconds += time.perf_counter() - started
        started = time.perf_counter()
        policy = regret_matching(advantages, legal, self.epsilon)
        self.timing_stats.regret_matching_seconds += time.perf_counter() - started
        return info, legal, np.asarray(policy, dtype=np.float32)

    def _league_bucket(self) -> str:
        cfg = self.self_play_league
        recent_count = min(len(self.league_advantage_nets), cfg.recent_window)
        older_count = max(0, len(self.league_advantage_nets) - recent_count)
        weights = np.asarray(
            [
                cfg.current_weight,
                cfg.recent_weight if recent_count > 0 else 0.0,
                cfg.older_weight if older_count > 0 else 0.0,
                cfg.anchor_weight,
            ],
            dtype=np.float64,
        )
        total = float(weights.sum())
        if total <= 0.0:
            return "current"
        weights /= total
        return str(self.rng.choice(["current", "recent", "older", "anchor"], p=weights))

    def _record_league_bucket(self, stats: TraversalStats, bucket: str) -> None:
        if bucket == "current":
            stats.league_current_traversals += 1
        elif bucket == "recent":
            stats.league_recent_traversals += 1
        elif bucket == "older":
            stats.league_older_traversals += 1
        elif bucket == "anchor":
            stats.league_anchor_traversals += 1

    def _select_league_snapshot(self, bucket: str | None = None) -> list[AdvantageNet] | None:
        if bucket is None:
            bucket = self._active_league_bucket or "current"
        if bucket in {"current", "anchor"} or not self.league_advantage_nets:
            return None
        recent_count = min(len(self.league_advantage_nets), self.self_play_league.recent_window)
        if bucket == "recent" and recent_count > 0:
            candidates = self.league_advantage_nets[-recent_count:]
        else:
            candidates = self.league_advantage_nets[: max(0, len(self.league_advantage_nets) - recent_count)]
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _league_advantage_net(self, player: int) -> AdvantageNet:
        snapshot = self._active_league_snapshot
        if snapshot is None:
            return self.advantage_nets[player]
        return snapshot[player]

    def _self_play_league_action(self, state: GameState, player: int) -> int | None:
        if self._active_league_bucket == "anchor":
            return self._rollout_bot_action(state, self.self_play_league.anchor_policy)
        net = self._league_advantage_net(player)
        _, legal, policy = self._policy_from_net(net, state, player)
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            return None
        return self._sample_legal_action(policy, legal_actions)

    def _record_strategy(
        self,
        info: np.ndarray,
        legal: np.ndarray,
        policy: np.ndarray,
        player: int,
        traverser: int,
        iteration: int,
        depth: int,
    ) -> None:
        if player == traverser:
            if not self.store_strategy_on_traverser_nodes:
                return
        elif not self.store_strategy_on_opponent_nodes:
            return
        if depth % self.strategy_sample_interval != 0:
            return
        if self.timing_stats is None:
            self.strategy_memory.add(info, policy, legal, player, iteration, self.rng)
            return
        started = time.perf_counter()
        self.strategy_memory.add(info, policy, legal, player, iteration, self.rng)
        self.timing_stats.memory_add_seconds += time.perf_counter() - started
        self.timing_stats.memory_add_calls += 1

    def _child_after_action(self, state: GameState, action: int) -> GameState:
        if self.timing_stats is None:
            child = clone_state(state)
            self._sample_deck_draw_chance(child, action)
            child.apply_unified_action(action)
            return child
        started = time.perf_counter()
        child = clone_state(state)
        self._sample_deck_draw_chance(child, action)
        child.apply_unified_action(action)
        self.timing_stats.clone_apply_seconds += time.perf_counter() - started
        self.timing_stats.clone_apply_calls += 1
        return child

    def _sample_deck_draw_chance(self, state: GameState, action: int) -> None:
        deck_draw_action = state.card_action_size
        if state.phase != "draw" or action != deck_draw_action or len(state.deck) <= 1:
            return
        sampled_index = int(self.rng.integers(0, len(state.deck)))
        state.deck[sampled_index], state.deck[-1] = state.deck[-1], state.deck[sampled_index]

    def _sampling_policy(self, policy: np.ndarray, legal: np.ndarray) -> np.ndarray:
        legal_count = int(legal.sum())
        if legal_count <= 0:
            return np.zeros_like(policy, dtype=np.float32)
        eps = min(1.0, max(0.0, self.outcome_sampling_epsilon))
        if eps <= 0.0:
            return np.asarray(policy, dtype=np.float32)
        uniform = legal.astype(np.float32) / float(legal_count)
        return ((1.0 - eps) * policy + eps * uniform).astype(np.float32)

    def _sample_legal_action(self, sampling_policy: np.ndarray, legal_actions: np.ndarray) -> int:
        legal_probs = sampling_policy[legal_actions].astype(np.float64)
        total = float(legal_probs.sum())
        if total <= 0.0:
            legal_probs = np.full(len(legal_actions), 1.0 / float(len(legal_actions)))
        else:
            legal_probs /= total
        return int(self.rng.choice(legal_actions, p=legal_probs))

    def _outcome_sampled_action_value(
        self,
        value: float,
        sampled_action: int,
        sampling_policy: np.ndarray,
    ) -> float:
        action_sample_prob = float(sampling_policy[sampled_action])
        if action_sample_prob <= self.sampling_probability_floor:
            return 0.0
        sampled_value = float(value) / action_sample_prob
        if self.outcome_sampling_value_clip is not None:
            sampled_value = float(
                np.clip(
                    sampled_value,
                    -self.outcome_sampling_value_clip,
                    self.outcome_sampling_value_clip,
                )
            )
        return sampled_value

    def _sampled_node_value(
        self,
        sampled_action_value: float,
        sampled_action: int,
        policy: np.ndarray,
    ) -> float:
        return float(policy[sampled_action]) * sampled_action_value

    def _outcome_sampled_regrets(
        self,
        sampled_action_value: float,
        sampled_action: int,
        policy: np.ndarray,
        legal: np.ndarray,
    ) -> np.ndarray:
        node_value = float(policy[sampled_action]) * sampled_action_value
        if self.outcome_unsampled_regret == "zero":
            regrets = np.zeros_like(policy, dtype=np.float32)
        else:
            regrets = np.where(legal, -node_value, 0.0).astype(np.float32)
        regrets[sampled_action] = np.float32(sampled_action_value - node_value)
        return regrets

    def _cutoff_rollout_action(self, rollout_state: GameState) -> int | None:
        return self._rollout_bot_action(rollout_state, self.cutoff_rollout_policy)

    def _rollout_bot_action(self, rollout_state: GameState, policy: str) -> int | None:
        legal_actions = np.flatnonzero(rollout_state.unified_legal_mask())
        if len(legal_actions) == 0:
            return None
        if policy == "random":
            return int(self.rng.choice(legal_actions))
        if policy == "safe_heuristic":
            if self._safe_heuristic_rollout_bot is None:
                self._safe_heuristic_rollout_bot = SafeHeuristicBot()
            action = self._safe_heuristic_rollout_bot.act(rollout_state)
            return rollout_state.to_unified_action(action)
        raise ValueError(f"unsupported rollout policy: {policy!r}")

    def _fixed_opponent_action(self, state: GameState) -> int | None:
        if self.opponent_policy == "network":
            return None
        if self.opponent_policy == "self_play_league":
            return self._self_play_league_action(state, state.current_player)
        return self._rollout_bot_action(state, self.opponent_policy)

    def _rollout_value(self, state: GameState, traverser: int, stats: TraversalStats) -> float:
        rollout_state = clone_state(state)
        steps = 0
        while not rollout_state.terminal and steps < self.cutoff_rollout_max_steps:
            action = self._cutoff_rollout_action(rollout_state)
            if action is None:
                break
            self._sample_deck_draw_chance(rollout_state, action)
            rollout_state.apply_unified_action(action)
            steps += 1
        stats.cutoff_rollouts += 1
        stats.cutoff_rollout_steps += steps
        if not rollout_state.terminal:
            stats.cutoff_rollout_max_step_timeouts += 1
        return float(rollout_state.score_diff(traverser))

    def _cutoff_value(self, state: GameState, traverser: int, stats: TraversalStats) -> float:
        if self.cutoff_value_mode == "score_diff" or self.cutoff_rollouts <= 0:
            return float(state.score_diff(traverser))
        if self.cutoff_value_mode != "random_rollout":
            raise ValueError(f"unsupported cutoff_value_mode: {self.cutoff_value_mode!r}")
        total = 0.0
        for _ in range(self.cutoff_rollouts):
            total += self._rollout_value(state, traverser, stats)
        return total / float(self.cutoff_rollouts)

    def _traverse(
        self,
        state: GameState,
        traverser: int,
        iteration: int,
        depth: int,
        stats: TraversalStats,
    ) -> float:
        stats.nodes += 1
        stats.max_depth_reached = max(stats.max_depth_reached, depth)
        if self._node_budget_reached(stats):
            stats.node_limit_cutoffs += 1
            self._record_endpoint_depth(stats, depth, "node_limit_cutoff")
            return self._cutoff_value(state, traverser, stats)
        if state.terminal:
            stats.terminals += 1
            self._record_endpoint_depth(stats, depth, "terminal")
            return float(state.score_diff(traverser))
        if self.max_depth is not None and depth >= self.max_depth:
            stats.cutoffs += 1
            self._record_endpoint_depth(stats, depth, "cutoff")
            return self._cutoff_value(state, traverser, stats)

        player = state.current_player
        if player != traverser and self.opponent_policy != "network":
            action = self._fixed_opponent_action(state)
            if action is None:
                stats.terminals += 1
                self._record_endpoint_depth(stats, depth, "terminal")
                return float(state.score_diff(traverser))
            child = self._child_after_action(state, action)
            return self._traverse(
                child,
                traverser,
                iteration,
                depth + 1,
                stats,
            )

        info, legal, policy = self._policy(state, player)
        self._record_strategy(info, legal, policy, player, traverser, iteration, depth)
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            stats.terminals += 1
            self._record_endpoint_depth(stats, depth, "terminal")
            return float(state.score_diff(traverser))

        sampling_policy = self._sampling_policy(policy, legal)
        action = self._sample_legal_action(sampling_policy, legal_actions)
        child = self._child_after_action(state, action)
        value = self._traverse(
            child,
            traverser,
            iteration,
            depth + 1,
            stats,
        )
        sampled_action_value = self._outcome_sampled_action_value(value, action, sampling_policy)
        node_value = self._sampled_node_value(sampled_action_value, action, policy)

        if player == traverser:
            regrets = self._outcome_sampled_regrets(
                sampled_action_value,
                action,
                policy,
                legal,
            )
            if self.timing_stats is None:
                self.advantage_memories[traverser].add(
                    info,
                    regrets,
                    legal,
                    traverser,
                    iteration,
                    self.rng,
                )
            else:
                started = time.perf_counter()
                self.advantage_memories[traverser].add(
                    info,
                    regrets,
                    legal,
                    traverser,
                    iteration,
                    self.rng,
                )
                self.timing_stats.memory_add_seconds += time.perf_counter() - started
                self.timing_stats.memory_add_calls += 1
        return node_value


def cfr_traverse(
    state: GameState,
    traverser: int,
    iteration: int,
    advantage_nets: list[AdvantageNet],
    advantage_memories: list[AdvantageMemory],
    strategy_memory: StrategyMemory,
    *,
    device: torch.device | None = None,
    epsilon: float = 1.0e-8,
    strategy_sample_interval: int = 1,
    store_strategy_on_opponent_nodes: bool = True,
    store_strategy_on_traverser_nodes: bool = True,
    max_depth: int | None = None,
    max_nodes_per_traversal: int | None = None,
    cutoff_value_mode: str = "score_diff",
    cutoff_rollouts: int = 0,
    cutoff_rollout_policy: str = "random",
    cutoff_rollout_max_steps: int = 10_000,
    opponent_policy: str = "network",
    league_advantage_nets: list[list[AdvantageNet]] | None = None,
    self_play_league: SelfPlayLeagueConfig | None = None,
    outcome_sampling_epsilon: float = 0.0,
    outcome_sampling_value_clip: float | None = None,
    outcome_unsampled_regret: str = "negative_node_value",
    endpoint_depth_bucket_width: int = DEFAULT_ENDPOINT_DEPTH_BUCKET_WIDTH,
    endpoint_depth_bucket_max: int = DEFAULT_ENDPOINT_DEPTH_BUCKET_MAX,
    rng: np.random.Generator | None = None,
    timing_stats: TraversalTimingStats | None = None,
) -> tuple[float, TraversalStats]:
    traverser_obj = DeepCFRTraverser(
        advantage_nets,
        advantage_memories,
        strategy_memory,
        device=device or torch.device("cpu"),
        epsilon=epsilon,
        strategy_sample_interval=strategy_sample_interval,
        store_strategy_on_opponent_nodes=store_strategy_on_opponent_nodes,
        store_strategy_on_traverser_nodes=store_strategy_on_traverser_nodes,
        max_depth=max_depth,
        max_nodes_per_traversal=max_nodes_per_traversal,
        cutoff_value_mode=cutoff_value_mode,
        cutoff_rollouts=cutoff_rollouts,
        cutoff_rollout_policy=cutoff_rollout_policy,
        cutoff_rollout_max_steps=cutoff_rollout_max_steps,
        opponent_policy=opponent_policy,
        league_advantage_nets=league_advantage_nets,
        self_play_league=self_play_league,
        outcome_sampling_epsilon=outcome_sampling_epsilon,
        outcome_sampling_value_clip=outcome_sampling_value_clip,
        outcome_unsampled_regret=outcome_unsampled_regret,
        endpoint_depth_bucket_width=endpoint_depth_bucket_width,
        endpoint_depth_bucket_max=endpoint_depth_bucket_max,
        rng=rng,
        timing_stats=timing_stats,
    )
    return traverser_obj.traverse(state, traverser, iteration)
