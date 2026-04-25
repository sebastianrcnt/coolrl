from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from ..game import GameState
from .encoding import encode_information_state, legal_mask_array
from .memory import AdvantageMemory, StrategyMemory
from .networks import AdvantageNet, regret_matching


@dataclass(slots=True)
class TraversalStats:
    nodes: int = 0
    terminals: int = 0
    cutoffs: int = 0
    node_limit_cutoffs: int = 0
    max_depth_reached: int = 0
    cutoff_rollouts: int = 0
    cutoff_rollout_steps: int = 0
    cutoff_rollout_max_step_timeouts: int = 0

    def accumulate(self, other: TraversalStats) -> None:
        self.nodes += other.nodes
        self.terminals += other.terminals
        self.cutoffs += other.cutoffs
        self.node_limit_cutoffs += other.node_limit_cutoffs
        self.max_depth_reached = max(self.max_depth_reached, other.max_depth_reached)
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
        self.rng = rng or np.random.default_rng()
        self.timing_stats = timing_stats

    def _node_budget_reached(self, stats: TraversalStats) -> bool:
        return self.max_nodes_per_traversal is not None and stats.nodes >= self.max_nodes_per_traversal

    def traverse(self, state: GameState, traverser: int, iteration: int) -> tuple[float, TraversalStats]:
        stats = TraversalStats()
        if self.timing_stats is None:
            value = self._traverse(state, traverser, iteration, 0, stats)
            return value, stats
        started = time.perf_counter()
        value = self._traverse(state, traverser, iteration, 0, stats)
        self.timing_stats.traversal_wall_seconds += time.perf_counter() - started
        return value, stats

    def _policy(self, state: GameState, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.timing_stats is None:
            info = encode_information_state(state, player)
            legal = legal_mask_array(state)
            with torch.inference_mode():
                x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
                advantages = self.advantage_nets[player](x).squeeze(0).detach().cpu().numpy()
            policy = regret_matching(advantages, legal, self.epsilon)
            return info, legal, np.asarray(policy, dtype=np.float32)

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
            child.apply_unified_action(action)
            return child
        started = time.perf_counter()
        child = clone_state(state)
        child.apply_unified_action(action)
        self.timing_stats.clone_apply_seconds += time.perf_counter() - started
        self.timing_stats.clone_apply_calls += 1
        return child

    def _random_rollout_value(self, state: GameState, traverser: int, stats: TraversalStats) -> float:
        rollout_state = clone_state(state)
        steps = 0
        while not rollout_state.terminal and steps < self.cutoff_rollout_max_steps:
            legal_actions = np.flatnonzero(rollout_state.unified_legal_mask())
            if len(legal_actions) == 0:
                break
            action = int(self.rng.choice(legal_actions))
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
        if self.cutoff_rollout_policy != "random":
            raise ValueError(f"unsupported cutoff_rollout_policy: {self.cutoff_rollout_policy!r}")
        total = 0.0
        for _ in range(self.cutoff_rollouts):
            total += self._random_rollout_value(state, traverser, stats)
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
        if self._node_budget_reached(stats):
            stats.node_limit_cutoffs += 1
            return self._cutoff_value(state, traverser, stats)
        stats.max_depth_reached = max(stats.max_depth_reached, depth)
        if state.terminal:
            stats.terminals += 1
            return float(state.score_diff(traverser))
        if self.max_depth is not None and depth >= self.max_depth:
            stats.cutoffs += 1
            stats.max_depth_reached = max(stats.max_depth_reached, depth)
            return self._cutoff_value(state, traverser, stats)

        player = state.current_player
        info, legal, policy = self._policy(state, player)
        self._record_strategy(info, legal, policy, player, traverser, iteration, depth)
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            stats.terminals += 1
            return float(state.score_diff(traverser))

        if player == traverser:
            action_values = np.zeros(state.action_size, dtype=np.float32)
            for index, action in enumerate(legal_actions):
                if self._node_budget_reached(stats):
                    remaining_actions = legal_actions[index:]
                    # Treat every unexpanded traverser action as a budget cutoff
                    # without recursing further, so max_nodes_per_traversal acts as
                    # a hard cap instead of a soft budget.
                    stats.node_limit_cutoffs += len(remaining_actions)
                    for remaining_action in remaining_actions:
                        child = self._child_after_action(state, int(remaining_action))
                        action_values[remaining_action] = self._cutoff_value(child, traverser, stats)
                    break
                child = self._child_after_action(state, int(action))
                action_values[action] = self._traverse(
                    child,
                    traverser,
                    iteration,
                    depth + 1,
                    stats,
                )
            node_value = float(np.dot(policy, action_values))
            regrets = np.where(legal, action_values - node_value, 0.0).astype(np.float32)
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

        action = int(self.rng.choice(legal_actions, p=policy[legal_actions] / policy[legal_actions].sum()))
        child = self._child_after_action(state, action)
        return self._traverse(
            child,
            traverser,
            iteration,
            depth + 1,
            stats,
        )


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
        rng=rng,
        timing_stats=timing_stats,
    )
    return traverser_obj.traverse(state, traverser, iteration)
