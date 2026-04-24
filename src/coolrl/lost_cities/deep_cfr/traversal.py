from __future__ import annotations

from dataclasses import dataclass

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
    max_depth_reached: int = 0


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
        max_depth: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.advantage_nets = advantage_nets
        self.advantage_memories = advantage_memories
        self.strategy_memory = strategy_memory
        self.device = device
        self.epsilon = float(epsilon)
        self.strategy_sample_interval = max(1, int(strategy_sample_interval))
        self.max_depth = max_depth
        self.rng = rng or np.random.default_rng()

    def traverse(self, state: GameState, traverser: int, iteration: int) -> tuple[float, TraversalStats]:
        stats = TraversalStats()
        value = self._traverse(state, traverser, iteration, 0, stats)
        return value, stats

    def _policy(self, state: GameState, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        info = encode_information_state(state, player)
        legal = legal_mask_array(state)
        with torch.no_grad():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = self.advantage_nets[player](x).squeeze(0).detach().cpu().numpy()
        policy = regret_matching(advantages, legal, self.epsilon)
        return info, legal, np.asarray(policy, dtype=np.float32)

    def _record_strategy(
        self,
        info: np.ndarray,
        legal: np.ndarray,
        policy: np.ndarray,
        player: int,
        iteration: int,
        depth: int,
    ) -> None:
        if depth % self.strategy_sample_interval == 0:
            self.strategy_memory.add(info, policy, legal, player, iteration, self.rng)

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
        if state.terminal:
            stats.terminals += 1
            return float(state.score_diff(traverser))
        if self.max_depth is not None and depth >= self.max_depth:
            stats.cutoffs += 1
            stats.max_depth_reached = max(stats.max_depth_reached, depth)
            return float(state.score_diff(traverser))

        player = state.current_player
        info, legal, policy = self._policy(state, player)
        self._record_strategy(info, legal, policy, player, iteration, depth)
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            stats.terminals += 1
            return float(state.score_diff(traverser))

        if player == traverser:
            action_values = np.zeros(state.action_size, dtype=np.float32)
            for action in legal_actions:
                child = clone_state(state)
                child.apply_unified_action(int(action))
                action_values[action] = self._traverse(
                    child,
                    traverser,
                    iteration,
                    depth + 1,
                    stats,
                )
            node_value = float(np.dot(policy, action_values))
            regrets = np.where(legal, action_values - node_value, 0.0).astype(np.float32)
            self.advantage_memories[traverser].add(
                info,
                regrets,
                legal,
                traverser,
                iteration,
                self.rng,
            )
            return node_value

        action = int(self.rng.choice(legal_actions, p=policy[legal_actions] / policy[legal_actions].sum()))
        child = clone_state(state)
        child.apply_unified_action(action)
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
    max_depth: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, TraversalStats]:
    traverser_obj = DeepCFRTraverser(
        advantage_nets,
        advantage_memories,
        strategy_memory,
        device=device or torch.device("cpu"),
        epsilon=epsilon,
        max_depth=max_depth,
        rng=rng,
    )
    return traverser_obj.traverse(state, traverser, iteration)
