from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def _selfplay_temperature(move_count: int, temperature_moves: int, temperature_end: float) -> float:
    if temperature_moves <= 0:
        return float(temperature_end)
    if move_count >= temperature_moves:
        return float(temperature_end)
    frac = move_count / float(temperature_moves)
    return 1.0 + (float(temperature_end) - 1.0) * frac


_WORKER_STATE: dict[str, Any] = {}


def worker_init(config_payload: dict, state_numpy: dict[str, np.ndarray]) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from tinygrad import Device, Tensor
    from tinygrad.nn.state import load_state_dict

    from .config import config_from_dict
    from .evaluator import ModelEvaluator
    from .mcts_backend import resolve_mcts_backend
    from .network import PolicyValueNet

    Device.DEFAULT = "CPU"
    config = config_from_dict(config_payload)
    mcts_module = resolve_mcts_backend(config.selfplay.mcts_backend)
    model = PolicyValueNet(config.rules.board_size, config.network)
    tensor_state = {key: Tensor(np.asarray(value)) for key, value in state_numpy.items()}
    load_state_dict(model, tensor_state, strict=True, verbose=False)
    evaluator = ModelEvaluator(model, device="CPU")
    search = mcts_module.MCTS(
        c_puct=config.selfplay.c_puct,
        dirichlet_alpha=config.selfplay.dirichlet_alpha,
        dirichlet_epsilon=config.selfplay.dirichlet_epsilon,
        evaluator=evaluator,
        search_threads=config.selfplay.search_threads,
    )
    _WORKER_STATE.clear()
    _WORKER_STATE["config"] = config
    _WORKER_STATE["model"] = model
    _WORKER_STATE["evaluator"] = evaluator
    _WORKER_STATE["search"] = search
    _WORKER_STATE["Tensor"] = Tensor


def run_selfplay_chunk(
    openings: list[list[int]],
    simulations: int,
    seed: int,
    chunk_id: int,
    progress_queue: Any | None = None,
) -> list[tuple[list[dict], int]]:
    from .board import GameState
    from .replay import PendingSample

    config = _WORKER_STATE["config"]
    search = _WORKER_STATE["search"]
    Tensor = _WORKER_STATE["Tensor"]

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    Tensor.manual_seed(seed)
    if progress_queue is not None:
        progress_queue.put(
            {
                "type": "chunk_started",
                "chunk_id": chunk_id,
                "chunk_size": len(openings),
                "pid": os.getpid(),
            }
        )

    finished: list[tuple[list[dict], int]] = []
    states: list[GameState] = []
    histories: list[list[PendingSample]] = []
    roots = []
    completed_in_chunk = 0

    for opening in openings:
        state = GameState(config.rules.board_size, config.rules.exactly_five)
        for action in opening:
            if state.terminal:
                break
            if state.legal_moves()[action]:
                state.apply_action(action)
        if state.terminal:
            finished.append(([], state.winner))
            completed_in_chunk += 1
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "type": "game_done",
                        "chunk_id": chunk_id,
                        "chunk_size": len(openings),
                        "chunk_done": completed_in_chunk,
                        "pid": os.getpid(),
                        "moves": 0,
                        "winner": state.winner,
                    }
                )
            continue
        states.append(state)
        histories.append([])
        roots.append(None)

    while states:
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "game_moves",
                    "chunk_id": chunk_id,
                    "pid": os.getpid(),
                    "moves": len(states),
                    "active_games": len(states),
                }
            )
        temperatures = [
            _selfplay_temperature(
                state.move_count,
                config.selfplay.temperature_moves,
                config.selfplay.temperature_end,
            )
            for state in states
        ]
        results = search.search_batch(
            states,
            simulations,
            temperatures,
            add_noise=True,
            roots=roots,
            leaves_per_batch=config.selfplay.leaves_per_batch,
        )
        next_states: list[GameState] = []
        next_histories: list[list[PendingSample]] = []
        next_roots = []
        for state, history, result in zip(states, histories, results, strict=True):
            history.append(
                PendingSample(
                    board=state.board.copy(),
                    to_play=state.to_play,
                    last_action=state.last_action,
                    policy=result.visit_policy.copy(),
                )
            )
            state.apply_action(result.action)
            if state.terminal:
                finished.append(
                    (
                        [
                            {
                                "board": sample.board,
                                "to_play": sample.to_play,
                                "last_action": sample.last_action,
                                "policy": sample.policy,
                            }
                            for sample in history
                        ],
                        state.winner,
                    )
                )
                completed_in_chunk += 1
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "type": "game_done",
                            "chunk_id": chunk_id,
                            "chunk_size": len(openings),
                            "chunk_done": completed_in_chunk,
                            "pid": os.getpid(),
                            "moves": len(history),
                            "winner": state.winner,
                        }
                    )
            else:
                next_states.append(state)
                next_histories.append(history)
                next_roots.append(result.next_root)
        states = next_states
        histories = next_histories
        roots = next_roots

    return finished


def model_state_to_numpy(model) -> dict[str, np.ndarray]:
    return {
        key: np.array(value.realize().numpy(), copy=True)
        for key, value in model.state_dict().items()
    }
