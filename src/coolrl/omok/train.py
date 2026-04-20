from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tinygrad import Tensor
from tinygrad.nn.optim import AdamW
from tqdm import tqdm

from .arena import Arena
from .board import GameState
from .checkpoint import (
    load_checkpoint,
    load_optimizer_state,
    load_trainer_state,
    save_checkpoint,
    save_optimizer_state,
    save_trainer_state,
)
from .config import RunConfig, load_config
from .device import configure_device
from .evaluator import ModelEvaluator
from .mcts import MCTS
from .network import PolicyValueNet, clone_model
from .openings import sample_balanced_openings
from .replay import PendingSample, ReplayBuffer
from .selfplay_worker import model_state_to_numpy, run_selfplay_chunk, worker_init


class TqdmLogSink:
    def write(self, message: str) -> None:
        if message.rstrip():
            tqdm.write(message, end="")

    def flush(self) -> None:
        sys.stderr.flush()


def configure_logging() -> None:
    logger.remove()
    logger.add(
        TqdmLogSink(),
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    Tensor.manual_seed(seed)


class Trainer:
    def __init__(self, config: RunConfig, resume_path: str | None = None) -> None:
        self.config = config
        self.device = configure_device(config.device)
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.checkpoint_dir / "metrics.jsonl"
        self.progress_file = self.checkpoint_dir / "runtime_progress.json"
        self.replay = ReplayBuffer(config.optimization.replay_capacity)
        self.model = PolicyValueNet(config.rules.board_size, config.network)
        self.best_model = clone_model(self.model)
        self.optimizer = self._build_optimizer()
        self.selfplay_rng = random.Random(config.seed + 1_048_583)
        self.iteration = 0
        self.total_updates = 0
        self.best_iteration = 0
        self.best_arena_win_rate = 0.0
        self.best_checkpoint_metadata: dict[str, Any] = {
            "iteration": 0,
            "best_iteration": 0,
            "best_arena_win_rate": 0.0,
            "total_updates": 0,
            "status": "initial_best",
        }
        self.start_time = time.monotonic()
        self.elapsed_seconds_offset = 0.0
        self.stop_requested = False
        self.num_workers, num_workers_auto = config.selfplay.resolved_num_workers()
        if num_workers_auto:
            logger.info(
                "Self-play num_workers=auto resolved to {} (os.cpu_count={})",
                self.num_workers,
                os.cpu_count(),
            )
        else:
            logger.info("Self-play num_workers={}", self.num_workers)
        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        if resume_path:
            self._restore_from_checkpoint(resume_path)

    def _build_optimizer(self) -> AdamW:
        return AdamW(
            self.model.parameters(),
            lr=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay,
        )

    @property
    def elapsed_hours(self) -> float:
        return (self.elapsed_seconds_offset + time.monotonic() - self.start_time) / 3600.0

    def run(self) -> None:
        logger.info(
            "Starting 9x9 Omok RL: experiment={} device={} checkpoint_dir={}",
            self.config.experiment_name,
            self.device,
            self.checkpoint_dir,
        )
        self.save_model_checkpoints({"iteration": self.iteration, "status": "startup"})
        self.save_runtime_state({"iteration": self.iteration, "status": "startup"})

        while self._should_start_iteration():
            self.iteration += 1
            iteration_started = time.monotonic()
            simulations = self.current_simulations()
            logger.info("Iteration {} started: simulations={}", self.iteration, simulations)

            selfplay_stats = self.generate_selfplay(simulations)
            if self._should_stop():
                self.save_runtime_state({"iteration": self.iteration, "status": "stopped_after_selfplay"})
                break

            training_stats: dict[str, float | int] = {}
            arena_stats: dict[str, float | int | bool | str] = {}
            if self.replay.games_seen < self.config.optimization.warmup_games:
                logger.info(
                    "Warmup: replay_games={} warmup_games={}",
                    self.replay.games_seen,
                    self.config.optimization.warmup_games,
                )
                status = "warmup"
            else:
                training_stats = self.train_model()
                arena_stats = self.evaluate_candidate(simulations)
                if bool(arena_stats.get("accepted", False)):
                    self.best_model = clone_model(self.model)
                    self.best_iteration = self.iteration
                    self.best_arena_win_rate = float(arena_stats.get("arena_win_rate", 0.0))
                    self.best_checkpoint_metadata = {
                        "iteration": self.iteration,
                        "best_iteration": self.best_iteration,
                        "best_arena_win_rate": self.best_arena_win_rate,
                        "total_updates": self.total_updates,
                        "status": "accepted",
                    }
                    logger.success(
                        "Promoted new best: iteration={} arena_win_rate={:.3f}",
                        self.best_iteration,
                        self.best_arena_win_rate,
                    )
                status = "trained"

            metadata = {
                "iteration": self.iteration,
                "elapsed_hours": round(self.elapsed_hours, 4),
                "status": status,
                "simulations": simulations,
                "best_iteration": self.best_iteration,
                "best_arena_win_rate": round(self.best_arena_win_rate, 4),
                "total_updates": self.total_updates,
                "duration_seconds": round(time.monotonic() - iteration_started, 3),
                **selfplay_stats,
                **training_stats,
                **arena_stats,
            }
            self._log(metadata)
            self.save_model_checkpoints(metadata)
            self.save_runtime_state(metadata)

        logger.success(
            "Training stopped: iteration={} elapsed_hours={:.4f} best_iteration={}",
            self.iteration,
            self.elapsed_hours,
            self.best_iteration,
        )

    def current_simulations(self) -> int:
        schedule = self.config.selfplay.simulation_schedule
        if not schedule:
            raise ValueError("simulation_schedule must not be empty")
        if self.config.max_iterations:
            fraction = min(1.0, max(0, self.iteration - 1) / max(1, self.config.max_iterations))
        elif self.config.max_hours:
            fraction = min(1.0, self.elapsed_hours / max(self.config.max_hours, 1.0e-6))
        elif self.config.selfplay.simulation_ramp_iterations:
            ramp = max(1, self.config.selfplay.simulation_ramp_iterations)
            fraction = min(1.0, max(0, self.iteration - 1) / ramp)
        else:
            return int(schedule[-1]["simulations"])

        simulations = int(schedule[0]["simulations"])
        for point in schedule:
            if fraction >= float(point["fraction"]):
                simulations = int(point["simulations"])
        return simulations

    def current_selfplay_plan(self) -> dict[str, int | str]:
        total_games = self.config.selfplay.games_per_iteration
        if self.best_iteration == 0:
            return {"label": "candidate", "candidate_games": total_games}

        mix_iterations = max(0, self.config.selfplay.mixed_iterations_after_promotion)
        mix_fraction = max(0.0, min(1.0, self.config.selfplay.candidate_mix_fraction))
        if mix_iterations > 0 and self.iteration <= self.best_iteration + mix_iterations and mix_fraction > 0.0:
            candidate_games = int(round(total_games * mix_fraction))
            candidate_games = min(total_games, max(0, candidate_games))
            if candidate_games == total_games:
                return {"label": "candidate", "candidate_games": candidate_games}
            if candidate_games > 0:
                return {"label": "mixed", "candidate_games": candidate_games}
        return {"label": "best", "candidate_games": 0}

    def generate_selfplay(self, simulations: int) -> dict[str, float | int]:
        selfplay_plan = self.current_selfplay_plan()
        source_label = str(selfplay_plan["label"])
        candidate_games = int(selfplay_plan["candidate_games"])
        total_games = self.config.selfplay.games_per_iteration
        openings = sample_balanced_openings(
            self.config.rules.board_size,
            total_games,
            self.selfplay_rng,
        )
        source_openings = {
            "candidate": openings[:candidate_games],
            "best": openings[candidate_games:],
        }
        black_wins = 0
        white_wins = 0
        draws = 0
        total_moves = 0
        candidate_completed = 0
        best_completed = 0

        with tqdm(total=len(openings), desc="Self-play", unit="game", leave=False) as progress:
            for source, source_model in (("candidate", self.model), ("best", self.best_model)):
                if not source_openings[source] or self._should_stop():
                    continue
                stats = self._run_selfplay_source(
                    source=source,
                    model=source_model,
                    openings=source_openings[source],
                    simulations=simulations,
                    progress=progress,
                )
                black_wins += int(stats["black_wins"])
                white_wins += int(stats["white_wins"])
                draws += int(stats["draws"])
                total_moves += int(stats["total_moves"])
                if source == "candidate":
                    candidate_completed += int(stats["games"])
                else:
                    best_completed += int(stats["games"])

        games = black_wins + white_wins + draws
        avg_moves = 0.0 if games == 0 else total_moves / games
        logger.info(
            "Self-play finished: source={} games={} black={} white={} draws={} avg_moves={:.2f}",
            source_label,
            games,
            black_wins,
            white_wins,
            draws,
            avg_moves,
        )
        return {
            "selfplay_source": source_label,
            "selfplay_games": games,
            "selfplay_black_wins": black_wins,
            "selfplay_white_wins": white_wins,
            "selfplay_draws": draws,
            "selfplay_avg_moves": round(avg_moves, 2),
            "selfplay_candidate_games": candidate_completed,
            "selfplay_best_games": best_completed,
            "selfplay_batch_size": max(1, self.config.selfplay.batch_size),
            "replay_samples": len(self.replay),
            "replay_games": self.replay.games_seen,
        }

    def _run_selfplay_source(
        self,
        source: str,
        model: PolicyValueNet,
        openings: list[list[int]],
        simulations: int,
        progress: tqdm,
    ) -> dict[str, int]:
        num_workers = self.num_workers
        if num_workers > 0:
            return self._run_selfplay_source_parallel(
                source=source,
                model=model,
                openings=openings,
                simulations=simulations,
                progress=progress,
                num_workers=num_workers,
            )
        return self._run_selfplay_source_sequential(
            source=source,
            model=model,
            openings=openings,
            simulations=simulations,
            progress=progress,
        )

    def _run_selfplay_source_sequential(
        self,
        source: str,
        model: PolicyValueNet,
        openings: list[list[int]],
        simulations: int,
        progress: tqdm,
    ) -> dict[str, int]:
        evaluator = ModelEvaluator(model, device=self.device)
        batch_size = max(1, self.config.selfplay.batch_size)
        black_wins = 0
        white_wins = 0
        draws = 0
        total_moves = 0
        completed_games = 0

        pending_openings = list(openings)
        while pending_openings and not self._should_stop():
            batch_openings = pending_openings[:batch_size]
            del pending_openings[:batch_size]
            search = MCTS(
                c_puct=self.config.selfplay.c_puct,
                dirichlet_alpha=self.config.selfplay.dirichlet_alpha,
                dirichlet_epsilon=self.config.selfplay.dirichlet_epsilon,
                evaluator=evaluator,
            )
            states: list[GameState] = []
            histories: list[list[PendingSample]] = []
            roots = []

            for opening in batch_openings:
                state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
                for action in opening:
                    if state.terminal:
                        break
                    if state.legal_moves()[action]:
                        state.apply_action(action)
                if state.terminal:
                    self.replay.add_game([], state.winner, self.config.optimization.value_discount)
                    black_wins += int(state.winner == 1)
                    white_wins += int(state.winner == -1)
                    draws += int(state.winner == 0)
                    completed_games += 1
                    progress.update(1)
                    continue
                states.append(state)
                histories.append([])
                roots.append(None)

            while states and not self._should_stop():
                temperatures = [self._selfplay_temperature(state.move_count) for state in states]
                results = search.search_batch(
                    states,
                    simulations,
                    temperatures,
                    add_noise=True,
                    roots=roots,
                    leaves_per_batch=self.config.selfplay.leaves_per_batch,
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
                        self.replay.add_game(history, state.winner, self.config.optimization.value_discount)
                        total_moves += len(history)
                        black_wins += int(state.winner == 1)
                        white_wins += int(state.winner == -1)
                        draws += int(state.winner == 0)
                        completed_games += 1
                        progress.update(1)
                    else:
                        next_states.append(state)
                        next_histories.append(history)
                        next_roots.append(result.next_root)
                states = next_states
                histories = next_histories
                roots = next_roots

        logger.debug(
            "Self-play source finished: source={} games={} batch_size={}",
            source,
            completed_games,
            batch_size,
        )
        return {
            "games": completed_games,
            "black_wins": black_wins,
            "white_wins": white_wins,
            "draws": draws,
            "total_moves": total_moves,
        }

    def _run_selfplay_source_parallel(
        self,
        source: str,
        model: PolicyValueNet,
        openings: list[list[int]],
        simulations: int,
        progress: tqdm,
        num_workers: int,
    ) -> dict[str, int]:
        batch_size = max(1, self.config.selfplay.batch_size)
        chunks = [openings[i : i + batch_size] for i in range(0, len(openings), batch_size)]
        if not chunks:
            return {"games": 0, "black_wins": 0, "white_wins": 0, "draws": 0, "total_moves": 0}

        state_numpy = model_state_to_numpy(model)
        config_payload = self.config.to_dict()
        seed_base = (self.config.seed * 1_000_003) ^ (self.iteration * 2_654_435_761) ^ hash(source)

        black_wins = 0
        white_wins = 0
        draws = 0
        total_moves = 0
        completed_games = 0

        ctx = mp.get_context("spawn")
        effective_workers = max(1, min(num_workers, len(chunks)))
        logger.info(
            "Self-play parallel: source={} workers={} chunks={} batch_size={}",
            source,
            effective_workers,
            len(chunks),
            batch_size,
        )
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=ctx,
            initializer=worker_init,
            initargs=(config_payload, state_numpy),
        ) as pool:
            futures = {
                pool.submit(
                    run_selfplay_chunk,
                    chunk,
                    simulations,
                    (seed_base ^ (idx * 0x9E3779B1)) & 0x7FFFFFFF,
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                if self._should_stop():
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    break
                try:
                    finished = future.result()
                except Exception:
                    logger.exception("Self-play worker failed: source={}", source)
                    raise
                for history_dicts, winner in finished:
                    history = [
                        PendingSample(
                            board=item["board"],
                            to_play=item["to_play"],
                            last_action=item["last_action"],
                            policy=item["policy"],
                        )
                        for item in history_dicts
                    ]
                    self.replay.add_game(history, int(winner), self.config.optimization.value_discount)
                    total_moves += len(history)
                    black_wins += int(winner == 1)
                    white_wins += int(winner == -1)
                    draws += int(winner == 0)
                    completed_games += 1
                    progress.update(1)

        logger.debug(
            "Self-play source finished (parallel): source={} games={} batch_size={} workers={}",
            source,
            completed_games,
            batch_size,
            effective_workers,
        )
        return {
            "games": completed_games,
            "black_wins": black_wins,
            "white_wins": white_wins,
            "draws": draws,
            "total_moves": total_moves,
        }

    def train_model(self) -> dict[str, float | int]:
        if len(self.replay) == 0:
            return {"updates_done": 0, "train_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        batch_size = min(self.config.optimization.batch_size, len(self.replay))
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        updates_done = 0

        progress = tqdm(
            range(self.config.optimization.updates_per_iteration),
            desc="Training",
            unit="update",
            leave=False,
        )
        with Tensor.train(True):
            for _ in progress:
                if self._should_stop():
                    break
                states, target_policy, target_value = self.replay.sample_batch(
                    batch_size,
                    device=self.device,
                    recency_temperature=self.config.optimization.recency_temperature,
                )
                self.optimizer.zero_grad()
                logits, value = self.model(states)
                policy_loss = -(target_policy * logits.log_softmax(axis=1)).sum(axis=1).mean()
                value_loss = ((value - target_value) ** 2).mean()
                loss = (
                    self.config.optimization.policy_loss_weight * policy_loss
                    + self.config.optimization.value_loss_weight * value_loss
                )
                loss.backward()
                self.optimizer.step()

                loss_value = float(loss.realize().numpy())
                policy_loss_value = float(policy_loss.realize().numpy())
                value_loss_value = float(value_loss.realize().numpy())
                total_loss += loss_value
                total_policy_loss += policy_loss_value
                total_value_loss += value_loss_value
                updates_done += 1
                self.total_updates += 1
                progress.set_postfix(loss=f"{loss_value:.4f}")

        updates = max(1, updates_done)
        stats = {
            "train_loss": round(total_loss / updates, 6),
            "policy_loss": round(total_policy_loss / updates, 6),
            "value_loss": round(total_value_loss / updates, 6),
            "updates_done": updates_done,
            "learning_rate": self.config.optimization.learning_rate,
        }
        logger.info(
            "Training finished: updates={} loss={:.6f} policy={:.6f} value={:.6f}",
            updates_done,
            stats["train_loss"],
            stats["policy_loss"],
            stats["value_loss"],
        )
        return stats

    def evaluate_candidate(self, selfplay_simulations: int) -> dict[str, float | int | bool | str]:
        if self.config.arena.games <= 0 or self.config.arena.simulations <= 0:
            return {
                "accepted": True,
                "arena_phase": "disabled",
                "arena_games": 0,
                "arena_win_rate": 1.0,
            }

        phase = "strict"
        games = self.config.arena.games
        simulations = self.config.arena.simulations
        accept_win_rate = self.config.arena.accept_win_rate
        min_white_win_rate = self.config.arena.min_white_win_rate
        if self.best_iteration == 0:
            phase = "bootstrap"
            games = self.config.arena.bootstrap_games
            simulations = max(self.config.arena.bootstrap_simulations, selfplay_simulations)
            accept_win_rate = self.config.arena.bootstrap_accept_win_rate
            min_white_win_rate = self.config.arena.bootstrap_min_white_win_rate

        arena = Arena(
            candidate_model=self.model,
            best_model=self.best_model,
            device=self.device,
            board_size=self.config.rules.board_size,
            exactly_five=self.config.rules.exactly_five,
            simulations=simulations,
            c_puct=self.config.selfplay.c_puct,
            leaves_per_batch=self.config.selfplay.leaves_per_batch,
        )
        result = arena.evaluate(games)
        side_games = max(1, result.games // 2)
        candidate_white_win_rate = result.candidate_white_wins / side_games
        passes_white_gate = candidate_white_win_rate >= min_white_win_rate
        accepted = result.candidate_win_rate >= accept_win_rate and passes_white_gate
        return {
            "accepted": accepted,
            "arena_phase": phase,
            "arena_games": result.games,
            "arena_simulations": simulations,
            "arena_accept_win_rate": round(accept_win_rate, 4),
            "arena_white_win_rate_threshold": round(min_white_win_rate, 4),
            "arena_passes_white_gate": passes_white_gate,
            "arena_candidate_wins": result.candidate_wins,
            "arena_best_wins": result.best_wins,
            "arena_draws": result.draws,
            "arena_candidate_black_wins": result.candidate_black_wins,
            "arena_candidate_white_wins": result.candidate_white_wins,
            "arena_candidate_white_win_rate": round(candidate_white_win_rate, 4),
            "arena_win_rate": round(result.candidate_win_rate, 4),
        }

    def save_model_checkpoints(self, metadata: dict[str, Any]) -> None:
        latest_metadata = {**metadata, "checkpoint_role": "candidate"}
        best_metadata = {
            **self.best_checkpoint_metadata,
            "checkpoint_role": "best",
            "best_iteration": self.best_iteration,
            "best_arena_win_rate": self.best_arena_win_rate,
        }
        save_checkpoint(self.checkpoint_dir / "latest.safetensors", self.model, self.config, latest_metadata)
        save_checkpoint(self.checkpoint_dir / "best.safetensors", self.best_model, self.config, best_metadata)
        save_interval = (
            1
            if self.config.checkpoint.save_every_iteration
            else max(0, self.config.checkpoint.save_iteration_interval)
        )
        if save_interval > 0 and self.iteration % save_interval == 0:
            save_checkpoint(
                self.checkpoint_dir / f"iter_{self.iteration:04d}.safetensors",
                self.model,
                self.config,
                latest_metadata,
            )

    def save_runtime_state(self, metadata: dict[str, Any]) -> None:
        payload = {
            **metadata,
            "iteration": self.iteration,
            "best_iteration": self.best_iteration,
            "best_arena_win_rate": self.best_arena_win_rate,
            "total_updates": self.total_updates,
            "elapsed_seconds": self.elapsed_seconds_offset + time.monotonic() - self.start_time,
        }
        save_trainer_state(self.checkpoint_dir, payload, self.replay.state_dict())
        save_optimizer_state(self.checkpoint_dir, self.optimizer)
        self.progress_file.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    def _restore_from_checkpoint(self, resume_path: str) -> None:
        path = Path(resume_path)
        root = path.parent if path.name == "trainer_state.json" else path
        if path.is_dir() or path.name == "trainer_state.json":
            metadata, replay_state = load_trainer_state(root)
            self.model, loaded_config, _ = load_checkpoint(root / "latest.safetensors")
            if (root / "best.safetensors").exists():
                self.best_model, _, best_metadata = load_checkpoint(root / "best.safetensors")
                self.best_checkpoint_metadata = dict(best_metadata)
            else:
                self.best_model = clone_model(self.model)
            self._validate_resume_config(loaded_config)
            self.optimizer = self._build_optimizer()
            load_optimizer_state(root, self.optimizer)
            if replay_state:
                self.replay.load_state_dict(replay_state)
            self.iteration = int(metadata.get("iteration", 0))
            self.best_iteration = int(metadata.get("best_iteration", 0))
            self.best_arena_win_rate = float(metadata.get("best_arena_win_rate", 0.0))
            self.total_updates = int(metadata.get("total_updates", 0))
            self.elapsed_seconds_offset = float(metadata.get("elapsed_seconds", 0.0))
            self.start_time = time.monotonic()
            logger.info(
                "Resumed trainer state: iteration={} best_iteration={} replay_games={}",
                self.iteration,
                self.best_iteration,
                self.replay.games_seen,
            )
            return

        self.model, loaded_config, metadata = load_checkpoint(path)
        self._validate_resume_config(loaded_config)
        self.best_model = clone_model(self.model)
        self.optimizer = self._build_optimizer()
        self.iteration = int(metadata.get("iteration", 0))
        self.total_updates = int(metadata.get("total_updates", 0))
        logger.info("Resumed model checkpoint: path={} iteration={}", path, self.iteration)

    def _validate_resume_config(self, loaded_config: RunConfig) -> None:
        if self.config.rules != loaded_config.rules:
            raise ValueError("resume config rules do not match checkpoint rules")
        if self.config.network != loaded_config.network:
            raise ValueError("resume config network does not match checkpoint network")

    def _selfplay_temperature(self, move_count: int) -> float:
        cfg = self.config.selfplay
        if cfg.temperature_moves <= 0:
            return float(cfg.temperature_end)
        if move_count >= cfg.temperature_moves:
            return float(cfg.temperature_end)
        frac = move_count / float(cfg.temperature_moves)
        return 1.0 + (float(cfg.temperature_end) - 1.0) * frac

    def _should_stop(self) -> bool:
        if self.stop_requested:
            return True
        return self.config.max_hours is not None and self.elapsed_hours >= self.config.max_hours

    def _should_start_iteration(self) -> bool:
        if self._should_stop():
            return False
        return self.config.max_iterations is None or self.iteration < self.config.max_iterations

    def _handle_stop_signal(self, signum: int, _frame: object) -> None:
        self.stop_requested = True
        logger.warning("Received stop signal {}", signum)

    def _log(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 9x9 Omok policy/value agent with tinygrad.")
    parser.add_argument("--config", type=str, default="configs/omok_quick.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    configure_logging()
    args = build_argparser().parse_args()
    config = load_config(args.config)
    if args.max_hours is not None:
        config.max_hours = args.max_hours
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.device is not None:
        config.device = args.device
    set_seed(config.seed)
    trainer = Trainer(config=config, resume_path=args.resume)
    logger.info(
        "Startup: device={} seed={} tinygrad_training={}",
        trainer.device,
        config.seed,
        Tensor.training,
    )
    trainer.run()


if __name__ == "__main__":
    main()
