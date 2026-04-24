from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..bots.base import first_legal
from ..bots.heuristic import SafeHeuristicBot
from ..bots.play import play_game
from ..bots.random import RandomBot
from ..game import GameState, LostCitiesConfig
from ..interfaces import BotInput, LostCitiesBot
from .config import config_from_dict
from .encoding import encode_information_state, legal_mask_array
from .networks import StrategyNet


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    token = str(device).strip().lower()
    if token == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if token == "cpu":
        return torch.device("cpu")
    if token == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is unavailable")
        return torch.device("cuda")
    raise ValueError(f"unsupported device: {device!r}")


class StrategyNetBot(LostCitiesBot):
    def __init__(
        self,
        strategy_net: StrategyNet,
        config: LostCitiesConfig,
        *,
        device: torch.device | str = "cpu",
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        self.strategy_net = strategy_net
        self.config = config
        self.device = torch.device(device)
        self.sample = sample
        self.rng = np.random.default_rng(seed)

    def act(self, obs_or_state: BotInput) -> int:
        if not isinstance(obs_or_state, GameState):
            legal = np.asarray(obs_or_state["legal_mask"], dtype=bool) if isinstance(obs_or_state, dict) else np.asarray(obs_or_state.legal_mask, dtype=bool)
            return first_legal(legal)
        state = obs_or_state
        info = encode_information_state(state, state.current_player)
        legal = legal_mask_array(state)
        legal_indices = np.flatnonzero(legal)
        if len(legal_indices) == 0:
            raise RuntimeError("no legal action available")
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_net(x).squeeze(0).detach().cpu().numpy()
        masked = np.where(legal, logits, -np.inf)
        if self.sample:
            stable = masked[legal_indices] - np.max(masked[legal_indices])
            probs = np.exp(stable)
            probs = probs / probs.sum()
            unified = int(self.rng.choice(legal_indices, p=probs))
        else:
            unified = int(np.argmax(masked))
        return state.from_unified_action(unified)


def load_strategy_bot_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    sample: bool = False,
    seed: int | None = None,
) -> tuple[StrategyNetBot, LostCitiesConfig]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = config_from_dict(payload["config"])
    resolved_device = _resolve_device(device)
    strategy_net = StrategyNet(
        int(payload["input_dim"]),
        int(payload["action_size"]),
        config.network,
    ).to(resolved_device)
    strategy_net.load_state_dict(payload["strategy_net"])
    strategy_net.eval()
    lc_config = config.rules.to_lost_cities_config(seed=config.seed)
    return (
        StrategyNetBot(
            strategy_net,
            lc_config,
            device=resolved_device,
            sample=sample,
            seed=seed,
        ),
        lc_config,
    )


def make_opponent(name: str, seed: int | None = None) -> LostCitiesBot:
    token = name.lower()
    if token == "random":
        return RandomBot(seed=seed)
    if token == "safe_heuristic":
        return SafeHeuristicBot()
    raise ValueError(f"unsupported opponent: {name!r}")


def evaluate_against_bot(
    strategy_net: StrategyNet,
    opponent_bot: LostCitiesBot,
    config: LostCitiesConfig,
    games: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float | int]:
    strategy_net.eval()
    diffs: list[int] = []
    wins = losses = draws = 0
    for index in range(games):
        net_bot = StrategyNetBot(strategy_net, config, device=device, sample=False, seed=seed + index)
        game_seed = seed + index
        if index % 2 == 0:
            final_state = play_game(net_bot, opponent_bot, replace(config, seed=game_seed))
            diff = final_state.score_diff(0)
        else:
            final_state = play_game(opponent_bot, net_bot, replace(config, seed=game_seed))
            diff = final_state.score_diff(1)
        diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1
    return {
        "games": int(games),
        "win_rate": float(wins / max(1, games)),
        "avg_diff": float(np.mean(diffs)) if diffs else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }
