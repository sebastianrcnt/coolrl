from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..bots.base import first_legal
from ..evaluation import (
    SUPPORTED_OPPONENTS,
    NoisySafeHeuristicBot,
    evaluate_agent_against_bot,
    is_card_discard_action,
    is_card_play_action,
    is_draw_deck_action,
    make_opponent,
    play_game_for_evaluation,
)
from ..game import GameState, LostCitiesConfig
from ..interfaces import BotInput, LostCitiesBot
from .config import EncodingConfig, config_from_dict
from .encoding import encode_information_state, legal_mask_array
from .networks import StrategyNet

_play_game_for_evaluation = play_game_for_evaluation

__all__ = [
    "SUPPORTED_OPPONENTS",
    "NoisySafeHeuristicBot",
    "StrategyNetBot",
    "evaluate_against_bot",
    "is_card_discard_action",
    "is_card_play_action",
    "is_draw_deck_action",
    "load_strategy_bot_from_checkpoint",
    "make_opponent",
]


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
        encoding: EncodingConfig | None = None,
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        self.strategy_net = strategy_net
        self.config = config
        self.device = torch.device(device)
        self.encoding = encoding or EncodingConfig()
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.last_policy_entropy: float | None = None

    def act(self, obs_or_state: BotInput) -> int:
        if not isinstance(obs_or_state, GameState):
            self.last_policy_entropy = None
            if isinstance(obs_or_state, dict):
                legal = np.asarray(obs_or_state["legal_mask"], dtype=bool)
            else:
                legal = np.asarray(obs_or_state.legal_mask, dtype=bool)
            return first_legal(legal)
        state = obs_or_state
        info = encode_information_state(state, state.current_player, self.encoding)
        legal = legal_mask_array(state)
        legal_indices = np.flatnonzero(legal)
        if len(legal_indices) == 0:
            raise RuntimeError("no legal action available")
        with torch.inference_mode():
            x = torch.as_tensor(info, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_net(x).squeeze(0).detach().cpu().numpy()
        masked = np.where(legal, logits, -np.inf)
        stable = masked[legal_indices] - np.max(masked[legal_indices])
        probs = np.exp(stable)
        probs = probs / probs.sum()
        if self.sample:
            unified = int(self.rng.choice(legal_indices, p=probs))
        else:
            unified = int(np.argmax(masked))
        self.last_policy_entropy = float(-(probs * np.log(np.clip(probs, 1.0e-12, 1.0))).sum())
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
            encoding=config.encoding,
            sample=sample,
            seed=seed,
        ),
        lc_config,
    )


def evaluate_against_bot(
    strategy_net: StrategyNet,
    opponent_bot: LostCitiesBot,
    config: LostCitiesConfig,
    games: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
    max_steps: int = 10_000,
    on_max_steps: str = "score_diff",
    sample: bool = False,
    encoding: EncodingConfig | None = None,
) -> dict[str, float | int]:
    strategy_net.eval()

    def make_strategy_bot(index: int) -> StrategyNetBot:
        return StrategyNetBot(
            strategy_net,
            config,
            device=device,
            encoding=encoding,
            sample=sample,
            seed=seed + index,
        )

    return evaluate_agent_against_bot(
        make_strategy_bot,
        opponent_bot,
        config,
        games,
        seed,
        max_steps=max_steps,
        on_max_steps=on_max_steps,
    )
