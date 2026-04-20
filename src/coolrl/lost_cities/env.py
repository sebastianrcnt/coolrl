from __future__ import annotations

from .game import GameState, LostCitiesConfig

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for LostCitiesEnv") from exc


class LostCitiesEnv:
    def __init__(self, config: LostCitiesConfig | None = None):
        self.config = config or LostCitiesConfig()
        self.state: GameState | None = None

    def reset(self) -> dict:
        self.state = GameState.new_game(self.config)
        return self._obs()

    def step(self, action_id: int) -> tuple[dict, float, bool, dict]:
        if self.state is None:
            self.reset()
        assert self.state is not None
        self.state.apply_action(action_id)
        done = self.state.terminal
        reward = float(self.state.score_diff(0)) if done else 0.0
        return self._obs(), reward, done, {}

    def legal_actions(self) -> np.ndarray:
        if self.state is None:
            self.reset()
        assert self.state is not None
        return np.asarray(self.state.legal_mask(), dtype=bool)

    @property
    def current_player(self) -> int:
        if self.state is None:
            self.reset()
        assert self.state is not None
        return self.state.current_player

    @property
    def phase(self) -> str:
        if self.state is None:
            self.reset()
        assert self.state is not None
        return self.state.phase

    def _obs(self) -> dict:
        assert self.state is not None
        return {
            "spatial": np.zeros((0,), dtype=np.float32),
            "scalar": np.zeros((0,), dtype=np.float32),
            "legal_mask": np.asarray(self.state.legal_mask(), dtype=bool),
            "phase": 0 if self.state.phase == "card" else 1,
            "player": self.state.current_player,
        }
