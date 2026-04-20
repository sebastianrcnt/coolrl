from __future__ import annotations

import numpy as np
from tinygrad import Tensor

from .board import GameState
from .features import states_to_feature_planes
from .network import PolicyValueNet


class Evaluator:
    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ModelEvaluator(Evaluator):
    def __init__(self, model: PolicyValueNet, device: str | None = None) -> None:
        self.model = model
        self.device = device

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        features = states_to_feature_planes(states)
        return self.evaluate_features(features)

    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = Tensor(np.ascontiguousarray(features), device=self.device)
        with Tensor.train(False):
            logits, values = self.model(tensor)
            priors = logits.softmax(axis=1).realize().numpy()
            value_np = values.realize().numpy()
        return priors.astype(np.float32, copy=False), value_np.astype(np.float32, copy=False)

