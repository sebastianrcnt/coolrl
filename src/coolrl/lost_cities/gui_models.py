from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .bots import RandomBot
from .pygame_common import Snapshot

ModelName = str
DEFAULT_MODEL: ModelName = "random"


class GuiModel(Protocol):
    name: str

    def act(self, snapshot: Snapshot) -> int:
        """Return a unified action id for the current snapshot."""


class RandomActionModel:
    name = DEFAULT_MODEL

    def __init__(self, seed: int | None = None):
        self.bot = RandomBot(seed=seed)

    def act(self, snapshot: Snapshot) -> int:
        return self.bot.act({"legal_mask": snapshot.legal_mask})


ModelFactory = Callable[[int | None], GuiModel]

MODEL_REGISTRY: dict[ModelName, ModelFactory] = {
    RandomActionModel.name: RandomActionModel,
}


def available_model_names() -> list[ModelName]:
    return sorted(MODEL_REGISTRY)


def build_model(name: ModelName, *, seed: int | None = None) -> GuiModel:
    try:
        model_factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown Lost Cities GUI model: {name}") from exc
    return model_factory(seed)
