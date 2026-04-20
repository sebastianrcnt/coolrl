from __future__ import annotations

from typing import Literal


MctsBackend = Literal["python", "c"]


def resolve_mcts_backend(name: str):
    backend = name.strip().lower()
    if backend == "python":
        from . import mcts

        return mcts
    if backend == "c":
        from . import cmcts_wrapper

        return cmcts_wrapper
    raise ValueError(f"unsupported selfplay.mcts_backend: {name!r}")
