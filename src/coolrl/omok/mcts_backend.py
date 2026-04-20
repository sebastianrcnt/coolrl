from __future__ import annotations

from typing import Literal


MctsBackend = Literal["python", "c", "rust"]


def resolve_mcts_backend(name: str):
    backend = name.strip().lower()
    if backend == "python":
        from . import mcts

        return mcts
    if backend == "c":
        from . import cmcts_wrapper

        return cmcts_wrapper
    if backend == "rust":
        from . import rmcts_wrapper

        return rmcts_wrapper
    raise ValueError(f"unsupported selfplay.mcts_backend: {name!r}")
