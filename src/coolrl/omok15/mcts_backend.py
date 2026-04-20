from __future__ import annotations

from typing import Literal


MctsBackend = Literal["python"]


def resolve_mcts_backend(name: str):
    backend = name.strip().lower()
    if backend == "python":
        from . import mcts

        return mcts
    raise ValueError(
        f"unsupported selfplay.mcts_backend: {name!r} (coolrl.omok15 currently supports only 'python')"
    )
