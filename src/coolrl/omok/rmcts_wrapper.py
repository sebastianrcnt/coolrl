from __future__ import annotations

import warnings

from .mcts import MCTS as _PythonMCTS
from .mcts import TreeNode, sample_action_from_policy


class MCTS(_PythonMCTS):
    """Rust backend shim.

    This backend keeps the public Python/C backend interface while we iterate on the
    native Rust implementation in `src/coolrl/omok/rmcts/`.
    """

    _warned = False

    def __init__(self, *args, **kwargs):
        if not MCTS._warned:
            warnings.warn(
                "selfplay.mcts_backend='rust' currently uses the Python runtime shim; "
                "the native Rust engine is available under src/coolrl/omok/rmcts/.",
                RuntimeWarning,
                stacklevel=2,
            )
            MCTS._warned = True
        super().__init__(*args, **kwargs)
