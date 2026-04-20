from __future__ import annotations

from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "coolrl.omok._cmcts_c",
            sources=[
                "src/coolrl/omok/cmcts/src/api.c",
                "src/coolrl/omok/cmcts/src/board.c",
                "src/coolrl/omok/cmcts/src/mcts.c",
                "src/coolrl/omok/cmcts/src/tree.c",
            ],
            include_dirs=["src/coolrl/omok/cmcts/include"],
            extra_compile_args=["-O3", "-std=c11"],
        )
    ]
)
