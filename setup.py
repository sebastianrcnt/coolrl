from __future__ import annotations

import os

from setuptools import Extension, setup


compile_args = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Wpedantic"]
link_args = []
if os.name != "nt":
    compile_args.append("-pthread")
    link_args.append("-pthread")
if os.environ.get("COOLRL_CMCTS_ASAN") == "1":
    compile_args.extend(["-fsanitize=address", "-fno-omit-frame-pointer"])
    link_args.append("-fsanitize=address")


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
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            "coolrl.omok15._cmcts_c",
            sources=[
                "src/coolrl/omok15/cmcts/src/api.c",
                "src/coolrl/omok15/cmcts/src/board.c",
                "src/coolrl/omok15/cmcts/src/mcts.c",
                "src/coolrl/omok15/cmcts/src/tree.c",
            ],
            include_dirs=["src/coolrl/omok15/cmcts/include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ]
)
