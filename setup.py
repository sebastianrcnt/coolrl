from __future__ import annotations

import os

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - build environments must provide Cython
    cythonize = None


compile_args = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Wpedantic"]
link_args = []
if os.name != "nt":
    compile_args.append("-pthread")
    link_args.append("-pthread")
if os.environ.get("COOLRL_CMCTS_ASAN") == "1":
    compile_args.extend(["-fsanitize=address", "-fno-omit-frame-pointer"])
    link_args.append("-fsanitize=address")


cmcts_extension = Extension(
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
)


cython_extension_specs = [
    ("coolrl.lost_cities.game", "src/coolrl/lost_cities/game.pyx"),
]

cython_extensions: list[Extension] = []
if cythonize is not None:
    raw_extensions = [
        Extension(name, [path])
        for name, path in cython_extension_specs
        if os.path.exists(path)
    ]
    if raw_extensions:
        cython_extensions = cythonize(
            raw_extensions,
            language_level=3,
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "initializedcheck": False,
            },
        )


setup(
    ext_modules=[cmcts_extension, *cython_extensions],
)
