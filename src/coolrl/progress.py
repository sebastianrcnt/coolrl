from __future__ import annotations

import sys

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


RICH_CONSOLE = Console(stderr=True)


def make_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
        transient=False,
    )


class RichLogSink:
    def write(self, message: str) -> None:
        if message.rstrip():
            RICH_CONSOLE.print(message, end="", markup=False, highlight=False, soft_wrap=True)

    def flush(self) -> None:
        sys.stderr.flush()
