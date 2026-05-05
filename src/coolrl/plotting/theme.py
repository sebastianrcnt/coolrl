"""Color themes shared by plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    bg: str
    panel_bg: str
    grid: str
    text_primary: str
    text_secondary: str
    text_tertiary: str
    accents: dict[str, tuple[str, str]]
    line_alpha: float = 1.0


DARK = Theme(
    name="dark",
    bg="#000000",
    panel_bg="#0e0e10",
    grid="#1c1c1e",
    text_primary="#f5f5f7",
    text_secondary="#a1a1a6",
    text_tertiary="#6e6e73",
    accents={
        "policy": ("#ff3b30", "#ff6961"),
        "value": ("#ff3b30", "#ff6961"),
        "total": ("#ff9500", "#ffb347"),
        "lr": ("#ff9500", "#ffb347"),
        "winrate": ("#ffcc00", "#ffe066"),
        "white": ("#34c759", "#5ce07c"),
        "moves": ("#34c759", "#5ce07c"),
        "buffer": ("#0a84ff", "#64d2ff"),
        "elapsed": ("#0a84ff", "#64d2ff"),
    },
)

LIGHT = Theme(
    name="light",
    bg="#ffffff",
    panel_bg="#f5f5f7",
    grid="#d2d2d7",
    text_primary="#1d1d1f",
    text_secondary="#424245",
    text_tertiary="#6e6e73",
    accents={
        "policy": ("#d70015", "#ff3b30"),
        "value": ("#d70015", "#ff3b30"),
        "total": ("#c93400", "#ff9500"),
        "lr": ("#c93400", "#ff9500"),
        "winrate": ("#946200", "#d2a200"),
        "white": ("#248a3d", "#34c759"),
        "moves": ("#248a3d", "#34c759"),
        "buffer": ("#0040dd", "#007aff"),
        "elapsed": ("#0040dd", "#007aff"),
    },
    line_alpha=0.85,
)

THEMES: dict[str, Theme] = {"dark": DARK, "light": LIGHT}
