"""Shared plotting helpers for CoolRL."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .theme import DARK, LIGHT, THEMES, Theme

__all__ = [
    "Theme",
    "DARK",
    "LIGHT",
    "THEMES",
    "configure_fonts",
    "gradient_fill",
    "gradient_line",
    "moving_average_smooth",
    "panel_title",
    "style_axis",
]

_LAZY_ATTRS = {
    "configure_fonts": (".fonts", "configure_fonts"),
    "gradient_fill": (".primitives", "gradient_fill"),
    "gradient_line": (".primitives", "gradient_line"),
    "moving_average_smooth": (".primitives", "moving_average_smooth"),
    "panel_title": (".primitives", "panel_title"),
    "style_axis": (".primitives", "style_axis"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ATTRS})


if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fonts import configure_fonts
    from .primitives import (
        gradient_fill,
        gradient_line,
        moving_average_smooth,
        panel_title,
        style_axis,
    )
