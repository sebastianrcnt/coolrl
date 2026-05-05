from __future__ import annotations

import numpy as np
from matplotlib import rcParams

import coolrl.plotting as plotting


def test_plotting_package_exports_theme() -> None:
    assert plotting.DARK.bg == "#000000"
    assert plotting.LIGHT.name == "light"


def test_moving_average_smooth_preserves_nan_positions() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=float)
    smoothed = plotting.moving_average_smooth(values, window=3)

    assert smoothed.shape == values.shape
    assert np.isnan(smoothed[2])
    assert smoothed[0] != values[0]


def test_configure_fonts_is_tolerant() -> None:
    plotting.configure_fonts()

    family = rcParams["font.family"]
    assert isinstance(family, list)
    assert "DejaVu Sans" in family
