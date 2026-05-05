"""Font configuration helpers for matplotlib plots."""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager, rcParams

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_FONT_DIR = REPO_ROOT / "static" / "fonts"

_SF_PRO_PATTERNS = ("SF-Pro-Display-*.otf", "SF-Pro-Display-*.ttf")
_CJK_FAMILY_CANDIDATES = (
    "Noto Sans CJK KR",
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Source Han Sans KR",
    "Source Han Sans JP",
    "Source Han Sans SC",
    "Apple SD Gothic Neo",
    "PingFang KR",
    "PingFang SC",
    "PingFang TC",
    "Malgun Gothic",
    "NanumGothic",
    "Yu Gothic",
    "Hiragino Sans",
    "WenQuanYi Zen Hei",
)


def _register_font(path: Path, names: list[str]) -> None:
    try:
        font_manager.fontManager.addfont(str(path))
    except Exception:
        return

    try:
        family = font_manager.FontProperties(fname=str(path)).get_name()
    except Exception:
        return

    if family not in names:
        names.append(family)


def configure_fonts() -> None:
    """Register preferred plot fonts and configure rcParams.

    The function is intentionally tolerant: missing files or unsupported font
    formats are ignored, and the fallback always ends with DejaVu Sans.
    """

    families: list[str] = []

    for pattern in _SF_PRO_PATTERNS:
        for font_path in sorted(STATIC_FONT_DIR.glob(pattern)):
            _register_font(font_path, families)

    cjk_families: list[str] = []
    ttflist = list(font_manager.fontManager.ttflist)
    for candidate in _CJK_FAMILY_CANDIDATES:
        if candidate in cjk_families:
            continue
        if any(entry.name == candidate for entry in ttflist):
            cjk_families.append(candidate)

    cjk_paths = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    )
    for font_path in cjk_paths:
        if font_path.exists():
            _register_font(font_path, cjk_families)

    family_list = [*families, *cjk_families, "DejaVu Sans"]
    rcParams.update(
        {
            "font.family": family_list,
            "font.sans-serif": family_list,
            "axes.unicode_minus": False,
        }
    )
