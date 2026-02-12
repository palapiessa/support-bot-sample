"""Thin shim so tests can import from bot directly."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
src_root_str = str(SRC_ROOT)
if src_root_str not in sys.path:
    sys.path.insert(0, src_root_str)

from support_bot.bot import SupportBot  # noqa: E402
from support_bot.config import BotConfig  # noqa: E402

__all__ = ["SupportBot", "BotConfig"]