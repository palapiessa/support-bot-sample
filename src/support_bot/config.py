"""Configuration helpers for the support bot sample."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BotConfig:
    """Immutable configuration for building a support bot."""

    knowledge_path: Path