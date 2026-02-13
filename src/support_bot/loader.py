"""Simple disk-backed loader helper for the support bot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .config import BotConfig


def load_from_disk(config: BotConfig) -> Mapping[str, str]:
    """Read the configured knowledge JSON and return the normalized map."""

    path = config.knowledge_path
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Knowledge base is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Knowledge base must be a JSON object mapping keywords to replies")

    sanitized: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All knowledge base entries must be string-to-string mappings")
        sanitized[key.strip().lower()] = value

    return sanitized