"""Entrypoint for the support bot logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .config import BotConfig
from .responses import choose_response


class SupportBot:
    """Lightweight wrapper that wires the configuration to responders."""

    def __init__(self, config: BotConfig) -> None:
        self._config = config
        self._knowledge = self._load_knowledge()

    def respond(self, question: str) -> str:
        """Return the most appropriate reply from the knowledge base."""

        return choose_response(self._knowledge, question)

    def _load_knowledge(self) -> Mapping[str, str]:
        path = self._config.knowledge_path
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