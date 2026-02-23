"""Entrypoint for the support bot logic."""

from __future__ import annotations

from typing import Callable, Mapping

from .config import BotConfig
from .loader import load_cached_embeddings, load_question_order
from .responses import choose_response

# alias for a lightweight interface for a loader
Loader = Callable[[BotConfig], Mapping[str, str]]

class SupportBot:
    """Lightweight wrapper that wires the configuration to responders."""

    def __init__(self, config: BotConfig, loader: Loader) -> None:
        self._config = config
        self._knowledge = loader(self._config)
        self._questions = load_question_order(self._config.knowledge_path)
        self._embeddings = load_cached_embeddings(self._config.knowledge_path)

    def respond(self, question: str) -> str:
        """Return the most appropriate reply from the knowledge base."""

        return choose_response(
            self._knowledge,
            question,
            questions=self._questions,
            embeddings=self._embeddings,
        )
