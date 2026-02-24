"""Simple disk-backed loader helper for the support bot."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import BotConfig
from .data_loader import load_faq_pairs



def load_from_disk(config: BotConfig) -> Mapping[str, str]:
    """Read the configured knowledge JSON and return the normalized map."""

    path = config.knowledge_path
    if not path.exists():
        load_faq_pairs(config=config)
        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found after refreshing dataset: {path}"
            )

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

    _persist_question_list(list(sanitized.keys()), path)
    _ensure_question_embeddings(list(sanitized.keys()), path)
    return sanitized


def _ensure_question_embeddings(questions: Sequence[str], knowledge_path: Path) -> None:
    if not questions:
        return

    cache_path = _embedding_cache_path(knowledge_path)
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            if cached.shape[0] == len(questions):
                return
        except (OSError, ValueError):
            pass

    embeddings = _embedding_model().encode(
        questions,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings, allow_pickle=False)


def load_question_order(knowledge_path: Path) -> Sequence[str]:
    return tuple(_load_question_list(knowledge_path))


def load_cached_embeddings(knowledge_path: Path) -> np.ndarray | None:
    cache_path = _embedding_cache_path(knowledge_path)
    if not cache_path.exists():
        return None
    try:
        return np.load(cache_path)
    except (OSError, ValueError):
        return None


def get_embedding_model() -> SentenceTransformer:
    return _embedding_model()


def _embedding_cache_path(knowledge_path: Path) -> Path:
    return knowledge_path.with_name(f"{knowledge_path.stem}.embeddings.npy")


def _question_list_path(knowledge_path: Path) -> Path:
    return knowledge_path.with_name(f"{knowledge_path.stem}.questions.json")


def _persist_question_list(questions: Iterable[str], knowledge_path: Path) -> None:
    cache_path = _question_list_path(knowledge_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(list(questions), ensure_ascii=False), encoding="utf-8")


def _load_question_list(knowledge_path: Path) -> list[str]:
    cache_path = _question_list_path(knowledge_path)
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
            return payload
    except json.JSONDecodeError:
        pass
    return []


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )