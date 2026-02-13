"""Helpers for loading knowledge from Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from datasets import load_dataset
from .config import BotConfig


def load_faq_pairs(
    dataset_name: str = "qgyd2021/e_commerce_customer_service",
    *,
    split: str = "train",
    question_field: str = "question",
    answer_field: str = "answer",
    config: BotConfig | None = None,
) -> list[Tuple[str, str]]:
    """Return normalized (question, answer) pairs from a Hugging Face dataset."""

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as exc:  # pragma: no cover - dataset errors are external
        raise RuntimeError(f"Unable to load dataset {dataset_name}:{split}") from exc

    faqs: list[Tuple[str, str]] = []

    for item in dataset:
        question = _coerce_to_str(item.get(question_field))
        answer = _coerce_to_str(item.get(answer_field))
        if question and answer:
            faqs.append((question, answer))

    if config:
        persist_faq_pairs(faqs, config=config)
    return faqs


def persist_faq_pairs(
    faq_pairs: Sequence[tuple[str, str]],
    *,
    knowledge_path: Path | None = None,
    config: BotConfig | None = None,
) -> None:
    """Write the normalized FAQ pairs out as JSON so SupportBot can load them."""

    resolved_path = knowledge_path or (config and config.knowledge_path)
    if resolved_path is None:
        raise ValueError("knowledge_path or config with a knowledge_path must be provided")

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge: dict[str, str] = {}
    for question, answer in faq_pairs:
        normalized_question = question.strip().lower()
        knowledge[normalized_question] = answer

    try:
        resolved_path.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unable to persist knowledge to {resolved_path}") from exc

# Takes whatever value the Hugging Face dataset returned for a field (it might already be a string, a list of strings, or None)
# and normalizes it into a single trimmed string or None. Strings get stripped and empty results become None; iterables get
# joined with spaces; other values are dropped. This keeps load_faq_pairs from adding blank or malformed entries when it builds
# the (question, answer) tuples.
def _coerce_to_str(value: Sequence[str] | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    if isinstance(value, Iterable):
        flattened = " ".join(str(part).strip() for part in value if part)
        return flattened or None
    return None