"""Helpers for loading knowledge from Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, Tuple

from datasets import load_dataset
from .config import BotConfig

def load_faq_pairs(
    *,
    config: BotConfig,
) -> list[Tuple[str, str]]:
    """Return normalized (question, answer) pairs from a Hugging Face dataset."""

    effective_dataset = config.dataset_name
    effective_split = config.dataset_split
    effective_question_field = config.question_field
    effective_answer_field = config.answer_field

    try:
        dataset = load_dataset(
            "json",
            data_files="https://huggingface.co/palapiessa/e_commerce_customer_service_squad/resolve/main/ecommerce_faq_as_squad.json",
            field="data",
            split=effective_split,
        )
    except Exception as exc:  # pragma: no cover - dataset errors are external
        raise RuntimeError(f"Unable to load dataset {effective_dataset}:{effective_split}") from exc

    faqs: list[Tuple[str, str]] = []

    for item in _iter_question_answer_rows(dataset):
        question = _coerce_to_str(item.get(effective_question_field))
        answer = _coerce_to_str(item.get(effective_answer_field))
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


def _iter_question_answer_rows(
    dataset: Iterable[Mapping[str, object]],
) -> Iterator[Mapping[str, object]]:
    for row in dataset:
        if not isinstance(row, Mapping):
            continue
        paragraphs = row.get("paragraphs")
        if _is_non_string_sequence(paragraphs):
            for paragraph in paragraphs:  # type: ignore[arg-type]
                if not isinstance(paragraph, Mapping):
                    continue
                qas = paragraph.get("qas")
                if _is_non_string_sequence(qas):
                    for qa in qas:  # type: ignore[arg-type]
                        if isinstance(qa, Mapping):
                            yield qa
            continue
        yield row


def _is_non_string_sequence(value: object | None) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

# Takes whatever value the Hugging Face dataset returned for a field (it might already be a string, a list of strings, or None)
# and normalizes it into a single trimmed string or None. Strings get stripped and empty results become None; iterables get
# joined with spaces; other values are dropped. This keeps load_faq_pairs from adding blank or malformed entries when it builds
# the (question, answer) tuples.
def _coerce_to_str(value: Sequence[str] | str | Mapping[str, object] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    if isinstance(value, Mapping):
        text_value = value.get("text") or value.get("answer")
        if text_value is not None:
            return _coerce_to_str(text_value)
        return None
    if isinstance(value, Iterable):
        flattened_parts: list[str] = []
        for part in value:
            part_text = _coerce_to_str(part)  # type: ignore[arg-type]
            if part_text:
                flattened_parts.append(part_text)
        flattened = " ".join(flattened_parts)
        return flattened or None
    return None