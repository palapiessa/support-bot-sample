"""Response selection helpers for the support bot."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .loader import get_embedding_model

SIMILARITY_THRESHOLD = 0.75


def choose_response(
    knowledge: Mapping[str, str],
    question: str,
    *,
    questions: Sequence[str] | None = None,
    embeddings: np.ndarray | None = None,
) -> str:
    """Pick a reply from the knowledge base using keyword and semantic matches."""

    normalized = question.strip().lower()
    if not normalized:
        return "Can you please provide more details so I can help?"

    fallback = knowledge.get("default")
    for keyword, reply in knowledge.items():
        if keyword == "default":
            continue
        keyword_terms = [term for term in keyword.split() if term]
        if keyword_terms and all(term in normalized for term in keyword_terms):
            return reply

    semantic_reply = _semantic_match(normalized, knowledge, questions, embeddings)
    if semantic_reply:
        return semantic_reply

    if fallback:
        return fallback

    return "I am still learning and cannot answer that right now."


def _semantic_match(
    question: str,
    knowledge: Mapping[str, str],
    questions: Sequence[str] | None,
    embeddings: np.ndarray | None,
) -> str | None:
    if not questions or embeddings is None or not embeddings.size:
        return None

    model = get_embedding_model()
    user_embedding = model.encode(
        [question],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    user_vector = np.asarray(user_embedding[0])
    scores = embeddings @ user_vector
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    if best_score >= SIMILARITY_THRESHOLD:
        matched_question = questions[best_index]
        return knowledge.get(matched_question)
    return None