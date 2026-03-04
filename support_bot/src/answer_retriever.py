"""Vector-based answer retrieval helpers for the support bot sample."""

from __future__ import annotations

import math
from typing import Protocol, Sequence

DEFAULT_FALLBACK = "I am still learning and cannot answer that right now."

# stays separate from SupportBot’s keyword logic so load_knowledge() (used by SupportBot) 
# can keep returning the JSON-derived Mapping[str, str], 
# and then answer_retriever.get_answer() can be the specialized helper that:

# Takes faq_pairs (your normalized question/answer tuples) and a Vectorizer implementation,
# Builds embeddings via fit_transform/transform,
# Measures cosine similarity against the question vector,
# Returns the best match or a fallback when nothing clears the similarity_threshold.
# That separation means the knowledge loader just hands structured data downstream, and you can plug in this retriever whenever you want semantic search without changing SupportBot’s existing behavior
class Vectorizer(Protocol):
    """Protocol that exposes the minimal vectorizer surface used in tests."""

    def fit_transform(self, documents: Sequence[str]) -> list[list[float]]:
        ...

    def transform(self, documents: Sequence[str]) -> list[list[float]]:
        ...


def get_answer(
    question: str,
    faq_pairs: Sequence[tuple[str, str]],
    vectorizer: Vectorizer,
    *,
    fallback: str | None = None,
    similarity_threshold: float = 0.5,
) -> str:
    """Return the best FAQ answer whose vectorization matches the question."""

    fallback_text = fallback or DEFAULT_FALLBACK
    if not faq_pairs:
        return fallback_text

    normalized_question = question.strip()
    if not normalized_question:
        return fallback_text

    question_texts = [pair[0] for pair in faq_pairs]
    knowledge_embeddings = vectorizer.fit_transform(question_texts)
    if len(knowledge_embeddings) != len(question_texts):
        raise ValueError("Vectorizer must return one embedding per FAQ entry")

    question_embeddings = vectorizer.transform([normalized_question])
    if not question_embeddings:
        return fallback_text

    question_embedding = question_embeddings[0]
    best_idx = -1
    best_score = -1.0
    for idx, knowledge_embedding in enumerate(knowledge_embeddings):
        score = _cosine_similarity(question_embedding, knowledge_embedding)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx == -1 or best_score < similarity_threshold:
        return fallback_text

    return faq_pairs[best_idx][1]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors, guarding against zero length."""

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)