"""Unit tests for the support bot sample."""

import json
from pathlib import Path

from bot import BotConfig, SupportBot
from support_bot.answer_retriever import get_answer
from test.loader import load_from_disk


def _write_knowledge(path: Path, overlay: dict[str, str]) -> None:
    (path.parent).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlay), encoding="utf-8")

#
# test_keyword_response writes a temporary JSON knowledge base containing a keyword reply ("reset password"),
# adds the default fallback entry, and asserts that asking "How do I reset my password?" returns the keyword reply so
# the keyword matching path in respond is exercised.
#
def test_keyword_response(tmp_path: Path) -> None:
    """Keyword lookup matches the corresponding reply when present."""

    knowledge = {
        "reset password": "To reset, click the link and follow the prompts.",
        "default": "I am sending this to an agent."
    }
    data_path = tmp_path / "knowledge_base.json"
    _write_knowledge(data_path, knowledge)

    bot = SupportBot(BotConfig(knowledge_path=data_path), loader=load_from_disk)

    response = bot.respond("How do I reset my password?")
    assert "reset" in response.lower()

#
# test_fallback_response ensures the default reply is returned when only that entry exists
# and no keyword matches.
#
def test_fallback_response(tmp_path: Path) -> None:
    """When only a default reply exists, unexpected questions trigger it."""

    knowledge = {"default": "Please hold while I connect you."}
    data_path = tmp_path / "knowledge_base.json"
    _write_knowledge(data_path, knowledge)

    bot = SupportBot(BotConfig(knowledge_path=data_path), loader=load_from_disk)
    response = bot.respond("Tell me something unexpected.")
    assert response == knowledge["default"]

#
# The _StubVectorizer class emulates what a real embedding model returns: it stores the
# float vectors supplied via the constructor and returns them directly from fit_transform/transform. The stub
# ignores the string documents so the tests control the embeddings with predefined float lists instead of
# running actual tokenizers or encoders.
class _StubVectorizer:
    """Minimal stand-in that returns the embeddings supplied by the test."""

    def __init__(self, knowledge_embeddings: list[list[float]], question_embedding: list[float]) -> None:
        self._knowledge_embeddings = knowledge_embeddings
        self._question_embedding = question_embedding

    def fit_transform(self, documents: list[str]) -> list[list[float]]:  # pragma: no cover - simple stub
        return self._knowledge_embeddings

    def transform(self, documents: list[str]) -> list[list[float]]:  # pragma: no cover - simple stub
        return [self._question_embedding for _ in documents]


# test_get_answer_prefers_highest_similarity() verifies that get_answer selects the FAQ entry whose embedding is most similar to the question.
# It stubs out a vectorizer that returns two orthogonal knowledge embeddings and a question embedding matching only the first entry,
# then asserts that the returned answer equals the first FAQ’s reply, confirming the helper ranks by cosine similarity.
def test_get_answer_prefers_highest_similarity() -> None:
    faq_pairs = [
        ("reset password", "Click the Reset link."),
        ("billing question", "Contact billing."),
    ]
    vectorizer = _StubVectorizer(
        knowledge_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        question_embedding=[1.0, 0.0],
    )

    answer = get_answer(
        question="How do I reset my password?",
        faq_pairs=faq_pairs,
        vectorizer=vectorizer,
        fallback="I cannot help",
    )

    assert answer == faq_pairs[0][1]

# test_get_answer_returns_fallback_when_similarity_is_low() proves that get_answer falls back when no FAQ entry is sufficiently similar to the user question.
# It supplies one FAQ pair and a stub vectorizer where the question embedding is a zero vector and then calls get_answer
# with a high similarity_threshold=0.9. Because the computed similarity (cosine of 0) never reaches the cutoff, the helper
# returns the supplied fallback text, which the test asserts.
def test_get_answer_returns_fallback_when_similarity_is_low() -> None:
    faq_pairs = [("reset password", "Click the Reset link.")]
    vectorizer = _StubVectorizer(
        knowledge_embeddings=[[1.0, 0.0]],
        question_embedding=[0.0, 0.0],
    )

    answer = get_answer(
        question="What is this?",
        faq_pairs=faq_pairs,
        vectorizer=vectorizer,
        fallback="fallback response",
        similarity_threshold=0.9,
    )

    assert answer == "fallback response"