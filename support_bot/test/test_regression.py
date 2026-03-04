"""Baseline regression tests for support-bot response behavior.

These tests are intentionally dataset-only and deterministic:
- no external LLM judge
- no network calls
- stable pass/fail assertions for CI
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from src.responses import SIMILARITY_THRESHOLD, choose_response


class _StubEmbeddingModel:
    """Simple embedding stub returning one predefined vector per encode call."""

    def __init__(self, vector: Sequence[float]) -> None:
        self._vector = np.asarray(vector, dtype=np.float32)

    def encode(self, _: Sequence[str], **__: object) -> np.ndarray:
        return np.asarray([self._vector], dtype=np.float32)


@pytest.fixture
def base_knowledge() -> dict[str, str]:
    return {
        "reset password": "Use the reset-password link from your account page.",
        "shipping status": "Open Orders in your profile to check shipping status.",
        "default": "I am sending this to an agent.",
    }


@pytest.mark.parametrize(
    "question, expected_reply",
    [
        ("How do I reset password?", "Use the reset-password link from your account page."),
        ("Can I see my shipping status now?", "Open Orders in your profile to check shipping status."),
        ("Tell me something unrelated", "I am sending this to an agent."),
    ],
)
def test_regression_keyword_and_fallback_paths(
    base_knowledge: dict[str, str],
    question: str,
    expected_reply: str,
) -> None:
    response = choose_response(base_knowledge, question)
    assert response == expected_reply


def test_regression_empty_question_message(base_knowledge: dict[str, str]) -> None:
    response = choose_response(base_knowledge, "   ")
    assert response == "Can you please provide more details so I can help?"


def test_regression_semantic_match_selected_when_keyword_not_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = {
        "order cancellation policy": "You can cancel before shipment in Orders.",
        "returns policy": "Unused items can be returned within 30 days.",
        "default": "I am sending this to an agent.",
    }
    questions = ["order cancellation policy", "returns policy"]
    embeddings = np.asarray(
        [
            [0.99, 0.01],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )

    def _fake_get_embedding_model() -> _StubEmbeddingModel:
        return _StubEmbeddingModel([1.0, 0.0])

    # monkeypatch replaces the get_embedding_model symbol inside responses.py for the duration of the test.
    monkeypatch.setattr("src.responses.get_embedding_model", _fake_get_embedding_model)

    response = choose_response(
        knowledge,
        "Could I cancel my order if needed?",
        questions=questions,
        embeddings=embeddings,
    )

    assert response == "You can cancel before shipment in Orders."


def test_regression_semantic_threshold_falls_back_when_score_too_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = {
        "billing issue": "Please contact billing support.",
        "default": "I am sending this to an agent.",
    }
    questions = ["billing issue"]
    below_threshold = SIMILARITY_THRESHOLD - 0.01
    embeddings = np.asarray([[below_threshold, 0.0]], dtype=np.float32)

    def _fake_get_embedding_model() -> _StubEmbeddingModel:
        return _StubEmbeddingModel([1.0, 0.0])

    monkeypatch.setattr("src.responses.get_embedding_model", _fake_get_embedding_model)

    response = choose_response(
        knowledge,
        "Need help with invoices",
        questions=questions,
        embeddings=embeddings,
    )

    assert response == "I am sending this to an agent."
