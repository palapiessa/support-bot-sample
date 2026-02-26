from pathlib import Path
import os
from typing import Sequence

import numpy as np
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient


load_dotenv(find_dotenv())

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_MODEL = "allenai/t5-small-squad2-question-generation"
project_root = Path(__file__).resolve().parents[3]


def _get_embeddings_path() -> Path:
    embeddings_path_value = os.getenv(
        "EXISTING_QUESTION_EMBEDDINGS_NPY", "data/support_bot_knowledge.embeddings.npy"
    )
    embeddings_path = Path(embeddings_path_value)
    if not embeddings_path.is_absolute():
        embeddings_path = project_root / embeddings_path
    return embeddings_path


def load_existing_embeddings() -> np.ndarray:
    embeddings_path = _get_embeddings_path()
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError("Expected embeddings file to contain a 2D array")
    return embeddings.astype(np.float32)


def _get_embedding(client: InferenceClient, text: str) -> np.ndarray:
    embedding = client.feature_extraction(model=EMBEDDING_MODEL, inputs=text)
    return np.asarray(embedding, dtype=np.float32)


def _clean_seed_questions(context_input: Sequence[str]) -> list[str]:
    cleaned_items = [item.strip() for item in context_input if item and item.strip()]
    if not cleaned_items:
        raise ValueError("context_input list is empty")
    return cleaned_items


def _build_generation_prompt(
    context_input: str | Sequence[str],
    focus_seed_question: str | None = None,
) -> str:
    if isinstance(context_input, str):
        context_text = context_input.strip()
    else:
        cleaned_items = _clean_seed_questions(context_input)
        numbered_examples = "\n".join(
            f"{index}. {question}" for index, question in enumerate(cleaned_items, start=1)
        )
        focus_line = (
            f"Prioritize the topic style of this seed question: {focus_seed_question}\n"
            if focus_seed_question
            else ""
        )
        context_text = (
            "Seed FAQ questions (topic examples):\n"
            f"{numbered_examples}\n"
            f"{focus_line}"
            "Infer the topics from these examples and generate one new FAQ question covering those topics. "
            "Keep the style similar, avoid copying wording, and return only the question."
        )

    if not context_text:
        raise ValueError("context_input is empty")

    return f"Generate one FAQ question from this context: {context_text}"


def is_duplicate_new_question(
    client: InferenceClient,
    new_question: str,
    existing_embeddings: np.ndarray,
    threshold: float = 0.85,
) -> bool:
    new_embedding = _get_embedding(client, new_question)

    denominator = np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(new_embedding)
    denominator = np.where(denominator == 0, 1e-12, denominator)
    similarity_scores = np.dot(existing_embeddings, new_embedding) / denominator
    return bool(np.any(similarity_scores > threshold))


def generate_questions(
    context_input: str | Sequence[str],
    num_questions: int = 5,
    threshold: float = 0.85,
    max_attempts: int | None = None,
    cover_seed_topics: bool = True,
) -> list[str]:
    if num_questions < 1:
        return []

    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    existing_embeddings = load_existing_embeddings()
    accepted_questions: list[str] = []
    attempts = 0
    max_attempts = max_attempts or (num_questions * 5)
    seed_questions = (
        _clean_seed_questions(context_input)
        if not isinstance(context_input, str)
        else None
    )

    while len(accepted_questions) < num_questions and attempts < max_attempts:
        attempts += 1
        focus_seed_question = None
        if seed_questions and cover_seed_topics:
            focus_seed_question = seed_questions[(attempts - 1) % len(seed_questions)]

        prompt = _build_generation_prompt(context_input, focus_seed_question)
        response = client.text_generation(
            model=QUESTION_MODEL,
            inputs=prompt,
            max_new_tokens=100,
        )
        candidate = str(response).strip().splitlines()[0].strip()
        if not candidate:
            continue

        if not candidate.endswith("?"):
            candidate = f"{candidate}?"

        if is_duplicate_new_question(client, candidate, existing_embeddings, threshold):
            continue

        accepted_questions.append(candidate)
        candidate_embedding = _get_embedding(client, candidate)
        existing_embeddings = np.vstack([existing_embeddings, candidate_embedding])

    return accepted_questions


# Example call:
# faqs = generate_questions(seed_questions, num_questions=300, cover_seed_topics=True)
# seed_questions = [
#     "Do I have to pay postal costs when returning a purchase?",
#     "How to return an unused product?",
#     "Do the clothes have any warranty?",
#     "Damaged package - outside/inside",
#     "Clothes - how to change size of the product",
#     "Can I get money back if I am not satisfied with the product and return it?",
#     "How long does the shopping cart items remain valid? Can I continue shopping another day?",
#     "How can I contact a support person?",
# ]
