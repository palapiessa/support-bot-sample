from pathlib import Path
import argparse
import json
import os
from typing import Callable, Sequence

import numpy as np
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


load_dotenv(find_dotenv())

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUESTION_MODEL = os.getenv("FAQ_QUESTION_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
ANSWER_MODEL = os.getenv("FAQ_ANSWER_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
project_root = Path(__file__).resolve().parents[3]


def _get_embeddings_path() -> Path:
    embeddings_path_value = os.getenv(
        "EXISTING_QUESTION_EMBEDDINGS_NPY", "data/support_bot_knowledge.embeddings.npy"
    )
    embeddings_path = Path(embeddings_path_value)
    if not embeddings_path.is_absolute():
        embeddings_path = project_root / embeddings_path
    return embeddings_path


def _get_knowledge_path() -> Path:
    knowledge_path_value = os.getenv("KNOWLEDGE_JSON_PATH", "data/support_bot_knowledge.json")
    knowledge_path = Path(knowledge_path_value)
    if not knowledge_path.is_absolute():
        knowledge_path = project_root / knowledge_path
    return knowledge_path


def _get_question_list_path() -> Path:
    knowledge_path = _get_knowledge_path()
    return knowledge_path.with_name(f"{knowledge_path.stem}.questions.json")


def load_existing_embeddings() -> np.ndarray:
    embeddings_path = _get_embeddings_path()
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError("Expected embeddings file to contain a 2D array")
    return embeddings.astype(np.float32)


def _get_embedding(client: InferenceClient, text: str) -> np.ndarray:
    try:
        embedding = client.feature_extraction(text=text, model=EMBEDDING_MODEL)
    except TypeError:
        embedding = client.feature_extraction(model=EMBEDDING_MODEL, inputs=text)
    return np.asarray(embedding, dtype=np.float32)


def _normalize_question(question: str) -> str:
    return question.strip().lower()


def load_knowledge_map() -> dict[str, str]:
    knowledge_path = _get_knowledge_path()
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Knowledge JSON not found: {knowledge_path}")

    payload = json.loads(knowledge_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Knowledge JSON must be an object mapping question to answer")

    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            normalized[_normalize_question(key)] = value
    return normalized


def _save_knowledge_map(knowledge: dict[str, str]) -> None:
    knowledge_path = _get_knowledge_path()
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge_path.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_question_list(questions: Sequence[str]) -> None:
    question_list_path = _get_question_list_path()
    question_list_path.parent.mkdir(parents=True, exist_ok=True)
    question_list_path.write_text(json.dumps(list(questions), ensure_ascii=False), encoding="utf-8")


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


def _build_answer_prompt(question: str, context_input: str | Sequence[str]) -> str:
    if isinstance(context_input, str):
        context_block = context_input.strip()
    else:
        seed_questions = _clean_seed_questions(context_input)
        context_block = "\n".join(f"- {item}" for item in seed_questions)

    return (
        "You write customer support FAQ answers. "
        "Provide one clear answer in 2-4 sentences and return only the answer text.\n"
        f"Context:\n{context_block}\n"
        f"Question: {question}"
    )


def _generate_answer(
    client: InferenceClient,
    question: str,
    context_input: str | Sequence[str],
    answer_model: str,
) -> str:
    answer = _text_generation(
        client=client,
        model=answer_model,
        prompt_text=_build_answer_prompt(question, context_input),
        max_new_tokens=220,
    )
    return str(answer).strip()


def _text_generation(
    client: InferenceClient,
    model: str,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    try:
        return str(
            client.text_generation(
                model=model,
                prompt=prompt_text,
                max_new_tokens=max_new_tokens,
            )
        )
    except TypeError:
        try:
            return str(
                client.text_generation(
                    model=model,
                    inputs=prompt_text,
                    max_new_tokens=max_new_tokens,
                )
            )
        except Exception:
            return _chat_completion(client=client, model=model, prompt_text=prompt_text)
    except StopIteration:
        return _chat_completion(client=client, model=model, prompt_text=prompt_text)
    except ValueError as exc:
        message = str(exc).lower()
        if (
            "not supported for task text-generation" in message
            or "task 'text-generation' not supported" in message
            or ("text-generation" in message and "conversational" in message)
        ):
            return _chat_completion(client=client, model=model, prompt_text=prompt_text)
        raise


def _chat_completion(client: InferenceClient, model: str, prompt_text: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.7,
    )
    return completion.choices[0].message.content or ""


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
    persist_updates: bool = True,
    question_model: str | None = None,
    answer_model: str | None = None,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> list[str]:
    """Generate FAQ questions and optionally persist generated Q/A + embeddings.

    When ``persist_updates`` is True, accepted questions get generated answers and are
    written to ``support_bot_knowledge.json``, while embeddings are saved back to
    ``support_bot_knowledge.embeddings.npy`` and question order to
    ``support_bot_knowledge.questions.json``.
    """
    if num_questions < 1:
        return []

    effective_question_model = question_model or QUESTION_MODEL
    effective_answer_model = answer_model or ANSWER_MODEL

    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    knowledge_map = load_knowledge_map()
    existing_knowledge_questions = set(knowledge_map.keys())
    existing_embeddings = load_existing_embeddings()
    accepted_questions: list[str] = []
    generated_faqs: list[tuple[str, str]] = []
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
        response = _text_generation(
            client=client,
            model=effective_question_model,
            prompt_text=prompt,
            max_new_tokens=100,
        )
        if progress_callback:
            progress_callback(attempts, len(accepted_questions), num_questions, max_attempts)
        candidate = str(response).strip().splitlines()[0].strip()
        if not candidate:
            continue

        if not candidate.endswith("?"):
            candidate = f"{candidate}?"

        normalized_candidate = _normalize_question(candidate)
        if normalized_candidate in existing_knowledge_questions:
            continue

        if is_duplicate_new_question(client, candidate, existing_embeddings, threshold):
            continue

        accepted_questions.append(candidate)
        generated_faqs.append(
            (
                candidate,
                _generate_answer(
                    client,
                    candidate,
                    context_input,
                    effective_answer_model,
                ),
            )
        )
        existing_knowledge_questions.add(normalized_candidate)
        candidate_embedding = _get_embedding(client, candidate)
        existing_embeddings = np.vstack([existing_embeddings, candidate_embedding])
        if progress_callback:
            progress_callback(attempts, len(accepted_questions), num_questions, max_attempts)

    if persist_updates and generated_faqs:
        for question, answer in generated_faqs:
            knowledge_map[_normalize_question(question)] = answer

        _save_knowledge_map(knowledge_map)
        _save_question_list(list(knowledge_map.keys()))

        embeddings_path = _get_embeddings_path()
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, existing_embeddings.astype(np.float32), allow_pickle=False)

    return accepted_questions


def _load_seed_questions_from_file(seed_file: Path) -> list[str]:
    if not seed_file.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_file}")

    raw_lines = seed_file.read_text(encoding="utf-8").splitlines()
    cleaned = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
    if not cleaned:
        raise ValueError(f"Seed file is empty: {seed_file}")
    return cleaned


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FAQ questions from seed examples")
    parser.add_argument(
        "--seed-file",
        type=Path,
        help="Path to a text file containing one seed question per line",
    )
    parser.add_argument(
        "--seed-text",
        type=str,
        help="Free-text context instead of a seed file",
    )
    parser.add_argument("--num-questions", type=int, default=20, help="Number of questions to generate")
    parser.add_argument("--threshold", type=float, default=0.85, help="Duplicate similarity threshold")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum generation attempts")
    parser.add_argument(
        "--no-cover-seed-topics",
        action="store_true",
        help="Disable rotating focus across seed questions",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist generated Q/A and embeddings back to knowledge files",
    )
    parser.add_argument(
        "--question-model",
        type=str,
        default=QUESTION_MODEL,
        help="Model to use for question generation",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default=ANSWER_MODEL,
        help="Model to use for answer generation",
    )
    return parser


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.seed_file and args.seed_text:
        parser.error("Use either --seed-file or --seed-text, not both")

    if args.seed_file:
        context_input: str | Sequence[str] = _load_seed_questions_from_file(args.seed_file)
    elif args.seed_text:
        context_input = args.seed_text
    else:
        default_seed_file = project_root / "data" / "additional_questions.txt"
        context_input = _load_seed_questions_from_file(default_seed_file)

    def _cli_progress(attempts: int, accepted: int, target: int, max_allowed: int) -> None:
        print(
            f"\rProgress: accepted {accepted}/{target} | attempts {attempts}/{max_allowed}",
            end="",
            flush=True,
        )

    try:
        faqs = generate_questions(
            context_input=context_input,
            num_questions=args.num_questions,
            threshold=args.threshold,
            max_attempts=args.max_attempts,
            cover_seed_topics=not args.no_cover_seed_topics,
            persist_updates=args.persist,
            question_model=args.question_model,
            answer_model=args.answer_model,
            progress_callback=_cli_progress,
        )
    except HfHubHTTPError as exc:
        message = str(exc)
        if "403 Forbidden" in message or "permissions" in message.lower():
            parser.exit(
                2,
                "Hugging Face token lacks inference permissions. "
                "Update HF_TOKEN with permissions for Inference Providers, then retry.\n",
            )
        if "model_not_supported" in message or "not supported by any provider" in message.lower():
            parser.exit(
                2,
                "Selected model is not supported by your enabled providers. "
                "Try --question-model/--answer-model with a model available to your account, "
                "or enable additional providers in Hugging Face settings.\n",
            )
        parser.exit(2, f"Hugging Face inference failed: {message}\n")
    except ValueError as exc:
        parser.exit(
            2,
            "Model/provider configuration error: "
            f"{exc}\n"
            "Set FAQ_QUESTION_MODEL / FAQ_ANSWER_MODEL to a provider-supported conversational model.\n",
        )

    print()

    for index, question in enumerate(faqs, start=1):
        print(f"{index}. {question}")

    print(f"Generated {len(faqs)} question(s). Persisted updates: {args.persist}")
    return 0


# Example cmdline calls (run from project root):
# .venv/bin/python -m support_bot.helpers.generate_faqs --seed-file data/additional_questions.txt --num-questions 20
# .venv/bin/python -m support_bot.helpers.generate_faqs --seed-file data/additional_questions.txt --num-questions 50 --persist
# .venv/bin/python -m support_bot.helpers.generate_faqs --seed-text "Returns and refunds policy" --num-questions 10
# .venv/bin/python -m support_bot.helpers.generate_faqs --seed-file data/additional_questions.txt --num-questions 20 --question-model moonshotai/Kimi-K2-Instruct-0905 --answer-model moonshotai/Kimi-K2-Instruct-0905


if __name__ == "__main__":
    raise SystemExit(main())
