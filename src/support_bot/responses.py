"""Response selection helpers for the support bot."""

from typing import Mapping


def choose_response(knowledge: Mapping[str, str], question: str) -> str:
    """Pick a reply from the knowledge base using the simplest match available."""

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

    if fallback:
        return fallback

    return "I am still learning and cannot answer that right now."