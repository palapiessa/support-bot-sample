"""Configuration helpers for the support bot sample."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "support_bot_knowledge.json"


@dataclass(frozen=True)
class BotConfig:
    """Immutable configuration for building a support bot."""

    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH
    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    dataset_split: str = "train"
    question_field: str = "instruction"
    answer_field: str = "response"