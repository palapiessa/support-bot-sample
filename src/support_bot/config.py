"""Configuration helpers for the support bot sample."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge_base.json"


@dataclass(frozen=True)
class BotConfig:
    """Immutable configuration for building a support bot."""

    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH
    dataset_name: str = "qgyd2021/e_commerce_customer_service"
    dataset_split: str = "train"
    question_field: str = "question"
    answer_field: str = "answer"