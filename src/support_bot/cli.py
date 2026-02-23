"""Command-line access to the support bot.

Example run from project root:
   PYTHONPATH=src python -m support_bot.cli --refresh 
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from .bot import SupportBot, Loader
from .config import BotConfig
from .loader import load_from_disk
from .data_loader import load_faq_pairs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive support bot CLI")
    parser.add_argument(
        "--knowledge-path",
        type=Path,
        default=BotConfig().knowledge_path,
        help="Path to the cached JSON knowledge base",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Override the Hugging Face dataset to load when refreshing",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        help="Override the dataset split name",
    )
    parser.add_argument(
        "--question-field",
        type=str,
        help="Override the field name that contains questions",
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        help="Override the field name that contains answers",
    )
    parser.add_argument("--refresh", action="store_true", help="Reload the dataset before starting")
    return parser


def _build_config(args: argparse.Namespace) -> BotConfig:
    override_kwargs = {}
    if args.dataset_name:
        override_kwargs["dataset_name"] = args.dataset_name
    if args.dataset_split:
        override_kwargs["dataset_split"] = args.dataset_split
    if args.question_field:
        override_kwargs["question_field"] = args.question_field
    if args.answer_field:
        override_kwargs["answer_field"] = args.answer_field
    return BotConfig(knowledge_path=args.knowledge_path, **override_kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _build_config(args)

    if args.refresh:
        print("Refreshing knowledge base from dataset…")
        load_faq_pairs(config=config)

    bot = SupportBot(config, loader=cast(Loader, load_from_disk))
    print("SupportBot ready. Ask a question or press Enter to quit.")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        print(bot.respond(question))


if __name__ == "__main__":
    main()