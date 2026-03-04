"""Dataset-only evaluation runner for the local support bot.

Purpose:
- Read a CSV of test questions.
- Run each question through ``SupportBot.respond``.
- Write answers back to a new CSV for later grading/analysis.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
GRADER_DATA_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
SUPPORT_BOT_DIR = REPO_ROOT / "support_bot"

try:
    from support_bot.bot import SupportBot
    from support_bot.config import BotConfig
    from support_bot.loader import load_from_disk
except ModuleNotFoundError:
    candidate_paths = [
        SUPPORT_BOT_DIR,
        SUPPORT_BOT_DIR / "src",
        REPO_ROOT / "src",
    ]
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    try:
        from src.bot import SupportBot
        from src.config import BotConfig
        from src.loader import load_from_disk
    except ModuleNotFoundError:
        from support_bot.bot import SupportBot
        from support_bot.config import BotConfig
        from support_bot.loader import load_from_disk


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for input/output paths and optional row limit."""
    parser = argparse.ArgumentParser(description="Run support bot on evaluation CSV and write answers")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=GRADER_DATA_DIR / "llm_eval_questions.csv",
        help="Path to input CSV with at least a 'question' column",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=GRADER_DATA_DIR / "llm_eval_answers.csv",
        help="Path to output CSV with bot answers",
    )
    parser.add_argument(
        "--knowledge-path",
        type=Path,
        default=SUPPORT_BOT_DIR / "data" / "support_bot_knowledge.json",
        help="Knowledge base JSON path used by the bot",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows to process (0 = all)",
    )
    return parser


def main() -> int:
    """Execute the eval pass: load bot, process rows, and persist answers."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.input_csv.exists():
        parser.error(f"Input CSV not found: {args.input_csv}")

    config = BotConfig(knowledge_path=args.knowledge_path)
    # Load the bot with disk-backed knowledge so results are reproducible.
    bot = SupportBot(config, loader=load_from_disk)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            parser.error("Input CSV must contain a 'question' column")

        output_fieldnames = list(reader.fieldnames)
        if "bot_answer" not in output_fieldnames:
            output_fieldnames.append("bot_answer")

        with args.output_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
            writer.writeheader()

            processed = 0
            for row in reader:
                # Keep empty questions explicit in output rather than calling the bot.
                question = (row.get("question") or "").strip()
                row["bot_answer"] = bot.respond(question) if question else ""
                writer.writerow(row)

                processed += 1
                print(f"\rProcessed: {processed}", end="", flush=True)
                if args.limit and processed >= args.limit:
                    break

    print(f"\nDone. Wrote {processed} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
