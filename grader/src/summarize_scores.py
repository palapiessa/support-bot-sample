from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize grader JSONL results")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("grader/data/llm_eval.core_baseline_v1.jsonl"),
        help="Path to grader JSONL file",
    )
    parser.add_argument(
        "--top-reasons",
        type=int,
        default=5,
        help="How many top failure reasons to print",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path to write markdown summary report",
    )
    return parser


def _score_bucket(value: int) -> str:
    if value <= 1:
        return "low(0-1)"
    if value <= 3:
        return "mid(2-3)"
    return "high(4-5)"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        parser.error(f"Input JSONL not found: {args.input_jsonl}")

    rows = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        parser.error(f"No rows found in JSONL: {args.input_jsonl}")

    total = len(rows)
    pass_count = 0
    fail_count = 0

    semantic_sum = 0
    helpfulness_sum = 0
    tone_sum = 0

    semantic_bucket_counts: Counter[str] = Counter()
    helpfulness_bucket_counts: Counter[str] = Counter()
    tone_bucket_counts: Counter[str] = Counter()

    failure_reason_counts: Counter[str] = Counter()

    for row in rows:
        grade = row.get("grade", {})
        semantic = int(grade.get("semantic_correctness", 0))
        helpfulness = int(grade.get("helpfulness", 0))
        tone = int(grade.get("tone_safety", 0))
        passed = bool(grade.get("passed", False))
        reason = str(grade.get("reason", "")).strip() or "(no reason)"

        semantic_sum += semantic
        helpfulness_sum += helpfulness
        tone_sum += tone

        semantic_bucket_counts[_score_bucket(semantic)] += 1
        helpfulness_bucket_counts[_score_bucket(helpfulness)] += 1
        tone_bucket_counts[_score_bucket(tone)] += 1

        if passed:
            pass_count += 1
        else:
            fail_count += 1
            failure_reason_counts[reason] += 1

    print("=== Summary ===")
    print(f"Total questions: {total}")
    print(f"Passed: {pass_count} ({(pass_count / total) * 100:.1f}%)")
    print(f"Failed: {fail_count} ({(fail_count / total) * 100:.1f}%)")

    print("\n=== Average Scores ===")
    print(f"semantic_correctness: {semantic_sum / total:.2f}")
    print(f"helpfulness: {helpfulness_sum / total:.2f}")
    print(f"tone_safety: {tone_sum / total:.2f}")
    print(f"total: {(semantic_sum + helpfulness_sum + tone_sum) / total:.2f} / 15")

    print("\n=== Score Buckets ===")
    print("semantic_correctness:", dict(semantic_bucket_counts))
    print("helpfulness:", dict(helpfulness_bucket_counts))
    print("tone_safety:", dict(tone_bucket_counts))

    print(f"\n=== Top {args.top_reasons} Failure Reasons ===")
    top_failure_reasons = failure_reason_counts.most_common(args.top_reasons)
    if not failure_reason_counts:
        print("No failure reasons (all rows passed).")
    else:
        for reason, count in top_failure_reasons:
            print(f"{count:>3}  {reason}")

    output_md = args.output_md
    if output_md is not None:
        avg_semantic = semantic_sum / total
        avg_helpfulness = helpfulness_sum / total
        avg_tone = tone_sum / total
        avg_total = (semantic_sum + helpfulness_sum + tone_sum) / total

        lines = [
            "# Grader Summary",
            "",
            f"- Input: {args.input_jsonl}",
            f"- Total questions: {total}",
            f"- Passed: {pass_count} ({(pass_count / total) * 100:.1f}%)",
            f"- Failed: {fail_count} ({(fail_count / total) * 100:.1f}%)",
            "",
            "## Average Scores",
            "",
            f"- semantic_correctness: {avg_semantic:.2f}",
            f"- helpfulness: {avg_helpfulness:.2f}",
            f"- tone_safety: {avg_tone:.2f}",
            f"- total: {avg_total:.2f} / 15",
            "",
            "## Score Buckets",
            "",
            f"- semantic_correctness: {dict(semantic_bucket_counts)}",
            f"- helpfulness: {dict(helpfulness_bucket_counts)}",
            f"- tone_safety: {dict(tone_bucket_counts)}",
            "",
            f"## Top {args.top_reasons} Failure Reasons",
            "",
        ]

        if top_failure_reasons:
            lines.extend(f"- {count} — {reason}" for reason, count in top_failure_reasons)
        else:
            lines.append("- No failure reasons (all rows passed).")

        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved markdown summary to {output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
