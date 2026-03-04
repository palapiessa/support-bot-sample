from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize grader JSONL results")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("grader/data/llm_eval.core_baseline_v1.jsonl"),
        help="Path to grader JSONL file",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("grader/data/llm_eval_baseline_v1.png"),
        help="Path to output PNG image",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        parser.error(f"Input JSONL not found: {args.input_jsonl}")

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        parser.error(
            "Missing visualization dependencies. Install with: "
            "../.venv/bin/python -m pip install pandas matplotlib"
        )

    rows = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        parser.error(f"No rows found in JSONL: {args.input_jsonl}")

    grades = pd.json_normalize([row["grade"] for row in rows])
    grades["total"] = (
        grades["semantic_correctness"] + grades["helpfulness"] + grades["tone_safety"]
    )

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].hist(
        [
            grades["semantic_correctness"],
            grades["helpfulness"],
            grades["tone_safety"],
        ],
        bins=6,
        alpha=0.7,
        label=["semantic_correctness", "helpfulness", "tone_safety"],
    )
    axes[0, 0].set_title("Score Distributions")
    axes[0, 0].legend(fontsize=8)

    grades["passed"].value_counts().plot(kind="bar", ax=axes[0, 1], title="Pass/Fail")

    grades[["semantic_correctness", "helpfulness", "tone_safety", "total"]].mean().plot(
        kind="bar", ax=axes[1, 0], title="Average Scores"
    )

    grades["reason"].astype(str).str.len().plot(
        kind="hist", bins=20, ax=axes[1, 1], title="Reason Length"
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=180)
    print(f"saved {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
