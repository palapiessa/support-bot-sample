# Grader Summary

- Input: grader/data/llm_eval.core_baseline_v1.jsonl
- Total questions: 50
- Passed: 15 (30.0%)
- Failed: 35 (70.0%)

## Average Scores

- semantic_correctness: 3.40
- helpfulness: 2.86
- tone_safety: 5.00
- total: 11.26 / 15

## Score Buckets

- semantic_correctness: {'high(4-5)': 15, 'low(0-1)': 5, 'mid(2-3)': 30}
- helpfulness: {'high(4-5)': 15, 'low(0-1)': 5, 'mid(2-3)': 30}
- tone_safety: {'high(4-5)': 50}

## Top 5 Failure Reasons

- 2 — Fails to address the question; avoids providing useful information.
- 2 — Fails to provide actionable information; avoids answering the question directly.
- 1 — Incorrect and unhelpful response; fails to provide actionable steps for missing package.
- 1 — No actual warranty policy stated; assumes validity without evidence.
- 1 — Fails to provide actionable steps; incorrect response as it avoids the issue.
