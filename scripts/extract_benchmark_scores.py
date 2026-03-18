#!/usr/bin/env python3
"""Extract and print benchmark scores from a results directory.

Scans subdirectories (e.g. ruler32k_fwe, ruler32k_qa_1) for metrics.json,
reads the overall_score (or a fallback score), and prints each dataset name
with its score plus the average across all datasets.

Usage:
    python scripts/extract_benchmark_scores.py --directory /path/to/model/masker
"""

import argparse
import json
from pathlib import Path


def get_score_from_metrics(metrics: dict) -> float | None:
    """Extract a single numeric score from a metrics dict.

    Prefers 'overall_score'. If missing, tries task_scores (e.g. string_match).
    """
    if "overall_score" in metrics:
        val = metrics["overall_score"]
        if isinstance(val, (int, float)):
            return float(val)
    if "task_scores" in metrics:
        task_scores = metrics["task_scores"]
        if not task_scores:
            return None
        # Use first task; prefer string_match then first numeric value
        first = next(iter(task_scores.values()))
        if isinstance(first, (int, float)):
            return float(first)
        if isinstance(first, dict) and "string_match" in first:
            return float(first["string_match"])
        for v in first.values() if isinstance(first, dict) else []:
            if isinstance(v, (int, float)):
                return float(v)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract benchmark scores from subdirs and print dataset: score plus average."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the results folder (e.g. .../Qwen_Qwen2.5-72B-Instruct/dense)",
    )
    args = parser.parse_args()

    base = Path(args.directory)
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    results: list[tuple[str, float]] = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics: dict = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        score = get_score_from_metrics(metrics)
        if score is not None:
            results.append((subdir.name, score))

    for name, score in results:
        print(f"{name}: {score}")

    if results:
        avg = sum(s for _, s in results) / len(results)
        print(f"average: {avg}")
    else:
        print("No metrics found in subdirectories.")


if __name__ == "__main__":
    main()
