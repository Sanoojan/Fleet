#!/usr/bin/env python3
"""Aggregate Fleet Table 1 few-shot runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_SUBSETS = {"Treasure": 64, "AIGIBench-13": 13}


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Binary AP with AI as the positive class, returned as a percentage."""
    if not labels or sum(labels) == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return 100.0 * precision_sum / sum(labels)


def weighted_accuracy(*results: dict) -> float:
    total = sum(int(result.get("n_samples", 0)) for result in results)
    if total == 0:
        return 0.0
    correct = sum(
        float(result.get("total_acc", 0.0)) * int(result.get("n_samples", 0))
        for result in results
    )
    return correct / total


def load_run(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text())
    final = payload["final_results"]
    query_key = next(key for key in final if key.startswith("Query"))
    query = final[query_key]
    non_ai = final["final/real"]
    val_fake = final["AIGIBench/val-fake"]
    val_real = final["AIGIBench/val-real"]

    labels = [1] * len(query.get("confidences", []))
    scores = [-float(value) for value in query.get("confidences", [])]
    labels.extend([0] * len(non_ai.get("confidences", [])))
    scores.extend([-float(value) for value in non_ai.get("confidences", [])])

    ai_acc = float(query["total_acc"])
    non_ai_acc = float(non_ai["total_acc"])
    return {
        "dataset": payload["dataset_name"],
        "ai_acc": ai_acc,
        "non_ai_acc": non_ai_acc,
        "macc": (ai_acc + non_ai_acc) / 2.0,
        "ap": average_precision(labels, scores),
        "pretrain_val": weighted_accuracy(val_fake, val_real),
        "result_path": str(result_path),
    }


def mean(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Fleet Table 1 results")
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()

    detail_rows = []
    summary_rows = []
    for benchmark, expected in EXPECTED_SUBSETS.items():
        for shot in (1, 5, 10):
            result_paths = sorted(
                (args.output_root / "runs" / benchmark / f"{shot}shot").glob(
                    "*/fewshot_results.json"
                )
            )
            rows = []
            for result_path in result_paths:
                try:
                    row = load_run(result_path)
                except Exception as exc:
                    print(f"[summary warning] {result_path}: {exc}")
                    continue
                row.update({"benchmark": benchmark, "shot": shot})
                rows.append(row)
                detail_rows.append(row)

            ai_acc = mean(rows, "ai_acc")
            non_ai_acc = mean(rows, "non_ai_acc")
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "shot": shot,
                    "complete": len(rows),
                    "expected": expected,
                    "ai_acc": ai_acc,
                    "non_ai_acc": non_ai_acc,
                    "macc": None
                    if ai_acc is None or non_ai_acc is None
                    else (ai_acc + non_ai_acc) / 2.0,
                    "map": mean(rows, "ap"),
                    "pretrain_val": mean(rows, "pretrain_val"),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "table1_summary.json").write_text(
        json.dumps({"summary": summary_rows, "details": detail_rows}, indent=2)
    )

    fieldnames = [
        "benchmark",
        "shot",
        "complete",
        "expected",
        "ai_acc",
        "non_ai_acc",
        "macc",
        "map",
        "pretrain_val",
    ]
    with (args.output_root / "table1_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "Fleet Table 1 reproduction",
        "Benchmark       Shot   Done       AI   Non-AI     mAcc     mAP  PretrainVal",
        "-" * 79,
    ]
    for row in summary_rows:
        def fmt(value: float | None) -> str:
            return "   N/A" if value is None else f"{value:7.2f}"

        lines.append(
            f"{row['benchmark']:<16} {row['shot']:>2}-shot "
            f"{row['complete']:>3}/{row['expected']:<3} "
            f"{fmt(row['ai_acc'])} {fmt(row['non_ai_acc'])} "
            f"{fmt(row['macc'])} {fmt(row['map'])} {fmt(row['pretrain_val'])}"
        )
    text = "\n".join(lines) + "\n"
    (args.output_root / "table1_summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
