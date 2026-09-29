#!/usr/bin/env python3
"""Plot zero-shot versus 1/5/10-shot Fleet Table 1 results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fleet.train.table1_summary import average_precision, weighted_accuracy


SETTINGS = ("Zero-shot", "1-shot", "5-shot", "10-shot")
SHOTS = (1, 5, 10)
METRICS = ("ai_acc", "non_ai_acc", "macc", "map", "pretrain_val")
METRIC_LABELS = {
    "ai_acc": "AI accuracy",
    "non_ai_acc": "Non-AI accuracy",
    "macc": "mAcc",
    "map": "mAP",
    "pretrain_val": "Pretrain validation",
}
COLORS = ("#5B5B5B", "#4C78A8", "#F2B134", "#E45756")

# Appendix Tables 10 and 12 order. Names below match the local directories.
AIGIBENCH_ORDER = (
    "ProGAN", "R3GAN", "StyleGAN3", "StyleGAN-XL", "StyleSwim", "WFIR",
    "DALLE-3", "FLUX1-dev", "GLIDE", "Imagen3", "Midjourney", "SD3", "SDXL",
)
TREASURE_ORDER = (
    "ADM", "BAGEL-7B", "BRIA_v3_2", "BigGAN", "CogView2", "CogView4",
    "Cogview3-plus", "DALLE-2", "DALLE-3", "DF-GAN", "DeepFloyd_IF",
    "FLUX.1-dev", "FLUX.2", "GLIDE", "GPT4O_Image_T2I", "GigaGAN",
    "HiDream-I1-Dev", "HunyuanDiT", "HunyuanImage-3.0", "Imagen", "Imagen4",
    "Infinity", "Janus-Pro-7B", "Kolors", "LlamaGen", "LongCat-Image", "Lumina",
    "MAE", "Midjourney V6.1", "Midjourney V7", "Midjourney_V4",
    "Midjourney_V5", "Midjourney_V6", "Nano Banana", "Nano-Banana-Pro",
    "NextStep", "OmniGen_v1", "OmniGen_v2", "Playground_v2", "Playground_v2.5",
    "ProGAN", "Qwen-Image", "SD3-Medium", "SDXL", "SDv1.4", "SDv1.5",
    "SDv2.1", "Sana_v1.5", "Show_o", "Show_o2", "StarGAN", "StyleGAN3",
    "VQDM", "Wukong", "Z-Image-Turbo", "doubao-seedream-3.0-t2i",
    "doubao-seedream-4.0", "gpt-image-1.5", "ideogram", "ovis-U1", "pixart-α",
    "sora-image", "wan2.2-t2i-flash", "wan2.5-t2i-preview",
)
ORDERS = {"Treasure": TREASURE_ORDER, "AIGIBench-13": AIGIBENCH_ORDER}


def metrics_from_block(block: dict) -> dict[str, float]:
    query_key = next(key for key in block if key.startswith("Query"))
    query = block[query_key]
    non_ai = block["final/real"]
    val_fake = block["AIGIBench/val-fake"]
    val_real = block["AIGIBench/val-real"]

    labels = [1] * len(query.get("confidences", []))
    scores = [-float(value) for value in query.get("confidences", [])]
    labels.extend([0] * len(non_ai.get("confidences", [])))
    scores.extend([-float(value) for value in non_ai.get("confidences", [])])

    ai_acc = float(query["total_acc"])
    non_ai_acc = float(non_ai["total_acc"])
    return {
        "ai_acc": ai_acc,
        "non_ai_acc": non_ai_acc,
        "macc": (ai_acc + non_ai_acc) / 2.0,
        "map": average_precision(labels, scores),
        "pretrain_val": weighted_accuracy(val_fake, val_real),
    }


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    return {metric: sum(item[metric] for item in items) / len(items) for metric in METRICS}


def load_results(input_root: Path) -> tuple[dict, list[dict]]:
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    raw_rows: list[dict] = []

    for benchmark, order in ORDERS.items():
        data[benchmark] = {}
        for dataset in order:
            initial_candidates = []
            final_by_shot = {}
            for shot in SHOTS:
                result_path = input_root / "runs" / benchmark / f"{shot}shot" / dataset / "fewshot_results.json"
                if not result_path.is_file():
                    raise FileNotFoundError(result_path)
                payload = json.loads(result_path.read_text())
                initial = metrics_from_block(payload["initial_results"])
                final = metrics_from_block(payload["final_results"])
                initial_candidates.append(initial)
                final_by_shot[shot] = final
                for phase, values in (("zero_initial", initial), (f"{shot}-shot", final)):
                    raw_rows.append(
                        {"benchmark": benchmark, "dataset": dataset, "source_shot": shot,
                         "phase": phase, **values}
                    )

            data[benchmark][dataset] = {
                "Zero-shot": mean_metrics(initial_candidates),
                "1-shot": final_by_shot[1],
                "5-shot": final_by_shot[5],
                "10-shot": final_by_shot[10],
            }
    return data, raw_rows


def aggregate(data: dict) -> list[dict]:
    rows = []
    for benchmark, datasets in data.items():
        for setting in SETTINGS:
            values = [dataset_values[setting] for dataset_values in datasets.values()]
            rows.append({"benchmark": benchmark, "setting": setting, **mean_metrics(values)})
    return rows


def plot_aggregate(rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True, constrained_layout=True)
    x = np.arange(len(METRICS))
    width = 0.19
    for ax, benchmark in zip(axes, ORDERS):
        benchmark_rows = {row["setting"]: row for row in rows if row["benchmark"] == benchmark}
        for index, setting in enumerate(SETTINGS):
            values = [benchmark_rows[setting][metric] for metric in METRICS]
            bars = ax.bar(x + (index - 1.5) * width, values, width, label=setting,
                          color=COLORS[index], edgecolor="white", linewidth=0.5)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2, rotation=90)
        ax.set_title(benchmark, fontsize=14, weight="bold")
        ax.set_xticks(x, [METRIC_LABELS[metric] for metric in METRICS], rotation=18, ha="right")
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Score (%)")
    axes[0].legend(ncols=4, loc="lower left", frameon=False)
    fig.suptitle("FLEET: AIGIBench-pretrained zero-shot vs. few-shot adaptation", fontsize=16, weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"table1_aggregate_zero_vs_fewshot.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_individual(data: dict, benchmark: str, output_dir: Path) -> None:
    order = ORDERS[benchmark]
    width = max(16, len(order) * 0.48)
    fig, axes = plt.subplots(2, 1, figsize=(width, 9), constrained_layout=True)
    for ax, metric in zip(axes, ("ai_acc", "non_ai_acc")):
        matrix = np.asarray(
            [[data[benchmark][dataset][setting][metric] for dataset in order] for setting in SETTINGS]
        )
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="viridis")
        ax.set_yticks(np.arange(len(SETTINGS)), SETTINGS)
        ax.set_xticks(np.arange(len(order)), order, rotation=62, ha="right", fontsize=8)
        ax.set_title(f"{METRIC_LABELS[metric]} by subset", loc="left", weight="bold")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if value < 45 or value > 85 else "black"
                ax.text(column, row, f"{value:.0f}", ha="center", va="center",
                        fontsize=5.5 if len(order) > 20 else 8, color=color)
        fig.colorbar(image, ax=ax, fraction=0.012, pad=0.01, label="Accuracy (%)")
    fig.suptitle(
        f"FLEET {benchmark}: zero-shot vs. 1/5/10-shot (paper subset order)",
        fontsize=16, weight="bold",
    )
    stem = benchmark.lower().replace("-", "").replace(" ", "_")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}_individual_zero_vs_fewshot.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(data: dict, raw_rows: list[dict], aggregate_rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["benchmark", "dataset", "source_shot", "phase", *METRICS]
    with (output_dir / "table1_zero_vs_fewshot_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    individual_rows = []
    for benchmark, datasets in data.items():
        for dataset, settings in datasets.items():
            for setting, values in settings.items():
                individual_rows.append({"benchmark": benchmark, "dataset": dataset,
                                        "setting": setting, **values})
    fields = ["benchmark", "dataset", "setting", *METRICS]
    with (output_dir / "table1_zero_vs_fewshot_individual.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(individual_rows)

    fields = ["benchmark", "setting", *METRICS]
    with (output_dir / "table1_zero_vs_fewshot_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    (output_dir / "table1_zero_vs_fewshot_summary.json").write_text(
        json.dumps({"aggregate": aggregate_rows, "individual": individual_rows}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs_table1"))
    parser.add_argument("--output-dir", type=Path, default=Path("Plots"))
    args = parser.parse_args()
    data, raw_rows = load_results(args.input_root)
    aggregate_rows = aggregate(data)
    write_outputs(data, raw_rows, aggregate_rows, args.output_dir)
    plot_aggregate(aggregate_rows, args.output_dir)
    for benchmark in ORDERS:
        plot_individual(data, benchmark, args.output_dir)
    for row in aggregate_rows:
        print(
            f"{row['benchmark']:>14} {row['setting']:>9}: "
            f"AI={row['ai_acc']:.2f} Non-AI={row['non_ai_acc']:.2f} "
            f"mAcc={row['macc']:.2f} mAP={row['map']:.2f} Val={row['pretrain_val']:.2f}"
        )


if __name__ == "__main__":
    main()
