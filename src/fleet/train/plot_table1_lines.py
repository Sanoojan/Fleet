#!/usr/bin/env python3
"""Create four-line per-subset plots for Fleet Table 1 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fleet.train.plot_table1_comparison import (
    COLORS,
    METRIC_LABELS,
    ORDERS,
    SETTINGS,
    load_results,
)


MARKERS = ("o", "s", "^", "D")


def plot_lines(data: dict, benchmark: str, output_dir: Path) -> None:
    order = ORDERS[benchmark]
    x = np.arange(len(order))
    figure_width = max(17, len(order) * 0.48)
    fig, axes = plt.subplots(2, 1, figsize=(figure_width, 10), sharex=True,
                             constrained_layout=True)

    for ax, metric in zip(axes, ("ai_acc", "non_ai_acc")):
        for index, setting in enumerate(SETTINGS):
            values = [data[benchmark][dataset][setting][metric] for dataset in order]
            ax.plot(
                x,
                values,
                label=setting,
                color=COLORS[index],
                marker=MARKERS[index],
                markersize=4.5 if len(order) < 20 else 3.2,
                linewidth=2.0 if len(order) < 20 else 1.5,
                alpha=0.95,
            )
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(-2, 104)
        ax.set_title(METRIC_LABELS[metric], loc="left", fontsize=13, weight="bold")
        ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.3)
        ax.legend(ncols=4, loc="lower left", frameon=False)

    axes[-1].set_xticks(x, order, rotation=62, ha="right", fontsize=8)
    axes[-1].set_xlabel("Generator subset (paper order)")
    fig.suptitle(
        f"FLEET {benchmark}: zero-shot vs. few-shot adaptation",
        fontsize=17,
        weight="bold",
    )

    stem = benchmark.lower().replace("-", "").replace(" ", "_")
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{stem}_individual_zero_vs_fewshot_lines.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs_table1"))
    parser.add_argument("--output-dir", type=Path, default=Path("Plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, _ = load_results(args.input_root)
    for benchmark in ORDERS:
        plot_lines(data, benchmark, args.output_dir)


if __name__ == "__main__":
    main()
