#!/usr/bin/env python3
"""Joint t-SNE visualization of pretrain and 10-shot routing weights."""

from __future__ import annotations

import argparse
import csv
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from fleet.train.plot_table1_comparison import TREASURE_ORDER


MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h")


def load_weights(input_root: Path):
    pretrain_chunks = []
    fewshot_chunks = []
    labels = []
    paths = []
    for dataset_index, dataset in enumerate(TREASURE_ORDER):
        archive_path = input_root / "routing" / dataset / "routing_weights.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive = np.load(archive_path)
        pretrain = archive["pretrain"].astype(np.float32)
        fewshot = archive["fewshot"].astype(np.float32)
        sample_paths = archive["paths"].astype(str)
        if pretrain.shape != fewshot.shape or pretrain.shape[0] != len(sample_paths):
            raise ValueError(f"Mismatched routing export: {archive_path}")
        pretrain_chunks.append(pretrain)
        fewshot_chunks.append(fewshot)
        labels.extend([dataset_index] * pretrain.shape[0])
        paths.extend(sample_paths.tolist())
    return (
        np.concatenate(pretrain_chunks),
        np.concatenate(fewshot_chunks),
        np.asarray(labels),
        np.asarray(paths),
    )


def scatter_phase(ax, embedding, labels, title, colors):
    marker_cycle = cycle(MARKERS)
    for index, dataset in enumerate(TREASURE_ORDER):
        mask = labels == index
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=15, alpha=0.68, color=colors[index], marker=next(marker_cycle),
            edgecolors="none", label=f"{index + 1:02d} {dataset}",
        )
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.16, linewidth=0.5)


def save_single(embedding, labels, title, stem, output_dir, colors):
    fig, ax = plt.subplots(figsize=(18, 13), constrained_layout=True)
    scatter_phase(ax, embedding, labels, title, colors)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncols=8,
               fontsize=7, frameon=False, markerscale=1.5)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs_routing_tsne"))
    parser.add_argument("--output-dir", type=Path, default=Path("Plots"))
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--max-iter", type=int, default=1500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pretrain, fewshot, labels, paths = load_weights(args.input_root)
    combined = np.concatenate([pretrain, fewshot], axis=0)
    print(f"Running joint t-SNE on {combined.shape[0]} routing vectors of dimension {combined.shape[1]}")
    embedding = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(combined)
    pretrain_embedding, fewshot_embedding = np.split(embedding, 2)

    colors = plt.colormaps["turbo"](np.linspace(0.02, 0.98, len(TREASURE_ORDER)))
    save_single(
        pretrain_embedding, labels,
        "Routing weights after AIGIBench pretraining (64 Treasure generators)",
        "routing_tsne_after_pretraining", args.output_dir, colors,
    )
    save_single(
        fewshot_embedding, labels,
        "Routing weights after category-specific 10-shot adaptation",
        "routing_tsne_after_full_fewshot", args.output_dir, colors,
    )

    fig, axes = plt.subplots(1, 2, figsize=(26, 12), constrained_layout=True)
    scatter_phase(axes[0], pretrain_embedding, labels, "After pretraining", colors)
    scatter_phase(axes[1], fewshot_embedding, labels, "After 10-shot adaptation", colors)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncols=8,
               fontsize=7, frameon=False, markerscale=1.5)
    fig.suptitle("FLEET routing distributions: matched-image joint t-SNE", fontsize=18, weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"routing_tsne_pretrain_vs_full_fewshot.{suffix}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        args.output_dir / "routing_tsne_embedding.npz",
        pretrain_embedding=pretrain_embedding,
        fewshot_embedding=fewshot_embedding,
        labels=labels,
        paths=paths,
        dataset_names=np.asarray(TREASURE_ORDER),
        pretrain_weights=pretrain,
        fewshot_weights=fewshot,
    )
    with (args.output_dir / "routing_tsne_embedding.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "dataset_index", "dataset", "path", "tsne_1", "tsne_2"))
        for phase, phase_embedding in (("pretrain", pretrain_embedding), ("fewshot", fewshot_embedding)):
            for index, point in enumerate(phase_embedding):
                dataset_index = int(labels[index])
                writer.writerow((phase, dataset_index + 1, TREASURE_ORDER[dataset_index],
                                 paths[index], float(point[0]), float(point[1])))


if __name__ == "__main__":
    main()
