#!/usr/bin/env python3
"""Extract and visualize pretrained FLEET routing weights by generator family.

The script deliberately evaluates the frozen checkpoint only.  It creates:

* Treasure: all 64 fake generators, colored by six architecture families.
* AIGIBench/val: fake versus real, with content-category centroids.
* AIGIBench/test: all 25 fake subsets, colored by five architecture families.

Silhouette scores are computed in the original routing-weight space rather than
the t-SNE projection, because t-SNE is intended for visualization only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from fleet.datasets.dual_branch_fewshot import DualBranchImageDataset
from fleet.train.plot_table1_comparison import TREASURE_ORDER
from fleet.utils import infer_q_dim_and_spec_from_paths, load_q_train_module


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

TREASURE_FAMILIES = OrderedDict(
    [
        ("GAN", ("BigGAN", "ProGAN", "StarGAN", "DF-GAN", "StyleGAN3", "GigaGAN")),
        (
            "Conventional / latent diffusion",
            (
                "ADM", "GLIDE", "Wukong", "VQDM", "SDv1.4", "SDv1.5", "DALLE-2",
                "Imagen", "SDXL", "SDv2.1", "DeepFloyd_IF", "Playground_v2",
                "Playground_v2.5", "Kolors",
            ),
        ),
        (
            "Diffusion transformer / rectified flow",
            (
                "pixart-α", "FLUX.1-dev", "HunyuanDiT", "SD3-Medium", "OmniGen_v1",
                "Cogview3-plus", "Sana_v1.5", "Lumina", "HiDream-I1-Dev", "BRIA_v3_2",
                "OmniGen_v2", "Z-Image-Turbo", "LongCat-Image", "Qwen-Image", "Imagen4",
                "doubao-seedream-4.0", "doubao-seedream-3.0-t2i", "FLUX.2",
                "wan2.2-t2i-flash", "wan2.5-t2i-preview", "CogView4",
            ),
        ),
        (
            "Autoregressive / token-based",
            ("CogView2", "LlamaGen", "Infinity", "Janus-Pro-7B", "ovis-U1", "NextStep", "HunyuanImage-3.0"),
        ),
        ("Hybrid masked-token / AR / diffusion", ("MAE", "Show_o", "BAGEL-7B", "Show_o2")),
        (
            "Proprietary / undisclosed",
            (
                "Midjourney_V4", "Midjourney_V5", "Midjourney_V6", "Midjourney V6.1",
                "Midjourney V7", "ideogram", "DALLE-3", "GPT4O_Image_T2I", "Nano Banana",
                "Nano-Banana-Pro", "sora-image", "gpt-image-1.5",
            ),
        ),
    ]
)

AIGIBENCH_FAMILIES = OrderedDict(
    [
        ("GAN — full-image generation", ("ProGAN", "R3GAN", "StyleGAN3", "StyleGAN-XL", "StyleSwim", "WFIR")),
        (
            "Diffusion / flow — text-to-image",
            ("SDXL", "SD3", "DALLE-3", "Midjourney", "FLUX1-dev", "Imagen3", "GLIDE"),
        ),
        ("GAN / StyleGAN — face manipulation", ("BlendFace", "E4S", "FaceSwap", "InSwap", "SimSwap")),
        (
            "Diffusion — personalized generation",
            ("InstantID", "Infinite_ID", "PhotoMaker", "BLIP", "IP_Adapter"),
        ),
        ("Mixed / unknown — in the wild", ("CommunityAI", "SocialRF")),
    ]
)

# Redundant color + marker encoding for color-vision accessibility. Repeated
# colors remain distinguishable by marker shape.
ACCESSIBLE_FAMILY_STYLES = (
    ("#0072B2", "o"),  # blue circle
    ("#F0C808", "o"),  # yellow circle
    ("#D62728", "o"),  # red circle
    ("#6B6B6B", "X"),  # gray cross
    ("#0072B2", "X"),  # blue cross
    ("#D62728", "X"),  # red cross
)


def invert_family_map(groups: OrderedDict[str, tuple[str, ...]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for family, datasets in groups.items():
        for dataset in datasets:
            if dataset in result:
                raise ValueError(f"Duplicate family assignment for {dataset}")
            result[dataset] = family
    return result


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def deterministic_sample(paths: list[Path], count: int, seed: int) -> list[Path]:
    if not paths:
        return []
    if count <= 0 or len(paths) <= count:
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, count))


def collect_treasure(root: Path, count: int) -> dict[str, list]:
    family_by_dataset = invert_family_map(TREASURE_FAMILIES)
    expected = set(TREASURE_ORDER)
    if set(family_by_dataset) != expected:
        raise ValueError(
            f"Treasure family map mismatch; missing={sorted(expected - set(family_by_dataset))}, "
            f"extra={sorted(set(family_by_dataset) - expected)}"
        )
    records = {"paths": [], "dataset": [], "family": [], "class_name": []}
    for index, dataset in enumerate(TREASURE_ORDER):
        selected = deterministic_sample(image_files(root / dataset), count, 1000 + index)
        if not selected:
            raise FileNotFoundError(f"No images found for Treasure/{dataset}")
        records["paths"].extend(map(str, selected))
        records["dataset"].extend([dataset] * len(selected))
        records["family"].extend([family_by_dataset[dataset]] * len(selected))
        records["class_name"].extend(["Fake"] * len(selected))
    return records


def resolve_binary_dirs(subset_dir: Path) -> tuple[Path | None, Path | None]:
    candidates = (subset_dir / subset_dir.name, subset_dir)
    for candidate in candidates:
        real_dir = candidate / "0_real"
        fake_dir = candidate / "1_fake"
        if not fake_dir.is_dir():
            fake_dir = candidate / "1_false"
        if real_dir.is_dir() or fake_dir.is_dir():
            return (real_dir if real_dir.is_dir() else None, fake_dir if fake_dir.is_dir() else None)
    return None, None


def collect_aigibench_val(root: Path, count_per_class: int) -> dict[str, list]:
    records = {"paths": [], "dataset": [], "family": [], "class_name": []}
    subset_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for index, subset_dir in enumerate(subset_dirs):
        real_dir, fake_dir = resolve_binary_dirs(subset_dir)
        for class_name, class_dir, offset in (("Real", real_dir, 0), ("Fake", fake_dir, 1)):
            if class_dir is None:
                continue
            selected = deterministic_sample(image_files(class_dir), count_per_class, 2000 + 2 * index + offset)
            records["paths"].extend(map(str, selected))
            records["dataset"].extend([subset_dir.name] * len(selected))
            records["family"].extend([class_name] * len(selected))
            records["class_name"].extend([class_name] * len(selected))
    if not records["paths"]:
        raise FileNotFoundError(f"No AIGIBench validation images under {root}")
    return records


def collect_aigibench_test(root: Path, count: int) -> dict[str, list]:
    family_by_dataset = invert_family_map(AIGIBENCH_FAMILIES)
    local = {path.name for path in root.iterdir() if path.is_dir()}
    expected = set(family_by_dataset)
    if local != expected:
        raise ValueError(
            f"AIGIBench test family map mismatch; missing={sorted(expected - local)}, "
            f"extra={sorted(local - expected)}"
        )
    records = {"paths": [], "dataset": [], "family": [], "class_name": []}
    ordered_datasets = [dataset for members in AIGIBENCH_FAMILIES.values() for dataset in members]
    for index, dataset in enumerate(ordered_datasets):
        _, fake_dir = resolve_binary_dirs(root / dataset)
        if fake_dir is None:
            raise FileNotFoundError(f"No fake directory for AIGIBench/test/{dataset}")
        selected = deterministic_sample(image_files(fake_dir), count, 3000 + index)
        records["paths"].extend(map(str, selected))
        records["dataset"].extend([dataset] * len(selected))
        records["family"].extend([family_by_dataset[dataset]] * len(selected))
        records["class_name"].extend(["Fake"] * len(selected))
    return records


def load_model(checkpoint_path: Path, dino_path: Path, device: torch.device):
    q_dim, q_spec = infer_q_dim_and_spec_from_paths([checkpoint_path])
    if q_dim is None:
        raise ValueError(f"Could not infer q dimension from {checkpoint_path}")
    q_mod = load_q_train_module(q_spec or str(q_dim), fallback_q_dim=q_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = q_mod.DualBranchContrastiveModel(
        dinov3_model_path=str(dino_path),
        xception_feature_dim=1024,
        projection_dim=256,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        num_heads=1024 // q_dim,
        q_dim=q_dim,
        feature_layer=checkpoint.get("feature_layer", -1),
        use_last_hidden_state=bool(checkpoint.get("use_last_hidden_state", False)),
        normalize_feature=bool(checkpoint.get("normalize_feature", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    return model.to(device).eval()


def extract_routing(
    model,
    processor,
    records: dict[str, list],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    labels = [0] * len(records["paths"])
    dataset = DualBranchImageDataset(
        records["paths"], labels, processor=processor, xception_crop_size=128, is_training=False
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    chunks = []
    with torch.inference_mode():
        for batch_index, (dino_images, xception_images, _) in enumerate(loader, start=1):
            dino_images = dino_images.to(device, non_blocking=True)
            xception_images = xception_images.to(device, non_blocking=True)
            _, routing = model(dino_images, xception_images)
            chunks.append(routing.float().cpu().numpy())
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"  routing batches: {batch_index}/{len(loader)}", flush=True)
    return np.concatenate(chunks, axis=0)


def encode(values: list[str]) -> tuple[np.ndarray, list[str]]:
    names = list(dict.fromkeys(values))
    index = {name: i for i, name in enumerate(names)}
    return np.asarray([index[value] for value in values], dtype=np.int32), names


def safe_silhouette(weights: np.ndarray, values: list[str]) -> float | None:
    labels, names = encode(values)
    if len(names) < 2 or len(names) >= len(weights):
        return None
    return float(silhouette_score(weights, labels, metric="euclidean"))


def run_tsne(weights: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    used_perplexity = min(perplexity, max(5.0, (len(weights) - 1) / 3.0))
    return TSNE(
        n_components=2, perplexity=used_perplexity, init="pca", learning_rate="auto",
        max_iter=1500, random_state=seed,
    ).fit_transform(weights)


def plot_grouped(
    embedding: np.ndarray,
    records: dict[str, list],
    family_order: list[str],
    title: str,
    subtitle: str,
    stem: Path,
) -> None:
    families = np.asarray(records["family"])
    datasets = np.asarray(records["dataset"])
    many_datasets = len(set(records["dataset"])) > 30
    fig = plt.figure(figsize=(19, 18 if many_datasets else 15))
    ax = fig.add_axes([0.06, 0.34 if many_datasets else 0.25, 0.74, 0.60 if many_datasets else 0.67])
    style_by_family = {
        family: ACCESSIBLE_FAMILY_STYLES[index]
        for index, family in enumerate(family_order)
    }
    for family in family_order:
        mask = families == family
        color, marker = style_by_family[family]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], s=22, alpha=0.72,
            color=color, marker=marker,
            edgecolors="#303030" if marker == "o" else color,
            linewidths=0.25 if marker == "o" else 0.65,
            rasterized=True, label=family,
        )

    dataset_order = list(dict.fromkeys(records["dataset"]))
    for index, dataset in enumerate(dataset_order, start=1):
        mask = datasets == dataset
        center = np.median(embedding[mask], axis=0)
        family = families[np.flatnonzero(mask)[0]]
        color, _ = style_by_family[family]
        ax.scatter(center[0], center[1], s=92, color=color, marker="o", edgecolor="black", linewidth=1.0)
        ax.annotate(
            f"{index:02d}", center, ha="center", va="center", fontsize=6.5,
            color="white", weight="bold",
        )

    ax.set_title(title, fontsize=18, weight="bold", pad=18)
    ax.text(0.5, 1.002, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.14, linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)

    rows = []
    for index, dataset in enumerate(dataset_order, start=1):
        family = families[np.flatnonzero(datasets == dataset)[0]]
        rows.append(f"{index:02d}  {dataset}  [{family}]")
    columns = 3 if len(rows) > 30 else 2
    chunk = (len(rows) + columns - 1) // columns
    fig.text(0.02, 0.025, "\n".join(rows[:chunk]), ha="left", va="bottom", fontsize=7.2, family="monospace")
    if columns >= 2:
        fig.text(0.355, 0.025, "\n".join(rows[chunk:2 * chunk]), ha="left", va="bottom", fontsize=7.2, family="monospace")
    if columns >= 3:
        fig.text(0.69, 0.025, "\n".join(rows[2 * chunk:]), ha="left", va="bottom", fontsize=7.2, family="monospace")
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_validation(
    embedding: np.ndarray,
    records: dict[str, list],
    title: str,
    subtitle: str,
    stem: Path,
) -> None:
    classes = np.asarray(records["class_name"])
    datasets = np.asarray(records["dataset"])
    colors = {"Real": "#0072B2", "Fake": "#D62728"}
    markers = {"Real": "o", "Fake": "X"}
    fig = plt.figure(figsize=(17, 13))
    ax = fig.add_axes([0.07, 0.13, 0.88, 0.79])
    for class_name in ("Real", "Fake"):
        mask = classes == class_name
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], s=22, alpha=0.67,
            color=colors[class_name], marker=markers[class_name], edgecolors=colors[class_name],
            linewidths=0.5, rasterized=True, label=class_name,
        )
    subset_order = list(dict.fromkeys(records["dataset"]))
    for index, subset in enumerate(subset_order, start=1):
        for class_name in ("Real", "Fake"):
            mask = (datasets == subset) & (classes == class_name)
            if not mask.any():
                continue
            center = np.median(embedding[mask], axis=0)
            ax.annotate(
                f"{index:02d}{class_name[0]}", center, fontsize=6.2, ha="center", va="center",
                color=colors[class_name], weight="bold",
            )
    ax.set_title(title, fontsize=18, weight="bold", pad=18)
    ax.text(0.5, 1.002, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.14, linewidth=0.5)
    ax.legend(frameon=False)
    mapping_items = [f"{index:02d}={name}" for index, name in enumerate(subset_order, start=1)]
    mapping_lines = ["   ".join(mapping_items[index:index + 7]) for index in range(0, len(mapping_items), 7)]
    fig.text(0.5, 0.025, "\n".join(mapping_lines), ha="center", va="bottom", fontsize=7.5)
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_archive(path: Path, weights: np.ndarray, embedding: np.ndarray, records: dict[str, list]) -> None:
    np.savez_compressed(
        path,
        routing_weights=weights,
        tsne_embedding=embedding,
        paths=np.asarray(records["paths"]),
        datasets=np.asarray(records["dataset"]),
        families=np.asarray(records["family"]),
        classes=np.asarray(records["class_name"]),
    )
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "dataset", "family", "class", "tsne_1", "tsne_2"))
        for index in range(len(records["paths"])):
            writer.writerow(
                (
                    records["paths"][index], records["dataset"][index], records["family"][index],
                    records["class_name"][index], float(embedding[index, 0]), float(embedding[index, 1]),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--treasure-root", type=Path, required=True)
    parser.add_argument("--aigibench-val", type=Path, required=True)
    parser.add_argument("--aigibench-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Plots"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs_pretrained_routing"))
    parser.add_argument("--treasure-samples", type=int, default=50)
    parser.add_argument("--aigibench-samples", type=int, default=50)
    parser.add_argument("--val-samples-per-class", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--perplexity", type=float, default=40.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; loading frozen checkpoint {args.checkpoint}", flush=True)
    processor = AutoImageProcessor.from_pretrained(str(args.dino), trust_remote_code=True)
    model = load_model(args.checkpoint, args.dino, device)

    specifications = [
        (
            "treasure64", collect_treasure(args.treasure_root, args.treasure_samples),
            list(TREASURE_FAMILIES),
            "FLEET pretrained routing weights: 64 Treasure generators",
            "Color = architecture family; numbered marker = median location of one generator",
        ),
        (
            "aigibench_val", collect_aigibench_val(args.aigibench_val, args.val_samples_per_class),
            ["Real", "Fake"],
            "FLEET pretrained routing weights: AIGIBench validation split",
            "Color/marker = real versus fake; labels identify content class (R/F suffix)",
        ),
        (
            "aigibench_test25", collect_aigibench_test(args.aigibench_test, args.aigibench_samples),
            list(AIGIBENCH_FAMILIES),
            "FLEET pretrained routing weights: 25 AIGIBench generator subsets",
            "Color = generator family from the paper table; numbered marker = subset median",
        ),
    ]

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "note": "Silhouette scores are evaluated on original routing weights, not t-SNE coordinates.",
        "splits": {},
    }
    for split_index, (name, records, family_order, title, subtitle) in enumerate(specifications):
        print(f"Extracting {name}: {len(records['paths'])} images", flush=True)
        weights = extract_routing(model, processor, records, device, args.batch_size, args.workers)
        embedding = run_tsne(weights, 42 + split_index, args.perplexity)
        stem = args.output_dir / f"routing_tsne_pretrained_{name}"
        if name == "aigibench_val":
            plot_validation(embedding, records, title, subtitle, stem)
        else:
            plot_grouped(embedding, records, family_order, title, subtitle, stem)
        save_archive(args.cache_dir / f"{name}.npz", weights, embedding, records)
        report["splits"][name] = {
            "samples": len(weights),
            "routing_dimensions": int(weights.shape[1]),
            "datasets": len(set(records["dataset"])),
            "families": len(set(records["family"])),
            "silhouette_by_family_or_class": safe_silhouette(weights, records["family"]),
            "silhouette_by_dataset": safe_silhouette(weights, records["dataset"]),
            "plot_png": str(stem.with_suffix(".png").resolve()),
        }
        print(f"Saved {stem}.png", flush=True)

    report_path = args.output_dir / "routing_pretrained_cluster_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
