"""Render held-out cosine heatmaps for every LoRA layer and projection.
The figures show direct example-pair conflict rates and subject-pair mean cosines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynamic_lora.full_corpus_run import PROJECTION_CATEGORIES


def plot_cosines(summary_path: Path, output_dir: Path) -> None:
    """Render overview and all 36 subject-pair layer heatmaps.

    Args:
        summary_path: Exact-cosine summary downloaded from Modal.
        output_dir: Destination for PNG figures.

    Raises:
        ValueError: If expected layer and module coverage is incomplete.
    """
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    layers = [f"layer_{index}" for index in range(36)]
    subjects = sorted(summary["subject_counts"])
    if set(summary["layers"]) != set(layers) or len(summary["modules"]) != 36 * 7:
        raise ValueError("cosine summary must cover 36 layers and seven projections each")
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_rates = np.asarray(
        [
            [summary["layers"][layer]["subject_negative_rate"][subject] for subject in subjects]
            for layer in layers
        ]
    )
    fig, axis = plt.subplots(figsize=(17, 12), layout="constrained")
    image = axis.imshow(subject_rates, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(
        np.arange(len(subjects)),
        [f"{subject}\n(n={summary['subject_counts'][subject]})" for subject in subjects],
        rotation=70,
        ha="right",
    )
    axis.set_yticks(np.arange(36), [str(index) for index in range(36)])
    axis.set_xlabel("Validation subject")
    axis.set_ylabel("Qwen transformer layer")
    axis.set_title(
        "Fraction of negative LoRA gradient cosines with other subjects\n"
        "Epoch-2 checkpoint · held-out examples · all seven projections combined"
    )
    fig.colorbar(image, ax=axis, label="Fraction of cross-subject example pairs with cosine < 0")
    fig.savefig(output_dir / "all-layers-subject-conflict.png", dpi=180)
    plt.close(fig)

    module_rates = np.asarray(
        [
            [
                summary["modules"][f"{layer}.{module}"]["cross_subject_negative_rate"]
                for module in PROJECTION_CATEGORIES
            ]
            for layer in layers
        ]
    )
    fig, axis = plt.subplots(figsize=(10, 12), layout="constrained")
    image = axis.imshow(module_rates, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(7), PROJECTION_CATEGORIES, rotation=45, ha="right")
    axis.set_yticks(np.arange(36), [str(index) for index in range(36)])
    axis.set_xlabel("LoRA host projection")
    axis.set_ylabel("Qwen transformer layer")
    axis.set_title("Negative cross-subject cosine rate by LoRA projection")
    fig.colorbar(image, ax=axis, label="Fraction of cross-subject example pairs with cosine < 0")
    fig.savefig(output_dir / "all-layers-projection-conflict.png", dpi=180)
    plt.close(fig)

    for layer in layers:
        _plot_pair_heatmap(
            summary["layers"][layer]["subject_pair_mean_cosine"],
            subjects=subjects,
            counts=summary["subject_counts"],
            title=f"{layer}: subject-pair mean gradient cosine",
            output_path=output_dir / f"{layer}-subject-pairs.png",
        )
    _plot_pair_heatmap(
        summary["modules"]["layer_35.down_proj"]["subject_pair_mean_cosine"],
        subjects=subjects,
        counts=summary["subject_counts"],
        title="Layer 35 down projection: subject-pair mean gradient cosine",
        output_path=output_dir / "layer_35-down_proj-subject-pairs.png",
    )


def _plot_pair_heatmap(
    means: dict[str, dict[str, float | None]],
    *,
    subjects: list[str],
    counts: dict[str, int],
    title: str,
    output_path: Path,
) -> None:
    """Render one subject-pair cosine matrix with fixed colors across plots."""
    matrix = np.asarray(
        [[means[left][right] for right in subjects] for left in subjects], dtype=float
    )
    fig, axis = plt.subplots(figsize=(13, 11), layout="constrained")
    palette = plt.get_cmap("coolwarm").copy()
    palette.set_bad("#aaaaaa")
    image = axis.imshow(matrix, cmap=palette, vmin=-0.4, vmax=0.4)
    axis.set_xticks(
        np.arange(len(subjects)),
        [f"{subject} ({counts[subject]})" for subject in subjects],
        rotation=70,
        ha="right",
    )
    axis.set_yticks(np.arange(len(subjects)), subjects)
    axis.set_title(f"{title}\nBlue = opposing · red = aligned · gray diagonal = only one example")
    fig.colorbar(image, ax=axis, label="Mean gradient cosine (clipped to ±0.4 for color)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Read input and destination paths from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    plot_cosines(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
