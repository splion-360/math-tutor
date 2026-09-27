"""Render the held-out subject-by-layer LoRA gradient-norm heatmap.
The chart marks validation sample counts and uses a logarithmic color scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(summary_path: Path, output_path: Path) -> None:
    """Plot the exact mean gradient norms saved by the Modal validation probe.

    Args:
        summary_path: Aggregated JSON from the checkpoint probe.
        output_path: Destination PNG path.
    """
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    layers = summary["layers"]
    subjects = list(summary["subjects"])
    counts = [summary["subjects"][subject]["count"] for subject in subjects]
    norms = np.asarray(
        [
            [summary["subjects"][subject]["mean_layer_norms"][layer] for subject in subjects]
            for layer in layers
        ],
        dtype=float,
    )
    if norms.shape != (36, len(subjects)) or not np.isfinite(norms).all():
        raise ValueError("expected finite values for all 36 layers and observed subjects")
    fig, axis = plt.subplots(figsize=(max(14, len(subjects) * 0.8), 12), layout="constrained")
    image = axis.imshow(np.log10(norms + 1e-12), aspect="auto", cmap="viridis")
    axis.set_xticks(
        np.arange(len(subjects)),
        [f"{name}\n(n={count})" for name, count in zip(subjects, counts, strict=True)],
        rotation=70,
        ha="right",
    )
    axis.set_yticks(np.arange(36), [layer.removeprefix("layer_") for layer in layers])
    axis.set_xlabel("Subject (sample count)")
    axis.set_ylabel("Qwen transformer layer")
    source = summary["gradient_source"]
    if source == "unadapted_base_transformer_layer_weights":
        title = "Base-weight gradients · unadapted Qwen · no optimizer updates"
    elif source == "exact_L2_of_LoRA_A_and_B_gradients_all_seven_projections":
        title = "LoRA A/B gradients · epoch-2 checkpoint · seven projections per layer"
    else:
        raise ValueError(f"unknown gradient source: {source}")
    axis.set_title(f"Mean per-example gradient norm by subject and layer\n{title}")
    fig.colorbar(image, ax=axis, label="log₁₀(mean gradient L₂ norm + 10⁻¹²)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Read CLI paths and render the saved validation summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_heatmap(args.summary, args.output)


if __name__ == "__main__":
    main()
