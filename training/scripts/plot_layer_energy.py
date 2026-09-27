"""Plot layer energy from a dynamic-LoRA run metadata file.
The figures compare raw category means, parameter-adjusted means, and relative layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def plot_layer_energy(metadata_path: Path, output_path: Path) -> None:
    """Save a three-panel diagnostic plot for a completed gradient probe.

    Args:
        metadata_path: JSON metadata produced by the training run.
        output_path: PNG or SVG path for the plot.

    Raises:
        ValueError: If the metadata has no enabled gradient probe.
    """
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    probe = metadata.get("layer_energy_probe", {})
    if not probe.get("enabled"):
        raise ValueError("metadata has no enabled layer energy probe")

    categories = [category for category in CATEGORIES if category in probe["category_mean_energy"]]
    raw = [probe["category_mean_energy"][category] for category in categories]
    per_parameter = [
        probe["category_mean_energy_per_parameter"][category] * 1_000_000
        for category in categories
    ]
    relative = probe["relative_energy_by_layer"]
    layer_numbers = sorted(
        {int(key.split(".", maxsplit=1)[0].removeprefix("layer_")) for key in relative}
    )
    grid = [
        [relative.get(f"layer_{layer}.{category}", float("nan")) for category in categories]
        for layer in layer_numbers
    ]

    figure, axes = plt.subplots(1, 3, figsize=(18, 10), width_ratios=(1.2, 1.2, 2.6))
    labels = [category.removesuffix("_proj") for category in categories]
    axes[0].barh(labels, raw, color="#55799a")
    axes[0].set_title("Mean gradient energy")
    axes[0].set_xlabel("Mean squared norm per module")
    axes[0].invert_yaxis()

    axes[1].barh(labels, per_parameter, color="#7c9787")
    axes[1].set_title("Energy per LoRA parameter")
    axes[1].set_xlabel("Mean squared norm ÷ parameter count (×10⁻⁶)")
    axes[1].invert_yaxis()

    image = axes[2].imshow(grid, aspect="auto", cmap="Blues", norm=PowerNorm(gamma=0.4))
    axes[2].set_title("Energy relative to category mean")
    axes[2].set_xlabel("Module category")
    axes[2].set_ylabel("Transformer layer")
    axes[2].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[2].set_yticks(range(len(layer_numbers)), [str(layer) for layer in layer_numbers])
    figure.colorbar(image, ax=axes[2], label="× category mean")

    figure.suptitle(f"LoRA gradient probe · {probe['sample_count']} prompts", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Parse CLI paths and write the layer energy figure."""
    parser = argparse.ArgumentParser(description="Plot gradient probe diagnostics")
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    plot_layer_energy(args.metadata_path, args.output_path)


if __name__ == "__main__":
    main()
