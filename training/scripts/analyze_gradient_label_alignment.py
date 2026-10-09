"""Run the recovered projected-gradient label-alignment analysis.
The command writes compact evidence and publication-ready SVG summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynamic_lora.gradient_label_alignment import (
    analyze_gradient_label_alignment,
    file_sha256,
    load_record_labels,
    load_signature_modules,
)


def _summarize_exact_reference(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    layer_rates = [float(row["cross_subject_negative_rate"]) for row in summary["layers"].values()]
    module_rates = [
        float(row["cross_subject_negative_rate"]) for row in summary["modules"].values()
    ]
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "gradient_source": summary["gradient_source"],
        "validation_count": int(summary["validation_count"]),
        "subject_count": len(summary["subject_counts"]),
        "layer_count": len(summary["layers"]),
        "module_count": len(summary["modules"]),
        "layer_cross_subject_negative_rate_range": [min(layer_rates), max(layer_rates)],
        "module_cross_subject_negative_rate_range": [min(module_rates), max(module_rates)],
        "limitation": (
            "The recovered exact artifact contains aggregated subject-pair statistics, "
            "not per-example vectors, so it cannot support the label-association tests."
        ),
    }


def _strip_svg_trailing_space(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _plot_results(summary: dict[str, Any], output_directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modules = summary["results"]["modules"]
    names = list(modules)
    labels = [name.replace("layer_", "L").replace("_proj", "") for name in names]
    outputs: list[str] = []

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, label_name, color in zip(
        axes, ("subject", "difficulty"), ("#3B82F6", "#F97316"), strict=True
    ):
        effects = [modules[name][label_name]["record_mean_contrast"] for name in names]
        intervals = [modules[name][label_name]["record_bootstrap_interval_95"] for name in names]
        lower = [effect - interval[0] for effect, interval in zip(effects, intervals, strict=True)]
        upper = [interval[1] - effect for effect, interval in zip(effects, intervals, strict=True)]
        axis.errorbar(range(len(names)), effects, yerr=[lower, upper], fmt="o", color=color)
        axis.axhline(0, color="#334155", linewidth=1)
        axis.set_ylabel(f"{label_name.title()}\nmean cosine contrast")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(len(names)), labels, rotation=65, ha="right", fontsize=8)
    figure.suptitle("Within-label versus across-label projected-gradient alignment")
    effect_path = output_directory / "gradient_label_effects.svg"
    figure.savefig(effect_path)
    plt.close(figure)
    _strip_svg_trailing_space(effect_path)
    outputs.append(effect_path.name)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, metric in zip(axes, ("ari", "nmi"), strict=True):
        width = 0.38
        positions = list(range(len(names)))
        subject = [modules[name]["subject"]["clustering"][metric] for name in names]
        difficulty = [modules[name]["difficulty"]["clustering"][metric] for name in names]
        axis.bar([position - width / 2 for position in positions], subject, width, label="Subject")
        axis.bar(
            [position + width / 2 for position in positions],
            difficulty,
            width,
            label="Code-length band",
        )
        axis.axhline(0, color="#334155", linewidth=1)
        axis.set_ylabel(metric.upper())
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[-1].set_xticks(range(len(names)), labels, rotation=65, ha="right", fontsize=8)
    figure.suptitle("Spherical k-means agreement with corpus labels")
    cluster_path = output_directory / "gradient_label_clustering.svg"
    figure.savefig(cluster_path)
    plt.close(figure)
    _strip_svg_trailing_space(cluster_path)
    outputs.append(cluster_path.name)
    return outputs


def main() -> None:
    """Validate recovered inputs, run the analysis, and write its evidence bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--signatures", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--exact-summary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--permutation-repeats", type=int, default=2_000)
    parser.add_argument("--cluster-starts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = load_record_labels(args.records)
    modules, checkpoint_step = load_signature_modules(args.signatures, labels)
    results = analyze_gradient_label_alignment(
        modules,
        labels,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        cluster_starts=args.cluster_starts,
        seed=args.seed,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "signatures": {
                "path": str(args.signatures),
                "sha256": file_sha256(args.signatures),
                "kind": "projected_per-example_gradients",
                "checkpoint_step": checkpoint_step,
            },
            "records": {
                "path": str(args.records),
                "sha256": file_sha256(args.records),
                "difficulty_definition": "target-code-line-count-tertiles",
            },
            "exact_held_out_reference": _summarize_exact_reference(args.exact_summary),
        },
        "scope": {
            "primary_question": (
                "association of projected gradients with subject or difficulty labels"
            ),
            "difficulty_is_proxy": True,
            "signatures_are_exact_gradients": False,
            "projected_stream_is_complete_training_corpus": False,
            "exact_reference_role": "conflict-prevalence context only",
            "layer_35_is_post_hoc_illustration_only": True,
            "bootstrap_interval": "record-contrast nonparametric bootstrap",
            "bootstrap_limitation": (
                "Record contrasts share pairwise cosine terms; intervals are descriptive."
            ),
            "multiple_testing_family": "28 modules adjusted separately for each label axis",
        },
        "results": results,
    }
    summary["figures"] = _plot_results(summary, args.output_directory)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
