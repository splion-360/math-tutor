"""Measure LoRA gradient energy for the shared LoRA training path.
This module ranks LoRA layer/module keys so later training stages can choose observed layers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict

from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key
from dynamic_lora.probe_support import (
    move_to_model_device,
    probe_indices,
    squared_gradient_norm,
    zero_model_grad,
)


class RankedLayer(TypedDict):
    layer: str
    gradient_energy: float


class LayerEnergyProbe(TypedDict, total=False):
    enabled: bool
    reason: str
    sample_count: int
    top_k: int
    grouping: str
    observed_lora_parameters: int
    energy_by_layer: dict[str, float]
    energy_per_parameter_by_layer: dict[str, float]
    category_mean_energy: dict[str, float]
    category_mean_energy_per_parameter: dict[str, float]
    relative_energy_by_layer: dict[str, float]
    ranked_layers: list[RankedLayer]
    selected_layers: list[str]


def measure_lora_layer_gradient_energy(
    *,
    model: Any,
    tokenized_dataset: Any,
    data_collator: Any,
    sample_count: int,
    top_k: int,
) -> LayerEnergyProbe:
    """Average per-example squared LoRA gradient norms over a spread of prompts.

    Args:
        model: Model with trainable LoRA parameters.
        tokenized_dataset: Dataset of tokenized training prompts.
        data_collator: Collator that accepts a single example at a time.
        sample_count: Maximum number of prompts to probe.
        top_k: Number of layer/module keys selected by raw mean energy.

    Returns:
        Raw rankings and category-relative diagnostics for the sampled prompts.
    """
    if top_k <= 0:
        return _disabled_probe("top_k_not_positive")

    indices = probe_indices(len(tokenized_dataset), sample_count)
    if not indices:
        return _disabled_probe("empty_dataset")

    energy_by_layer: dict[str, float] = defaultdict(float)
    parameter_counts: dict[str, int] = defaultdict(int)
    observed_parameter_names: set[str] = set()
    was_training = getattr(model, "training", None)
    if callable(getattr(model, "eval", None)):
        model.eval()
    try:
        for index in indices:
            batch = data_collator([tokenized_dataset[index]])
            zero_model_grad(model)
            outputs = model(**move_to_model_device(batch, model))
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()
            for name, parameter in model.named_parameters():
                gradient = getattr(parameter, "grad", None)
                if gradient is None or not is_lora_parameter(name):
                    continue
                layer = lora_layer_key(name)
                energy_by_layer[layer] += squared_gradient_norm(gradient)
                if name not in observed_parameter_names:
                    observed_parameter_names.add(name)
                    parameter_counts[layer] += parameter.numel()
    finally:
        zero_model_grad(model)
        if isinstance(was_training, bool) and callable(getattr(model, "train", None)):
            model.train(was_training)

    mean_energy = {layer: energy / len(indices) for layer, energy in energy_by_layer.items()}
    category_totals: dict[str, float] = defaultdict(float)
    category_counts: dict[str, int] = defaultdict(int)
    for layer, energy in mean_energy.items():
        category = layer.rsplit(".", maxsplit=1)[-1]
        category_totals[category] += energy
        category_counts[category] += 1
    category_means = {
        category: total / category_counts[category]
        for category, total in category_totals.items()
    }
    energy_per_parameter = {
        layer: energy / parameter_counts[layer]
        for layer, energy in mean_energy.items()
        if parameter_counts[layer] > 0
    }
    category_per_parameter_totals: dict[str, float] = defaultdict(float)
    category_per_parameter_counts: dict[str, int] = defaultdict(int)
    for layer, energy in energy_per_parameter.items():
        category = layer.rsplit(".", maxsplit=1)[-1]
        category_per_parameter_totals[category] += energy
        category_per_parameter_counts[category] += 1
    ranked_layer_items = sorted(
        mean_energy.items(),
        key=lambda item: (-item[1], item[0]),
    )
    selected_layers = [layer for layer, _energy in ranked_layer_items[:top_k]]
    ranked_layers: list[RankedLayer] = [
        {"layer": layer, "gradient_energy": energy}
        for layer, energy in ranked_layer_items
    ]
    return {
        "enabled": True,
        "sample_count": len(indices),
        "top_k": top_k,
        "grouping": "transformer_layer_from_lora_parameter_name",
        "observed_lora_parameters": len(observed_parameter_names),
        "energy_by_layer": dict(sorted(mean_energy.items())),
        "energy_per_parameter_by_layer": dict(sorted(energy_per_parameter.items())),
        "category_mean_energy": dict(sorted(category_means.items())),
        "category_mean_energy_per_parameter": {
            category: total / category_per_parameter_counts[category]
            for category, total in sorted(category_per_parameter_totals.items())
        },
        "relative_energy_by_layer": {
            layer: (
                energy / category_means[layer.rsplit(".", maxsplit=1)[-1]]
                if category_means[layer.rsplit(".", maxsplit=1)[-1]] > 0
                else 0.0
            )
            for layer, energy in sorted(mean_energy.items())
        },
        "ranked_layers": ranked_layers,
        "selected_layers": selected_layers,
    }


def gradient_probe_metrics(probe: LayerEnergyProbe) -> dict[str, int | float | str]:
    """Flatten probe output into W&B-friendly scalar/string metrics."""
    if not probe.get("enabled"):
        return {"gradient_probe/enabled": 0}
    metrics: dict[str, int | float | str] = {
        "gradient_probe/enabled": 1,
        "gradient_probe/top_k": int(probe["top_k"]),
        "gradient_probe/sample_count": int(probe["sample_count"]),
        "gradient_probe/observed_lora_parameters": int(probe["observed_lora_parameters"]),
        "gradient_probe/selected_layers": ",".join(probe["selected_layers"]),
    }
    for layer, energy in probe["energy_by_layer"].items():
        metrics[f"gradient_probe/energy/{layer}"] = float(energy)
    for category, energy in probe.get("category_mean_energy", {}).items():
        metrics[f"gradient_probe/category_mean_energy/{category}"] = float(energy)
    for category, energy in probe.get("category_mean_energy_per_parameter", {}).items():
        metrics[f"gradient_probe/category_mean_energy_per_parameter/{category}"] = float(energy)
    return metrics


def _disabled_probe(reason: str) -> LayerEnergyProbe:
    return {
        "enabled": False,
        "reason": reason,
        "energy_by_layer": {},
        "energy_per_parameter_by_layer": {},
        "category_mean_energy": {},
        "category_mean_energy_per_parameter": {},
        "relative_energy_by_layer": {},
        "selected_layers": [],
    }
