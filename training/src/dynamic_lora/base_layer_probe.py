"""Rank base and matched-LoRA layers using sampled-subset gradients.
The base pass includes all transformer-layer weights and never updates them."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Literal, TypedDict

from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key
from dynamic_lora.probe_support import (
    move_to_model_device,
    probe_indices,
    squared_gradient_norm,
    zero_model_grad,
)

BASE_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
BASE_MODULE_PATTERN = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight$"
)
GradientSource = Literal["base_weights", "lora_parameters"]


class BaseLayerProbeResult(TypedDict):
    """Subset-gradient energy and rankings for whole layers and eligible modules."""

    gradient_source: GradientSource
    sample_indices: list[int]
    module_energy: dict[str, float]
    module_energy_per_parameter: dict[str, float]
    layer_energy: dict[str, float]
    selected_modules: list[str]
    selected_layers: list[str]


class ProbeComparison(TypedDict):
    """Top-module agreement between a base-gradient and LoRA-gradient probe."""

    base_top_modules: list[str]
    lora_top_modules: list[str]
    overlap: list[str]
    base_only: list[str]
    lora_only: list[str]


def measure_base_layer_gradient_energy(
    *,
    model: Any,
    tokenized_dataset: Any,
    data_collator: Any,
    sample_count: int,
    top_k: int,
    target_modules: tuple[str, ...],
) -> BaseLayerProbeResult:
    """Rank base-weight gradients accumulated over one sampled subset.

    Args:
        model: Unadapted model with differentiable base weights.
        tokenized_dataset: Indexable collection of supervised probe examples.
        data_collator: Converts one example to model inputs with labels.
        sample_count: Number of evenly spaced examples in the subset.
        top_k: Number of whole-layer and module keys to select.
        target_modules: Projection names eligible for later LoRA allocation.

    Returns:
        Squared norm of the mean subset gradient for each complete transformer
        layer and each LoRA-eligible projection module. Squaring preserves the
        layer order induced by D-MoLE's L2 norm score.

    Raises:
        ValueError: If the sample or ranking count is non-positive.
        RuntimeError: If no eligible gradients are produced.
    """
    return _measure_subset_gradient_energy(
        model=model,
        tokenized_dataset=tokenized_dataset,
        data_collator=data_collator,
        sample_count=sample_count,
        top_k=top_k,
        target_modules=target_modules,
        source="base_weights",
    )


def measure_lora_subset_gradient_energy(
    *,
    model: Any,
    tokenized_dataset: Any,
    data_collator: Any,
    sample_count: int,
    top_k: int,
    target_modules: tuple[str, ...],
) -> BaseLayerProbeResult:
    """Rank LoRA parameter gradients with the same subset score as the base probe.

    Args:
        model: Model with a freshly attached, untrained LoRA adapter.
        tokenized_dataset: Same supervised examples used by the base probe.
        data_collator: Same collator used by the base probe.
        sample_count: Number of evenly spaced examples in the subset.
        top_k: Number of whole-layer and module keys to select.
        target_modules: Projection names eligible for LoRA allocation.

    Returns:
        Squared norm of the mean subset gradient for LoRA modules and layers.
    """
    return _measure_subset_gradient_energy(
        model=model,
        tokenized_dataset=tokenized_dataset,
        data_collator=data_collator,
        sample_count=sample_count,
        top_k=top_k,
        target_modules=target_modules,
        source="lora_parameters",
    )


def _measure_subset_gradient_energy(
    *,
    model: Any,
    tokenized_dataset: Any,
    data_collator: Any,
    sample_count: int,
    top_k: int,
    target_modules: tuple[str, ...],
    source: GradientSource,
) -> BaseLayerProbeResult:
    if sample_count <= 0 or top_k <= 0:
        raise ValueError("sample_count and top_k must be positive")
    indices = probe_indices(len(tokenized_dataset), sample_count)
    if not indices:
        raise ValueError("tokenized_dataset must not be empty")

    was_training = getattr(model, "training", None)
    model.eval()
    try:
        zero_model_grad(model)
        for index in indices:
            batch = data_collator([tokenized_dataset[index]])
            outputs = model(**move_to_model_device(batch, model))
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()

        module_energy: dict[str, float] = defaultdict(float)
        layer_energy: dict[str, float] = defaultdict(float)
        parameter_counts: dict[str, int] = defaultdict(int)
        scale = len(indices) ** 2
        for name, parameter in model.named_parameters():
            gradient = getattr(parameter, "grad", None)
            if gradient is None:
                continue
            keys = _parameter_keys(name, source=source, target_modules=target_modules)
            if keys is None:
                continue
            layer_key, module_key = keys
            energy = squared_gradient_norm(gradient) / scale
            layer_energy[layer_key] += energy
            if module_key is not None:
                module_energy[module_key] += energy
                parameter_counts[module_key] += parameter.numel()
    finally:
        zero_model_grad(model)
        if isinstance(was_training, bool):
            model.train(was_training)

    if not module_energy:
        raise RuntimeError("no LoRA-eligible gradients were observed")
    return {
        "gradient_source": source,
        "sample_indices": indices,
        "module_energy": dict(sorted(module_energy.items())),
        "module_energy_per_parameter": {
            key: energy / parameter_counts[key] for key, energy in sorted(module_energy.items())
        },
        "layer_energy": dict(sorted(layer_energy.items())),
        "selected_modules": _top_k(module_energy, top_k),
        "selected_layers": _top_k(layer_energy, top_k),
    }


def _parameter_keys(
    name: str, *, source: GradientSource, target_modules: tuple[str, ...]
) -> tuple[str, str | None] | None:
    if source == "lora_parameters":
        if not is_lora_parameter(name):
            return None
        module_key = lora_layer_key(name)
        if module_key.rsplit(".", maxsplit=1)[-1] not in target_modules:
            return None
        return module_key.split(".", maxsplit=1)[0], module_key

    if ".lora_" in name:
        return None
    layer_match = BASE_LAYER_PATTERN.search(name)
    if layer_match is None:
        return None
    layer_key = f"layer_{layer_match.group(1)}"
    module_match = BASE_MODULE_PATTERN.search(name)
    if module_match is None or module_match.group(2) not in target_modules:
        return layer_key, None
    return layer_key, f"{layer_key}.{module_match.group(2)}"


def compare_probe_rankings(
    *, base_modules: list[str], lora_modules: list[str]
) -> ProbeComparison:
    """Compare top module keys while preserving each probe's original ranking."""
    base_set = set(base_modules)
    lora_set = set(lora_modules)
    return {
        "base_top_modules": list(base_modules),
        "lora_top_modules": list(lora_modules),
        "overlap": [key for key in base_modules if key in lora_set],
        "base_only": [key for key in base_modules if key not in lora_set],
        "lora_only": [key for key in lora_modules if key not in base_set],
    }


def selection_frequency(rankings: list[list[str]]) -> dict[str, float]:
    """Report top-k selection frequency across independently probed subsets."""
    if not rankings:
        raise ValueError("rankings must not be empty")
    counts: dict[str, int] = defaultdict(int)
    for ranking in rankings:
        for key in ranking:
            counts[key] += 1
    return {key: count / len(rankings) for key, count in sorted(counts.items())}


def _top_k(energy: dict[str, float], count: int) -> list[str]:
    ranked = sorted(energy.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _value in ranked[:count]]
