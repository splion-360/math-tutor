"""Measure transformer base-weight gradient energy before adapter allocation.
The probe ranks whole layers and LoRA-eligible modules without updating weights."""

from __future__ import annotations

import re
from collections import defaultdict
from random import Random
from typing import Any, TypedDict

from dynamic_lora.layer_selection import (
    _move_to_model_device,
    _probe_indices,
    _squared_norm,
    _zero_grad,
)

BASE_WEIGHT_PATTERN = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight$"
)


class BaseLayerProbeResult(TypedDict):
    gradient_source: str
    sample_indices: list[int]
    sample_module_energy: list[dict[str, float]]
    module_energy: dict[str, float]
    module_energy_per_parameter: dict[str, float]
    layer_energy: dict[str, float]
    selected_modules: list[str]
    selected_layers: list[str]


class ProbeComparison(TypedDict):
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
    """Rank base weights after backward passes that perform no optimizer updates.

    Args:
        model: Unadapted model whose base parameters permit gradient computation.
        tokenized_dataset: Indexable collection of probe examples.
        data_collator: Converts one example to model inputs with labels.
        sample_count: Number of evenly spaced examples to inspect.
        top_k: Number of module keys and whole-layer keys to select.
        target_modules: LoRA-eligible projection module names.

    Returns:
        Mean squared gradient norms and separate module/whole-layer rankings.

    Raises:
        ValueError: If the requested sample or ranking count is non-positive.
        RuntimeError: If no eligible base-weight gradients are produced.
    """
    if sample_count <= 0 or top_k <= 0:
        raise ValueError("sample_count and top_k must be positive")
    indices = _probe_indices(len(tokenized_dataset), sample_count)
    if not indices:
        raise ValueError("tokenized_dataset must not be empty")

    sample_energies: list[dict[str, float]] = []
    parameter_counts: dict[str, int] = {}
    was_training = getattr(model, "training", None)
    model.eval()
    try:
        for index in indices:
            batch = data_collator([tokenized_dataset[index]])
            _zero_grad(model)
            outputs = model(**_move_to_model_device(batch, model))
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()
            energy: dict[str, float] = defaultdict(float)
            for name, parameter in model.named_parameters():
                key = _base_module_key(name, target_modules)
                if key is None or parameter.grad is None:
                    continue
                energy[key] += _squared_norm(parameter.grad)
                parameter_counts[key] = parameter.numel()
            sample_energies.append(dict(energy))
    finally:
        _zero_grad(model)
        if isinstance(was_training, bool):
            model.train(was_training)

    if not parameter_counts:
        raise RuntimeError("no LoRA-eligible base-weight gradients were observed")
    module_energy = {
        key: sum(sample.get(key, 0.0) for sample in sample_energies) / len(indices)
        for key in sorted(parameter_counts)
    }
    layer_energy: dict[str, float] = defaultdict(float)
    for key, mean_energy in module_energy.items():
        layer_energy[key.split(".", maxsplit=1)[0]] += mean_energy

    return {
        "gradient_source": "base_weights",
        "sample_indices": indices,
        "sample_module_energy": sample_energies,
        "module_energy": module_energy,
        "module_energy_per_parameter": {
            key: energy / parameter_counts[key] for key, energy in module_energy.items()
        },
        "layer_energy": dict(sorted(layer_energy.items())),
        "selected_modules": _top_k(module_energy, top_k),
        "selected_layers": _top_k(layer_energy, top_k),
    }


def _base_module_key(name: str, target_modules: tuple[str, ...]) -> str | None:
    match = BASE_WEIGHT_PATTERN.search(name)
    if match is None or match.group(2) not in target_modules:
        return None
    return f"layer_{match.group(1)}.{match.group(2)}"


def bootstrap_topk_frequency(
    sample_energy: list[dict[str, float]],
    *,
    top_k: int,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """Estimate how often each key enters top-k under sample resampling.

    Args:
        sample_energy: Per-example squared gradient energies by module or layer.
        top_k: Number of highest-energy keys selected in each resample.
        repeats: Number of deterministic bootstrap resamples.
        seed: Random seed used only for resampling.

    Returns:
        Selection frequency for every observed key.

    Raises:
        ValueError: If no samples are supplied or counts are non-positive.
    """
    if not sample_energy or top_k <= 0 or repeats <= 0:
        raise ValueError("samples, top_k, and repeats must be positive")
    keys = sorted({key for sample in sample_energy for key in sample})
    selections = dict.fromkeys(keys, 0)
    rng = Random(seed)
    for _ in range(repeats):
        indices = rng.choices(range(len(sample_energy)), k=len(sample_energy))
        mean = {
            key: sum(sample_energy[index].get(key, 0.0) for index in indices) / len(indices)
            for key in keys
        }
        for key in _top_k(mean, top_k):
            selections[key] += 1
    return {key: count / repeats for key, count in selections.items()}


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


def _top_k(energy: dict[str, float], count: int) -> list[str]:
    ranked = sorted(energy.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _value in ranked[:count]]
