"""Measure LoRA gradient energy for the shared LoRA training path.
This module ranks LoRA layer/module keys so later training stages can choose observed layers."""

from __future__ import annotations

import re
from collections import defaultdict
from importlib import import_module
from typing import Any, TypedDict

LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


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
    """Measure squared gradient norm per LoRA-bearing layer on one probe batch."""
    if top_k <= 0:
        return _disabled_probe("top_k_not_positive")

    batch = _probe_batch(tokenized_dataset, data_collator, sample_count)
    if batch is None:
        return _disabled_probe("empty_dataset")

    _zero_grad(model)
    outputs = model(**_move_to_model_device(batch, model))
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    loss.backward()

    energy_by_layer: dict[str, float] = defaultdict(float)
    observed_parameters = 0
    for name, parameter in model.named_parameters():
        gradient = getattr(parameter, "grad", None)
        if gradient is None or "lora_" not in name:
            continue
        observed_parameters += 1
        energy_by_layer[_layer_key(name)] += _squared_norm(gradient)

    _zero_grad(model)
    ranked_layer_items = sorted(
        energy_by_layer.items(),
        key=lambda item: (-item[1], item[0]),
    )
    selected_layers = [layer for layer, _energy in ranked_layer_items[:top_k]]
    ranked_layers: list[RankedLayer] = [
        {"layer": layer, "gradient_energy": energy}
        for layer, energy in ranked_layer_items
    ]
    return {
        "enabled": True,
        "sample_count": min(sample_count, len(tokenized_dataset)),
        "top_k": top_k,
        "grouping": "transformer_layer_from_lora_parameter_name",
        "observed_lora_parameters": observed_parameters,
        "energy_by_layer": dict(sorted(energy_by_layer.items())),
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
    return metrics


def _disabled_probe(reason: str) -> LayerEnergyProbe:
    return {
        "enabled": False,
        "reason": reason,
        "energy_by_layer": {},
        "selected_layers": [],
    }


def _probe_batch(tokenized_dataset: Any, data_collator: Any, sample_count: int) -> Any | None:
    size = min(sample_count, len(tokenized_dataset))
    if size <= 0:
        return None
    examples = [tokenized_dataset[index] for index in range(size)]
    return data_collator(examples)


def _move_to_model_device(batch: Any, model: Any) -> Any:
    device = _model_device(model)
    if device is None:
        return batch
    if hasattr(batch, "to"):
        return batch.to(device)
    if not hasattr(batch, "items"):
        return batch
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _model_device(model: Any) -> Any | None:
    embedding_device = _input_embedding_device(model)
    if embedding_device is not None:
        return embedding_device
    first_device = None
    try:
        parameters = model.parameters()
    except (StopIteration, TypeError, AttributeError):
        return None
    for parameter in parameters:
        device = getattr(parameter, "device", None)
        if device is None:
            continue
        if first_device is None:
            first_device = device
        if str(device) != "cpu":
            return device
    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "cpu":
        return model_device
    return _cuda_device() or model_device or first_device


def _input_embedding_device(model: Any) -> Any | None:
    if not hasattr(model, "get_input_embeddings"):
        return None
    embeddings = model.get_input_embeddings()
    return getattr(getattr(embeddings, "weight", None), "device", None)


def _cuda_device() -> Any | None:
    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    return torch.device("cuda:0")


def _zero_grad(model: Any) -> None:
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()


def _squared_norm(gradient: Any) -> float:
    return float(gradient.detach().float().pow(2).sum().item())


def _layer_key(parameter_name: str) -> str:
    layer_match = LAYER_PATTERN.search(parameter_name)
    module_name = _module_name(parameter_name)
    if layer_match is None:
        return module_name
    return f"layer_{layer_match.group(1)}.{module_name}"


def _module_name(parameter_name: str) -> str:
    before_lora = parameter_name.split(".lora_", maxsplit=1)[0]
    return before_lora.rsplit(".", maxsplit=1)[-1]
