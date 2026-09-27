"""Share deterministic sampling and gradient utilities across layer probes.
The helpers handle model devices and gradient cleanup without choosing a score."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def probe_indices(dataset_size: int, sample_count: int) -> list[int]:
    """Return evenly spaced dataset positions for a bounded probe.

    Args:
        dataset_size: Number of available records.
        sample_count: Maximum number of records to select.

    Returns:
        Deterministic positions, or an empty list for an empty dataset.
    """
    size = min(sample_count, dataset_size)
    if size <= 0:
        return []
    if size == 1:
        return [dataset_size // 2]
    return [index * (dataset_size - 1) // (size - 1) for index in range(size)]


def move_to_model_device(batch: Any, model: Any) -> Any:
    """Move a collated batch to the model's input-embedding device.

    Args:
        batch: Tensor-like batch or mapping of input tensors.
        model: Model whose embedding location determines the input device.

    Returns:
        Batch with supported tensor values moved to the input device.
    """
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


def zero_model_grad(model: Any) -> None:
    """Clear gradients without retaining previous probe allocations.

    Args:
        model: Model exposing ``zero_grad``.
    """
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()


def squared_gradient_norm(gradient: Any) -> float:
    """Return a gradient's squared L2 norm using float accumulation.

    Args:
        gradient: Tensor-like gradient supporting detach, float, pow, and sum.

    Returns:
        Squared L2 norm as a Python float.
    """
    return float(gradient.detach().float().pow(2).sum().item())


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
