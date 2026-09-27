"""Capture projected LoRA gradients during optimizer steps.
The callback persists signatures and reports cosine-based conflict statistics."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from dynamic_lora.artifact_snapshots import SnapshotStore
from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key

MetricValue = int | float | str
MetricLogger = Callable[[dict[str, MetricValue], int], None]


class LayerStatisticsSummary(TypedDict):
    observations: int
    comparisons: int
    conflicts: int
    conflict_rate: float
    mean_cosine: float | None
    minimum_cosine: float | None


@dataclass
class _LayerStatistics:
    observations: int = 0
    comparisons: int = 0
    conflicts: int = 0
    cosine_sum: float = 0.0
    minimum_cosine: float | None = None

    def observe(self, cosine: float | None) -> None:
        self.observations += 1
        if cosine is None:
            return
        self.comparisons += 1
        self.cosine_sum += cosine
        self.minimum_cosine = (
            cosine if self.minimum_cosine is None else min(self.minimum_cosine, cosine)
        )
        if cosine < 0:
            self.conflicts += 1

    def summary(self) -> LayerStatisticsSummary:
        return {
            "observations": self.observations,
            "comparisons": self.comparisons,
            "conflicts": self.conflicts,
            "conflict_rate": (
                self.conflicts / self.comparisons if self.comparisons else 0.0
            ),
            "mean_cosine": (
                self.cosine_sum / self.comparisons if self.comparisons else None
            ),
            "minimum_cosine": self.minimum_cosine,
        }


class GradientSignatureCallback:
    """Capture selected LoRA layer gradients before each optimizer step."""

    def __init__(
        self,
        *,
        selected_layers: tuple[str, ...],
        projection_dim: int,
        every_steps: int,
        seed: int,
        artifact_dir: Path,
        log_metrics: MetricLogger,
        start_step: int = 1,
    ) -> None:
        if not selected_layers:
            raise ValueError("selected_layers must not be empty")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if every_steps <= 0:
            raise ValueError("every_steps must be positive")
        if start_step <= 0:
            raise ValueError("start_step must be positive")
        self.selected_layers = selected_layers
        self.projection_dim = projection_dim
        self.every_steps = every_steps
        self.start_step = start_step
        self.seed = seed
        self._log_metrics = log_metrics
        self._store = SnapshotStore(artifact_dir, name="gradient_signatures")
        self._previous: dict[str, list[float]] = {}
        self._statistics = {layer: _LayerStatistics() for layer in selected_layers}
        self._snapshot_count = 0

    @property
    def snapshot_path(self) -> Path:
        """Return the JSONL path containing projected gradient signatures."""
        return self._store.snapshot_path

    def on_pre_optimizer_step(
        self,
        _args: object,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        """Capture gradients after accumulation and before the optimizer updates weights."""
        step = int(state.global_step) + 1
        if step < self.start_step or step % self.every_steps != 0:
            return control
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("gradient signature callback requires the trainer model")
        signatures = _project_model_gradients(
            model,
            selected_layers=self.selected_layers,
            projection_dim=self.projection_dim,
            seed=self.seed,
        )
        records: list[dict[str, Any]] = []
        metrics: dict[str, MetricValue] = {
            "gradient_signatures/captured_layers": len(signatures),
        }
        for layer in self.selected_layers:
            signature = signatures.get(layer)
            if signature is None:
                continue
            previous = self._previous.get(layer)
            cosine = _cosine_similarity(previous, signature) if previous is not None else None
            conflict = cosine is not None and cosine < 0
            self._statistics[layer].observe(cosine)
            self._previous[layer] = signature
            self._snapshot_count += 1
            records.append(
                {
                    "step": step,
                    "layer": layer,
                    "signature": signature,
                    "cosine_to_previous": cosine,
                    "conflict": conflict,
                }
            )
            if cosine is not None:
                prefix = f"gradient_signatures/{layer}"
                metrics[f"{prefix}/cosine_to_previous"] = cosine
                metrics[f"{prefix}/conflict"] = int(conflict)
                metrics[f"{prefix}/conflict_rate"] = self._statistics[layer].summary()[
                    "conflict_rate"
                ]
        metrics["gradient_signatures/snapshot_count"] = self._snapshot_count
        self._store.append(records)
        self._log_metrics(metrics, step)
        return control

    def finalize(self) -> dict[str, Any]:
        """Persist and return aggregate signature and conflict statistics."""
        artifact_prefix = self.snapshot_path.parent.name
        summary = {
            "enabled": True,
            "projection": "deterministic_signed_bucket_projection",
            "projection_dim": self.projection_dim,
            "every_steps": self.every_steps,
            "start_step": self.start_step,
            "selected_layers": list(self.selected_layers),
            "snapshot_count": self._snapshot_count,
            "snapshot_path": f"{artifact_prefix}/{self.snapshot_path.name}",
            "manifest_path": f"{artifact_prefix}/{self._store.manifest_path.name}",
            "layers": {
                layer: statistics.summary()
                for layer, statistics in self._statistics.items()
            },
        }
        self._store.write_manifest(summary)
        return summary


def build_gradient_signature_callback(
    *,
    selected_layers: tuple[str, ...],
    projection_dim: int,
    every_steps: int,
    seed: int,
    artifact_dir: Path,
    log_metrics: MetricLogger,
    callback_base: type[Any],
    start_step: int = 1,
) -> GradientSignatureCallback:
    """Build a callback compatible with the installed Transformers version."""
    callback_type = type(
        "TransformersGradientSignatureCallback",
        (GradientSignatureCallback, callback_base),
        {},
    )
    return cast(
        GradientSignatureCallback,
        callback_type(
            selected_layers=selected_layers,
            projection_dim=projection_dim,
            every_steps=every_steps,
            seed=seed,
            artifact_dir=artifact_dir,
            log_metrics=log_metrics,
            start_step=start_step,
        ),
    )


def _project_model_gradients(
    model: Any,
    *,
    selected_layers: tuple[str, ...],
    projection_dim: int,
    seed: int,
) -> dict[str, list[float]]:
    selected = set(selected_layers)
    signatures = {layer: [0.0] * projection_dim for layer in selected_layers}
    offsets = {layer: 0 for layer in selected_layers}
    observed: set[str] = set()
    for parameter_name, parameter in model.named_parameters():
        gradient = getattr(parameter, "grad", None)
        if gradient is None or not is_lora_parameter(parameter_name):
            continue
        layer = lora_layer_key(parameter_name)
        if layer not in selected:
            continue
        values = gradient.detach().float().reshape(-1).cpu().tolist()
        _add_signed_bucket_projection(
            signatures[layer],
            values,
            layer_seed=_layer_seed(seed, layer),
            offset=offsets[layer],
        )
        offsets[layer] += len(values)
        observed.add(layer)
    return {layer: signatures[layer] for layer in selected_layers if layer in observed}


def _add_signed_bucket_projection(
    signature: list[float],
    values: list[float],
    *,
    layer_seed: int,
    offset: int,
) -> None:
    projection_dim = len(signature)
    for local_index, value in enumerate(values):
        index = offset + local_index
        bucket_multiplier = layer_seed | 1
        bucket_hash = (index * bucket_multiplier + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        bucket = (bucket_hash * projection_dim) >> 64
        sign_multiplier = (layer_seed ^ 0xD6E8FEB86659FD93) | 1
        sign_hash = (index * sign_multiplier + 0xA0761D6478BD642F) & 0xFFFFFFFFFFFFFFFF
        sign = -1.0 if sign_hash >> 63 else 1.0
        signature[bucket] += sign * float(value)


def _layer_seed(seed: int, layer: str) -> int:
    digest = hashlib.sha256(f"{seed}:{layer}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    cosine = dot_product / (left_norm * right_norm)
    return round(max(-1.0, min(1.0, cosine)), 12)
