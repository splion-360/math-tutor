"""Capture repeated per-example LoRA gradients for a fixed prompt set.
The probe preserves optimizer gradients and persists identified signatures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dynamic_lora.artifact_snapshots import SnapshotStore
from dynamic_lora.gradient_signatures import project_named_gradients
from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key


@dataclass(frozen=True)
class ProbeExample:
    """One identified, tokenized example reserved from optimizer training.

    Attributes:
        record_id: Stable source record ID, used only in diagnostic artifacts.
        tokenized: Model inputs and mask for one supervised Manim example.
    """

    record_id: str
    tokenized: dict[str, list[int]]


def select_probe_indices(
    records: list[dict[str, Any]], *, sample_count: int, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ID-stable probe indices and the remaining training indices.

    Args:
        records: Validated Manim training records.
        sample_count: Number of records reserved for repeated probing.
        seed: Stable selection seed.

    Returns:
        Probe indices in hash order, then training indices in source order.

    Raises:
        ValueError: If the probe would leave no training examples.
    """
    if sample_count <= 0 or sample_count >= len(records):
        raise ValueError("probe sample count must be positive and fewer than the dataset")
    ranked = sorted(
        range(len(records)),
        key=lambda index: (
            hashlib.sha256(f"{seed}:{records[index]['id']}".encode()).digest(),
            str(records[index]["id"]),
        ),
    )
    probe = tuple(ranked[:sample_count])
    excluded = set(probe)
    training = tuple(index for index in range(len(records)) if index not in excluded)
    return probe, training


class FixedPromptSignatureCallback:
    """Capture gradients of the same held-out examples at chosen optimizer steps.

    The probe runs with dropout disabled and uses ``torch.autograd.grad``, leaving
    parameter ``.grad`` values for the optimizer unchanged.
    """

    def __init__(
        self,
        *,
        selected_layers: tuple[str, ...],
        examples: tuple[ProbeExample, ...],
        steps: tuple[int, ...],
        projection_dim: int,
        seed: int,
        artifact_dir: Path,
        data_collator: Any,
        torch_module: Any,
    ) -> None:
        if not selected_layers or not examples or not steps or projection_dim <= 0:
            raise ValueError("fixed prompt signature probe requires modules, examples, and steps")
        self.selected_layers = selected_layers
        self.examples = examples
        self.steps = steps
        self.projection_dim = projection_dim
        self.seed = seed
        self.data_collator = data_collator
        self.torch_module = torch_module
        self._store = SnapshotStore(artifact_dir, name="fixed_prompt_signatures")
        self._captured_steps: set[int] = set()
        self._snapshot_count = 0

    @property
    def snapshot_path(self) -> Path:
        """Return the JSONL path for identified per-example signatures."""
        return self._store.snapshot_path

    def on_train_begin(self, _args: object, state: Any, control: Any, **kwargs: Any) -> Any:
        """Capture the untrained adapter before its first optimizer update."""
        if int(state.global_step) == 0 and 0 in self.steps:
            self._capture(0, kwargs.get("model"))
        return control

    def on_step_end(self, _args: object, state: Any, control: Any, **kwargs: Any) -> Any:
        """Capture identified gradients after a requested optimizer update."""
        step = int(state.global_step)
        if step in self.steps:
            self._capture(step, kwargs.get("model"))
        return control

    def _capture(self, step: int, model: Any) -> None:
        if step in self._captured_steps:
            return
        if model is None:
            raise RuntimeError("fixed prompt signature callback requires the trainer model")
        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if is_lora_parameter(name) and lora_layer_key(name) in self.selected_layers
        ]
        if not named_parameters:
            raise RuntimeError("selected LoRA modules have no trainable parameters")
        names = tuple(name for name, _ in named_parameters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        was_training = model.training
        model.eval()
        records: list[dict[str, Any]] = []
        try:
            with self.torch_module.enable_grad():
                for example in self.examples:
                    batch = {
                        key: tensor.to(model.device)
                        for key, tensor in self.data_collator([example.tokenized]).items()
                    }
                    loss = model(**batch).loss
                    gradients = self.torch_module.autograd.grad(loss, parameters, allow_unused=True)
                    signatures = project_named_gradients(
                        zip(names, gradients, strict=True),
                        selected_layers=self.selected_layers,
                        projection_dim=self.projection_dim,
                        seed=self.seed,
                    )
                    if set(signatures) != set(self.selected_layers):
                        raise RuntimeError(
                            "fixed prompt probe is missing selected module gradients"
                        )
                    records.extend(
                        {
                            "step": step,
                            "record_id": example.record_id,
                            "layer": layer,
                            "signature": signatures[layer],
                        }
                        for layer in self.selected_layers
                    )
        finally:
            model.train(was_training)
        self._store.append(records)
        self._snapshot_count += len(records)
        self._captured_steps.add(step)

    def finalize(self) -> dict[str, Any]:
        """Persist the capture manifest.

        Returns:
            Artifact paths, captured checkpoints, and probe provenance.
        """
        prefix = self.snapshot_path.parent.name
        summary = {
            "enabled": True,
            "projection": "deterministic_signed_bucket_projection",
            "projection_dim": self.projection_dim,
            "selected_layers": list(self.selected_layers),
            "sample_ids": [example.record_id for example in self.examples],
            "requested_steps": list(self.steps),
            "captured_steps": sorted(self._captured_steps),
            "snapshot_count": self._snapshot_count,
            "snapshot_path": f"{prefix}/{self.snapshot_path.name}",
            "manifest_path": f"{prefix}/{self._store.manifest_path.name}",
            "model_mode": "eval_during_probe",
            "optimizer_gradients_modified": False,
        }
        self._store.write_manifest(summary)
        return summary


def build_fixed_prompt_signature_callback(
    *,
    selected_layers: tuple[str, ...],
    examples: tuple[ProbeExample, ...],
    steps: tuple[int, ...],
    projection_dim: int,
    seed: int,
    artifact_dir: Path,
    data_collator: Any,
    torch_module: Any,
    callback_base: type[Any],
) -> FixedPromptSignatureCallback:
    """Build a Transformers callback for repeated identified gradients.

    Args:
        selected_layers: Layer-module keys measured at each checkpoint.
        examples: Fixed, identified probe examples excluded from optimizer training.
        steps: Optimizer-step numbers at which to capture gradients.
        projection_dim: Width of each deterministic projected vector.
        seed: Stable projection seed.
        artifact_dir: Directory for the JSONL snapshots and manifest.
        data_collator: Single-example model batch builder.
        torch_module: Torch module used for isolated autograd calculations.
        callback_base: Installed Transformers callback superclass.

    Returns:
        Callback compatible with the installed Transformers trainer.
    """
    callback_type = type(
        "TransformersFixedPromptSignatureCallback",
        (FixedPromptSignatureCallback, callback_base),
        {},
    )
    return cast(
        FixedPromptSignatureCallback,
        callback_type(
            selected_layers=selected_layers,
            examples=examples,
            steps=steps,
            projection_dim=projection_dim,
            seed=seed,
            artifact_dir=artifact_dir,
            data_collator=data_collator,
            torch_module=torch_module,
        ),
    )
