"""Test deterministic prompt reservation and repeated gradient capture.
The callback tests enforce prompt identity and optimizer-gradient isolation."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dynamic_lora.fixed_prompt_signatures import (
    ProbeExample,
    build_fixed_prompt_signature_callback,
    select_probe_indices,
)


def test_probe_indices_are_stable_by_id_and_excluded_from_training() -> None:
    records = [{"id": item} for item in ("alpha", "beta", "gamma", "delta", "epsilon")]
    selected, training = select_probe_indices(records, sample_count=2, seed=42)
    reversed_selected, _ = select_probe_indices(list(reversed(records)), sample_count=2, seed=42)

    assert len(selected) == 2
    assert {records[index]["id"] for index in selected} == {
        list(reversed(records))[index]["id"] for index in reversed_selected
    }
    assert set(selected).isdisjoint(training)
    assert set(selected) | set(training) == set(range(5))


def test_probe_cannot_consume_all_training_examples() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        select_probe_indices([{"id": "one"}, {"id": "two"}], sample_count=2, seed=42)


class FakeGradient:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def detach(self) -> FakeGradient:
        return self

    def float(self) -> FakeGradient:
        return self

    def reshape(self, _shape: int) -> FakeGradient:
        return self

    def cpu(self) -> FakeGradient:
        return self

    def tolist(self) -> list[float]:
        return self.values


class FakeTensor:
    def to(self, _device: str) -> FakeTensor:
        return self


class FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.training = True
        self.parameter = SimpleNamespace(grad="optimizer-gradient")

    def named_parameters(self) -> list[tuple[str, Any]]:
        return [
            (
                "base_model.model.model.layers.6.mlp.down_proj.lora_B.default.weight",
                self.parameter,
            )
        ]

    def eval(self) -> FakeModel:
        self.training = False
        return self

    def train(self, mode: bool = True) -> FakeModel:
        self.training = mode
        return self

    def __call__(self, **_batch: object) -> SimpleNamespace:
        assert self.training is False
        return SimpleNamespace(loss=object())


def test_probe_captures_same_ids_at_checkpoints_without_touching_optimizer_gradients(
    tmp_path: Path,
) -> None:
    model = FakeModel()
    torch_module = SimpleNamespace(
        enable_grad=nullcontext,
        autograd=SimpleNamespace(
            grad=lambda _loss, _parameters, **_kwargs: (FakeGradient([1.0, 2.0]),)
        ),
    )
    callback = build_fixed_prompt_signature_callback(
        selected_layers=("layer_6.down_proj",),
        examples=(
            ProbeExample("alpha", {"input_ids": [1, 2]}),
            ProbeExample("beta", {"input_ids": [3, 4]}),
        ),
        steps=(0, 2, 4),
        projection_dim=8,
        seed=42,
        artifact_dir=tmp_path,
        data_collator=lambda _rows: {"input_ids": FakeTensor()},
        torch_module=torch_module,
        callback_base=object,
    )
    control = object()

    assert (
        callback.on_train_begin(None, SimpleNamespace(global_step=0), control, model=model)
        is control
    )
    for step in range(1, 5):
        assert (
            callback.on_step_end(None, SimpleNamespace(global_step=step), control, model=model)
            is control
        )

    rows = [json.loads(line) for line in callback.snapshot_path.read_text().splitlines()]
    assert {(row["step"], row["record_id"]) for row in rows} == {
        (step, record_id) for step in (0, 2, 4) for record_id in ("alpha", "beta")
    }
    assert all(row["layer"] == "layer_6.down_proj" and len(row["signature"]) == 8 for row in rows)
    assert model.parameter.grad == "optimizer-gradient"
    assert model.training is True
    assert callback.finalize()["snapshot_count"] == 6
