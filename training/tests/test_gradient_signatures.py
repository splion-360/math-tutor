"""Test projected gradient signature capture through the trainer callback interface.
These tests verify snapshots, cosine similarity, and conflict metrics without a real model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from dynamic_lora.gradient_signatures import build_gradient_signature_callback


class FakeGradient:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def detach(self) -> FakeGradient:
        return self

    def float(self) -> FakeGradient:
        return self

    def reshape(self, _size: int) -> FakeGradient:
        return self

    def cpu(self) -> FakeGradient:
        return self

    def tolist(self) -> list[float]:
        return self.values


class FakeParameter:
    def __init__(self, values: list[float]) -> None:
        self.grad = FakeGradient(values)


class FakeModel:
    def __init__(self) -> None:
        self.gradient = [1.0, 2.0, 3.0]

    def named_parameters(self) -> Iterator[tuple[str, FakeParameter]]:
        return iter(
            (
                (
                    "base_model.model.model.layers.7.self_attn.q_proj.lora_A.default.weight",
                    FakeParameter(self.gradient),
                ),
                (
                    "base_model.model.model.layers.3.mlp.down_proj.lora_A.default.weight",
                    FakeParameter([100.0]),
                ),
            )
        )


class FakeTrainerCallback:
    def on_train_begin(
        self,
        _args: object,
        _state: object,
        control: object,
        **_kwargs: object,
    ) -> object:
        return control


def test_callback_saves_projected_signatures_and_reports_gradient_conflict(
    tmp_path: Path,
) -> None:
    logged: list[tuple[int, dict[str, int | float | str]]] = []
    callback = build_gradient_signature_callback(
        selected_layers=("layer_7.q_proj",),
        projection_dim=4,
        every_steps=1,
        seed=42,
        artifact_dir=tmp_path,
        log_metrics=lambda metrics, step: logged.append((step, metrics)),
        callback_base=FakeTrainerCallback,
    )
    model = FakeModel()

    callback.on_pre_optimizer_step(
        None,
        SimpleNamespace(global_step=0),
        object(),
        model=model,
    )
    model.gradient = [-1.0, -2.0, -3.0]
    callback.on_pre_optimizer_step(
        None,
        SimpleNamespace(global_step=1),
        object(),
        model=model,
    )

    snapshots = [
        json.loads(line)
        for line in callback.snapshot_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(snapshots) == 2
    assert snapshots[0]["layer"] == "layer_7.q_proj"
    assert len(snapshots[0]["signature"]) == 4
    assert snapshots[1]["cosine_to_previous"] == -1.0
    assert snapshots[1]["conflict"] is True

    control = object()
    assert callback.on_train_begin(None, object(), control) is control

    summary = callback.finalize()
    assert summary["snapshot_count"] == 2
    assert summary["layers"]["layer_7.q_proj"] == {
        "observations": 2,
        "comparisons": 1,
        "conflicts": 1,
        "conflict_rate": 1.0,
        "mean_cosine": -1.0,
        "minimum_cosine": -1.0,
    }
    assert logged[-1] == (
        2,
        {
            "gradient_signatures/captured_layers": 1,
            "gradient_signatures/snapshot_count": 2,
            "gradient_signatures/layer_7.q_proj/cosine_to_previous": -1.0,
            "gradient_signatures/layer_7.q_proj/conflict": 1,
            "gradient_signatures/layer_7.q_proj/conflict_rate": 1.0,
        },
    )


def test_callback_starts_capture_after_warmup(tmp_path: Path) -> None:
    callback = build_gradient_signature_callback(
        selected_layers=("layer_7.q_proj",),
        projection_dim=4,
        every_steps=1,
        start_step=3,
        seed=42,
        artifact_dir=tmp_path,
        log_metrics=lambda _metrics, _step: None,
        callback_base=FakeTrainerCallback,
    )
    model = FakeModel()
    for global_step in range(4):
        model.gradient = [1.0, 2.0, 3.0] if global_step < 3 else [-1.0, -2.0, -3.0]
        callback.on_pre_optimizer_step(
            None,
            SimpleNamespace(global_step=global_step),
            object(),
            model=model,
        )

    snapshots = [
        json.loads(line)
        for line in callback.snapshot_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [snapshot["step"] for snapshot in snapshots] == [3, 4]
    assert callback.finalize()["layers"]["layer_7.q_proj"]["conflicts"] == 1
