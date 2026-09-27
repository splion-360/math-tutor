"""Check base-probe signature selection and provenance validation.
These tests keep the diagnostic scope tied to the measured model and dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_lora.signature_selection import select_base_probe_modules


def reference_metadata() -> dict[str, object]:
    return {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "1b4199c4f36b0cef378bfb12390c18780c18af4c",
        "gradient_source": "base_weights",
        "optimizer_steps": 0,
        "prompt_includes_difficulty": False,
        "probe_config": {"max_seq_length": 3072, "seed": 42},
        "dataset": {
            "training_content_sha256": "train-hash",
            "holdout_content_sha256": "holdout-hash",
        },
        "probe": {"selected_modules": ["layer_6.down_proj", "layer_34.q_proj", "layer_22.v_proj"]},
    }


def test_selects_base_probe_modules_for_matching_run(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(reference_metadata()), encoding="utf-8")

    selected = select_base_probe_modules(
        path,
        train_hash="train-hash",
        holdout_hash="holdout-hash",
        max_seq_length=3072,
        seed=42,
        target_modules=("q_proj", "v_proj", "down_proj"),
    )

    assert selected == ("layer_6.down_proj", "layer_34.q_proj", "layer_22.v_proj")


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"gradient_source": "lora_weights"}, "base-weight"),
        ({"probe_config": {"max_seq_length": 3072, "seed": 7}}, "seed"),
        ({"probe": {"selected_modules": ["layer_6.k_proj"]}}, "target modules"),
        ({"probe": {"selected_modules": []}}, "selected modules"),
    ],
)
def test_rejects_incompatible_base_probe(
    tmp_path: Path, change: dict[str, object], error: str
) -> None:
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({**reference_metadata(), **change}), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        select_base_probe_modules(
            path,
            train_hash="train-hash",
            holdout_hash="holdout-hash",
            max_seq_length=3072,
            seed=42,
            target_modules=("q_proj", "v_proj", "down_proj"),
        )
