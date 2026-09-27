"""Test base-gradient probe configuration and reference-run safeguards.
These checks run without downloading a model or contacting Modal or W&B."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_lora.base_probe_run import (
    BaseProbeConfigError,
    load_base_probe_config,
    validate_reference_probe,
)


def test_base_probe_config_resolves_paths_and_rejects_small_sample(tmp_path: Path) -> None:
    path = tmp_path / "base.json"
    raw = {
        "train_path": "train.jsonl",
        "holdout_path": "holdout.jsonl",
        "metadata_path": "base-result.json",
        "reference_metadata_path": "old-result.json",
        "sample_count": 16,
        "top_k": 4,
        "max_seq_length": 1024,
        "target_modules": ["q_proj", "v_proj"],
        "wandb_project": "math-tutor-dynamic-lora",
        "wandb_run_name": "base-probe-16",
        "modal_artifact_path": "modal-volume://dream-ai-training-artifacts/base-probe-16",
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_base_probe_config(path)

    assert config.train_path == tmp_path / "train.jsonl"
    assert config.sample_count == 16
    assert config.target_modules == ("q_proj", "v_proj")

    raw["sample_count"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BaseProbeConfigError, match="sample_count"):
        load_base_probe_config(path)


def test_reference_probe_requires_matching_model_and_training_data() -> None:
    reference = {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "revision-1",
        "dataset": {
            "training_content_sha256": "train-hash",
            "holdout_content_sha256": "holdout-hash",
        },
        "layer_energy_probe": {"selected_layers": ["layer_7.q_proj"]},
    }

    assert validate_reference_probe(
        reference,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="revision-1",
        training_content_sha256="train-hash",
        holdout_content_sha256="holdout-hash",
    ) == ["layer_7.q_proj"]

    with pytest.raises(ValueError, match="training dataset"):
        validate_reference_probe(
            reference,
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            model_revision="revision-1",
            training_content_sha256="different",
            holdout_content_sha256="holdout-hash",
        )
