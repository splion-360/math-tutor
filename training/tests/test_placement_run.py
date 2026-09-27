"""Test placement-run configuration and probe provenance checks.
These checks reject mismatched data before loading the Qwen model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_lora.placement_run import load_placement_config, validate_placement_probe


def test_placement_config_requires_bounded_training(tmp_path: Path) -> None:
    config_path = tmp_path / "placement.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "train.jsonl",
                "holdout_path": "holdout.jsonl",
                "probe_metadata_path": "probe.json",
                "artifact_root": "artifacts",
                "seed": 42,
                "max_steps": 64,
                "max_seq_length": 1024,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "learning_rate": 0.0002,
                "wandb_project": "math-tutor-dynamic-lora",
            }
        ),
        encoding="utf-8",
    )

    config = load_placement_config(config_path)

    assert config.max_steps == 64
    assert config.train_path == tmp_path / "train.jsonl"


def test_placement_probe_must_match_frozen_model_and_exact_dataset() -> None:
    reference = {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "1b4199c4f36b0cef378bfb12390c18780c18af4c",
        "gradient_source": "base_weights",
        "optimizer_steps": 0,
        "prompt_includes_difficulty": False,
        "probe_config": {"max_seq_length": 3072},
        "dataset": {
            "training_content_sha256": "train-hash",
            "holdout_content_sha256": "holdout-hash",
        },
        "probe": {"sample_indices": [0, 1], "selected_modules": []},
    }

    validate_placement_probe(
        reference, train_hash="train-hash", holdout_hash="holdout-hash", max_seq_length=3072
    )

    reference["dataset"]["training_content_sha256"] = "other"  # type: ignore[index]
    with pytest.raises(ValueError, match="training dataset"):
        validate_placement_probe(
            reference, train_hash="train-hash", holdout_hash="holdout-hash", max_seq_length=3072
        )

    reference["dataset"]["training_content_sha256"] = "train-hash"  # type: ignore[index]
    reference["prompt_includes_difficulty"] = True
    with pytest.raises(ValueError, match="difficulty"):
        validate_placement_probe(
            reference, train_hash="train-hash", holdout_hash="holdout-hash", max_seq_length=3072
        )

    reference["prompt_includes_difficulty"] = False
    with pytest.raises(ValueError, match="token limit"):
        validate_placement_probe(
            reference, train_hash="train-hash", holdout_hash="holdout-hash", max_seq_length=1024
        )
