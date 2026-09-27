from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_lora.config import ConfigError, load_config


def test_relative_paths_resolve_from_config_directory(tmp_path: Path, monkeypatch: object) -> None:
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "../data/train.jsonl",
                "holdout_path": "../data/holdout.jsonl",
                "output_dir": "../artifacts/adapter",
                "metadata_path": "../artifacts/run.json",
            }
        ),
        encoding="utf-8",
    )
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)  # type: ignore[attr-defined]

    config = load_config(config_path)

    assert config.train_path == config_dir / "../data/train.jsonl"
    assert config.holdout_path == config_dir / "../data/holdout.jsonl"
    assert config.output_dir == config_dir / "../artifacts/adapter"
    assert config.metadata_path == config_dir / "../artifacts/run.json"


def test_tracking_config_is_loaded_from_nested_object(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "train.jsonl",
                "holdout_path": "holdout.jsonl",
                "output_dir": "adapter",
                "metadata_path": "run.json",
                "tracking": {
                    "mode": "online",
                    "project": "math-tutor-dynamic-lora",
                    "run_name": "smoke-run",
                    "tags": ["dynamic-lora", "smoke"],
                    "modal_artifact_path": "modal://dream-ai-training/runs/smoke-run",
                },
                "run_smoke_eval": True,
                "max_steps": 8,
                "layer_energy_probe_top_k": 4,
                "layer_energy_probe_sample_count": 2,
                "gradient_signature_dim": 64,
                "gradient_signature_every_steps": 2,
                "gradient_signature_start_step": 7,
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.tracking.mode == "online"
    assert config.tracking.project == "math-tutor-dynamic-lora"
    assert config.tracking.run_name == "smoke-run"
    assert config.tracking.tags == ("dynamic-lora", "smoke")
    assert config.tracking.modal_artifact_path == "modal://dream-ai-training/runs/smoke-run"
    assert config.run_smoke_eval is True
    assert config.layer_energy_probe_top_k == 4
    assert config.layer_energy_probe_sample_count == 2
    assert config.gradient_signature_dim == 64
    assert config.gradient_signature_every_steps == 2
    assert config.gradient_signature_start_step == 7


def test_gradient_signatures_require_selected_layers(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "train.jsonl",
                "holdout_path": "holdout.jsonl",
                "output_dir": "adapter",
                "metadata_path": "run.json",
                "gradient_signature_dim": 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="layer_energy_probe_top_k"):
        load_config(config_path)


def test_gradient_signatures_accept_base_probe_reference(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "train.jsonl",
                "holdout_path": "holdout.jsonl",
                "output_dir": "adapter",
                "metadata_path": "run.json",
                "gradient_signature_dim": 256,
                "signature_probe_metadata_path": "probe.json",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.layer_energy_probe_top_k == 0
    assert config.signature_probe_metadata_path == tmp_path / "probe.json"


def test_signature_probe_reference_cannot_be_combined_with_lora_probe(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "train.jsonl",
                "holdout_path": "holdout.jsonl",
                "output_dir": "adapter",
                "metadata_path": "run.json",
                "gradient_signature_dim": 256,
                "layer_energy_probe_top_k": 4,
                "signature_probe_metadata_path": "probe.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="exactly one"):
        load_config(config_path)


def test_gradient_signatures_start_step_must_fit_training_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_path": "train.jsonl",
                "holdout_path": "holdout.jsonl",
                "output_dir": "adapter",
                "metadata_path": "run.json",
                "max_steps": 4,
                "layer_energy_probe_top_k": 2,
                "gradient_signature_dim": 64,
                "gradient_signature_start_step": 5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="gradient_signature_start_step"):
        load_config(config_path)
