from __future__ import annotations

import json
from pathlib import Path

from dynamic_lora.config import load_config


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
                "layer_energy_probe_top_k": 4,
                "layer_energy_probe_sample_count": 2,
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
