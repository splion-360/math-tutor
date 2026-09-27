"""Test deterministic shared-LoRA run plans and saved provenance.
These checks keep training inputs and artifact paths inspectable."""

from __future__ import annotations

import json
from pathlib import Path

from dynamic_lora.config import load_config
from dynamic_lora.run_plan import build_run_plan, write_run_metadata


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def training_record(record_id: str, difficulty: str) -> dict[str, object]:
    manim_code = (
        "from manim import *\n\n"
        "class Lesson(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Text('A = bh / 2'))\n"
    )
    return {
        "id": record_id,
        "difficulty": difficulty,
        "topic": "geometry",
        "prompt": "Create a Manim lesson for a triangle area formula.",
        "manim_code": manim_code,
        "split": "train",
        "source": {"name": "unit-fixture", "reference": record_id},
    }


def test_dry_run_plan_is_redacted_deterministic_and_machine_readable(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    output_dir = tmp_path / "outputs" / "shared-lora"
    metadata_path = tmp_path / "metadata" / "run.json"
    config_path = tmp_path / "config.json"
    write_jsonl(
        train_path,
        [
            training_record("train-foundational-001", "foundational"),
            training_record("train-intermediate-001", "intermediate"),
            training_record("train-advanced-001", "advanced"),
        ],
    )
    write_jsonl(
        holdout_path,
        [
            {
                "id": "eval-foundational-001",
                "difficulty": "foundational",
                "topic": "fractions",
                "prompt": "Holdout prompt.",
                "split": "evaluation",
                "source": {"name": "holdout", "reference": "eval-foundational-001"},
            }
        ],
    )
    config_path.write_text(
        json.dumps(
            {
                "train_path": str(train_path),
                "holdout_path": str(holdout_path),
                "output_dir": str(output_dir),
                "metadata_path": str(metadata_path),
                "seed": 17,
                "max_steps": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0002,
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    first_plan = build_run_plan(config)
    second_plan = build_run_plan(config)
    write_run_metadata(first_plan, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert first_plan == second_plan
    assert str(tmp_path) not in first_plan.redacted_text
    assert "<external>" in first_plan.redacted_text
    assert metadata["condition"] == "shared_lora_static_control"
    assert metadata["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert metadata["dynamic_adapter_spawning"] is False
    assert metadata["learned_router"] is False
    assert metadata["model_revision"] == "1b4199c4f36b0cef378bfb12390c18780c18af4c"
    assert metadata["weights_loaded"] is False
    assert metadata["runtime_versions"] == {}
    assert len(metadata["dataset"]["training_content_sha256"]) == 64
    assert len(metadata["dataset"]["holdout_content_sha256"]) == 64
    assert metadata["dataset"]["difficulty_counts"] == {
        "advanced": 1,
        "foundational": 1,
        "intermediate": 1,
    }
    assert metadata["training"]["prompt_includes_difficulty"] is False
    assert metadata["training"]["target_truncation"] == "reject"
