from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_lora.data import (
    DatasetValidationError,
    tokenize_training_batch,
    validate_training_dataset,
)


def test_tokenize_training_batch_appends_eos_after_truncation() -> None:
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, _text: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
            assert kwargs["max_length"] == 7
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    encoded = tokenize_training_batch({"text": ["example"]}, Tokenizer(), max_seq_length=8)

    assert encoded == {"input_ids": [[1, 2, 99]], "attention_mask": [[1, 1, 1]]}


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def valid_record(record_id: str, difficulty: str) -> dict[str, object]:
    manim_code = (
        "from manim import *\n\n"
        "class Lesson(Scene):\n"
        "    def construct(self):\n"
        "        self.add(Text('d/dx x^2 = 2x'))\n"
    )
    return {
        "id": record_id,
        "difficulty": difficulty,
        "topic": "calculus",
        "prompt": "Create a Manim lesson for the derivative of x squared.",
        "manim_code": manim_code,
        "split": "train",
        "source": {"name": "unit-fixture", "reference": record_id},
    }


def test_training_records_must_cover_allowed_difficulties_and_avoid_holdout(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    write_jsonl(
        train_path,
        [
            valid_record("train-foundational-001", "foundational"),
            valid_record("train-intermediate-001", "intermediate"),
            valid_record("train-advanced-001", "advanced"),
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

    report = validate_training_dataset(train_path, holdout_path)

    assert report.record_count == 3
    assert report.difficulty_counts == {"foundational": 1, "intermediate": 1, "advanced": 1}
    assert (
        report.training_ids_sha256
        == "00ad1476d0252cea523a104372073a97bf291704b98e14942b2bb76cc23c5485"
    )


def test_validation_rejects_duplicate_unstable_and_holdout_ids(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    overlapping = valid_record("eval-foundational-001", "foundational")
    write_jsonl(
        train_path,
        [
            valid_record("bad id", "foundational"),
            valid_record("train-advanced-001", "expert"),
            overlapping,
            overlapping,
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

    with pytest.raises(DatasetValidationError) as exc_info:
        validate_training_dataset(train_path, holdout_path)

    message = str(exc_info.value)
    assert "duplicate training id: eval-foundational-001" in message
    assert "invalid id format: bad id" in message
    assert "invalid difficulty for train-advanced-001: expert" in message
    assert "training IDs overlap holdout IDs: eval-foundational-001" in message


def test_validation_rejects_empty_holdout(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    write_jsonl(
        train_path,
        [
            valid_record("train-foundational-001", "foundational"),
            valid_record("train-intermediate-001", "intermediate"),
            valid_record("train-advanced-001", "advanced"),
        ],
    )
    holdout_path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="holdout dataset must not be empty"):
        validate_training_dataset(train_path, holdout_path)


def test_dataset_fingerprints_change_with_content_and_order(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    records = [
        valid_record("train-foundational-001", "foundational"),
        valid_record("train-intermediate-001", "intermediate"),
        valid_record("train-advanced-001", "advanced"),
    ]
    write_jsonl(train_path, records)
    write_jsonl(
        holdout_path,
        [
            {
                "id": "eval-foundational-001",
                "difficulty": "foundational",
                "topic": "fractions",
                "prompt": "Holdout prompt.",
            }
        ],
    )
    first = validate_training_dataset(train_path, holdout_path)

    write_jsonl(train_path, list(reversed(records)))
    reordered = validate_training_dataset(train_path, holdout_path)
    records[0]["prompt"] = "Changed training prompt."
    write_jsonl(train_path, records)
    changed = validate_training_dataset(train_path, holdout_path)

    assert first.training_content_sha256 != reordered.training_content_sha256
    assert first.training_content_sha256 != changed.training_content_sha256
    assert first.holdout_content_sha256 == reordered.holdout_content_sha256
