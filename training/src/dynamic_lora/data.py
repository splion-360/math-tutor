"""Load and validate supervised Manim training records.
The module owns deterministic dataset validation and content fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamic_lora.constants import ALLOWED_DIFFICULTIES

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class DatasetValidationReport:
    record_count: int
    difficulty_counts: dict[str, int]
    training_ids_sha256: str
    training_content_sha256: str
    holdout_content_sha256: str


class DatasetValidationError(ValueError):
    pass


def validate_training_dataset(train_path: Path, holdout_path: Path) -> DatasetValidationReport:
    errors: list[str] = []
    training_records = _read_jsonl_objects(train_path, label="training", errors=errors)
    holdout_records = _read_jsonl_objects(holdout_path, label="holdout", errors=errors)
    if not holdout_records:
        errors.append("holdout dataset must not be empty")
    holdout_ids = _collect_ids(holdout_records, label="holdout", errors=errors)

    seen_ids: set[str] = set()
    training_ids: list[str] = []
    difficulty_counts = {difficulty: 0 for difficulty in ALLOWED_DIFFICULTIES}

    for index, record in enumerate(training_records, start=1):
        record_id = _string_field(record, "id")
        if record_id is None:
            errors.append(f"training line {index} missing string id")
            continue
        if record_id in seen_ids:
            errors.append(f"duplicate training id: {record_id}")
        seen_ids.add(record_id)
        training_ids.append(record_id)
        if ID_PATTERN.fullmatch(record_id) is None:
            errors.append(f"invalid id format: {record_id}")

        difficulty = _string_field(record, "difficulty")
        if difficulty not in ALLOWED_DIFFICULTIES:
            errors.append(f"invalid difficulty for {record_id}: {difficulty}")
        else:
            difficulty_counts[difficulty] += 1

        split = _string_field(record, "split")
        if split != "train":
            errors.append(f"invalid split for {record_id}: {split}")

        for field_name in ("topic", "prompt", "manim_code"):
            value = _string_field(record, field_name)
            if value is None or not value.strip():
                errors.append(f"{field_name} must be non-empty for {record_id}")

        source = record.get("source")
        if not isinstance(source, dict):
            errors.append(f"source must be an object for {record_id}")
        else:
            for source_field in ("name", "reference"):
                source_value = source.get(source_field)
                if not isinstance(source_value, str) or not source_value.strip():
                    errors.append(f"source.{source_field} must be non-empty for {record_id}")

    missing_difficulties = [
        difficulty for difficulty, count in difficulty_counts.items() if count == 0
    ]
    if missing_difficulties:
        errors.append("missing difficulty coverage: " + ", ".join(missing_difficulties))

    overlap = sorted(set(training_ids) & holdout_ids)
    if overlap:
        errors.append("training IDs overlap holdout IDs: " + ", ".join(overlap))

    if errors:
        raise DatasetValidationError("; ".join(errors))

    joined_ids = "\n".join(sorted(training_ids)) + "\n"
    return DatasetValidationReport(
        record_count=len(training_records),
        difficulty_counts=difficulty_counts,
        training_ids_sha256=hashlib.sha256(joined_ids.encode("utf-8")).hexdigest(),
        training_content_sha256=_sha256_file(train_path),
        holdout_content_sha256=_sha256_file(holdout_path),
    )


def load_training_records(train_path: Path) -> list[dict[str, Any]]:
    errors: list[str] = []
    records = _read_jsonl_objects(train_path, label="training", errors=errors)
    if errors:
        raise DatasetValidationError("; ".join(errors))
    return records


def _read_jsonl_objects(path: Path, *, label: str, errors: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        errors.append(f"{label} path does not exist: {path}")
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label} line {line_number} is invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{label} line {line_number} must be a JSON object")
            continue
        records.append(value)
    return records


def _collect_ids(records: list[dict[str, Any]], *, label: str, errors: list[str]) -> set[str]:
    ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        record_id = _string_field(record, "id")
        if record_id is None:
            errors.append(f"{label} line {index} missing string id")
        else:
            ids.add(record_id)
    return ids


def _string_field(record: dict[str, Any], name: str) -> str | None:
    value = record.get(name)
    return value if isinstance(value, str) else None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
