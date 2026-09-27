"""Load and validate supervised Manim training records.
The module owns deterministic dataset validation and content fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dynamic_lora.constants import ALLOWED_DIFFICULTIES

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SYSTEM_PROMPT = (
    "You generate concise, runnable Manim Community Edition Python scenes for math tutoring. "
    "Return only Python code."
)


class ChatTemplateTokenizer(Protocol):
    """Tokenizer contract needed to render supervised Manim chat records."""

    eos_token: str | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


def _task_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Difficulty: {record['difficulty']}\n"
                f"Topic: {record['topic']}\n"
                f"Task: {record['prompt']}"
            ),
        },
    ]


def format_training_prompt(record: dict[str, Any], tokenizer: ChatTemplateTokenizer) -> str:
    """Render the generation prefix used by both placement training and evaluation.

    Args:
        record: A supervised lesson or prompt-only evaluation record.
        tokenizer: Qwen tokenizer providing its chat template.

    Returns:
        System and user turns followed by the assistant generation marker.
    """
    return tokenizer.apply_chat_template(
        _task_messages(record), tokenize=False, add_generation_prompt=True
    )


def format_training_record(
    record: dict[str, Any], tokenizer: ChatTemplateTokenizer
) -> dict[str, str]:
    """Format one Manim record with the shared supervised chat template.

    Args:
        record: Validated training record containing prompt, topic, difficulty, and code.
        tokenizer: Tokenizer that renders chat messages to text.

    Returns:
        A text field suitable for the training tokenizer.
    """
    messages = [
        *_task_messages(record),
        {"role": "assistant", "content": str(record["manim_code"])},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if tokenizer.eos_token is not None and not text.rstrip().endswith(tokenizer.eos_token):
        text = text.rstrip() + tokenizer.eos_token
    return {"text": text}


def tokenize_completion_record(
    record: dict[str, Any], tokenizer: Any, *, max_seq_length: int
) -> dict[str, list[int]]:
    """Tokenize a lesson while masking the prompt from supervised code loss.

    Args:
        record: Validated training record with target Manim code.
        tokenizer: Tokenizer for the frozen Qwen revision.
        max_seq_length: Maximum sequence length including the final EOS token.

    Returns:
        Input IDs, attention mask, and labels with prompt tokens set to -100.

    Raises:
        ValueError: If truncation removes the target code or templates do not align.
    """
    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least 2")
    prompt = format_training_prompt(record, tokenizer)
    full = format_training_record(record, tokenizer)["text"]
    prompt_ids: list[int] = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids: list[int] = tokenizer(
        full,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length - 1,
    )["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError(f"prompt and completion tokens do not align for {record['id']}")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and full_ids[-1] != eos_token_id:
        full_ids.append(eos_token_id)
        labels.append(eos_token_id)
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def tokenize_training_batch(
    batch: dict[str, list[str]], tokenizer: Any, *, max_seq_length: int
) -> dict[str, Any]:
    """Tokenize formatted lessons with the same truncation and EOS contract.

    Args:
        batch: Text fields produced by ``format_training_record``.
        tokenizer: Tokenizer for the frozen Qwen revision.
        max_seq_length: Maximum token count including the final EOS token.

    Returns:
        Token IDs and attention masks suitable for causal-LM collation.

    Raises:
        ValueError: If the context length cannot contain content and EOS.
    """
    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least 2")
    encoded: dict[str, Any] = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_length - 1,
        padding=False,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        for index, input_ids in enumerate(encoded["input_ids"]):
            if not input_ids or input_ids[-1] != eos_token_id:
                input_ids.append(eos_token_id)
                if "attention_mask" in encoded:
                    encoded["attention_mask"][index].append(1)
    return encoded


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
