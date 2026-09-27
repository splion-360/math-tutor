"""Define a disjoint train and validation plan for full-corpus experiments.
The plan records exact row coverage before model weights are loaded."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class FullCorpusPlan:
    """Exact row IDs and presentation count for a full-epoch experiment."""

    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    epochs: int
    examples_per_epoch: int
    total_training_presentations: int


def build_full_corpus_plan(
    records: list[dict[str, str]], *, seed: int, epochs: int
) -> FullCorpusPlan:
    """Reserve 10% per difficulty and schedule every remaining row per epoch.

    Args:
        records: Validated source records with unique IDs and difficulty metadata.
        seed: Stable hash seed for the disjoint validation selection.
        epochs: Number of complete training passes.

    Returns:
        The exact train and validation ID sets and planned example presentations.

    Raises:
        ValueError: If the corpus or epoch count cannot support the split.
    """
    if epochs <= 0 or not records:
        raise ValueError("epochs must be positive and records must not be empty")
    by_difficulty: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_difficulty[record["difficulty"]].append(record["id"])
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("record IDs must be unique")
    validation: set[str] = set()
    for difficulty, record_ids in by_difficulty.items():
        count = max(1, round(len(record_ids) * 0.1))
        if count >= len(record_ids):
            raise ValueError(f"not enough {difficulty} rows for training and validation")
        ranked = sorted(
            record_ids,
            key=lambda record_id: (
                hashlib.sha256(f"{seed}:{record_id}".encode()).digest(),
                record_id,
            ),
        )
        validation.update(ranked[:count])
    training_ids = tuple(record["id"] for record in records if record["id"] not in validation)
    return FullCorpusPlan(
        training_ids=training_ids,
        validation_ids=tuple(sorted(validation)),
        epochs=epochs,
        examples_per_epoch=len(training_ids),
        total_training_presentations=len(training_ids) * epochs,
    )
