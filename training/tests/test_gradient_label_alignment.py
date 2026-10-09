"""Test deterministic joins and statistics for gradient-label alignment.
The fixtures keep label effects explicit enough to verify their direction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dynamic_lora.gradient_label_alignment import (
    SignatureModule,
    adjusted_rand_index,
    analyze_module_labels,
    benjamini_hochberg,
    load_record_labels,
    load_signature_modules,
    normalized_mutual_information,
)


def test_loaders_join_unique_records_and_reject_missing_labels(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "subject": "algebra", "difficulty": "short"}),
                json.dumps({"id": "b", "subject": "geometry", "difficulty": "long"}),
            ]
        ),
        encoding="utf-8",
    )
    signatures = tmp_path / "signatures.jsonl"
    signatures.write_text(
        json.dumps(
            {
                "record_id": "missing",
                "layer": "layer_0.q_proj",
                "signature": [1, 0],
                "step": 4,
            }
        ),
        encoding="utf-8",
    )

    labels = load_record_labels(records)

    with pytest.raises(ValueError, match="missing labels"):
        load_signature_modules(signatures, labels)


def test_load_signature_modules_requires_matching_module_coverage(tmp_path: Path) -> None:
    labels = {
        "a": {"subject": "algebra", "difficulty": "short"},
        "b": {"subject": "geometry", "difficulty": "long"},
    }
    signatures = tmp_path / "signatures.jsonl"
    rows = [
        {"record_id": "a", "layer": "layer_0.q_proj", "signature": [1, 0], "step": 4},
        {"record_id": "b", "layer": "layer_0.q_proj", "signature": [0, 1], "step": 4},
        {"record_id": "a", "layer": "layer_0.k_proj", "signature": [1, 0], "step": 4},
    ]
    signatures.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent record coverage"):
        load_signature_modules(signatures, labels)  # type: ignore[arg-type]


def test_label_analysis_detects_separated_directions() -> None:
    directions = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, -0.01]], dtype=np.float64
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    module = SignatureModule(("a", "b", "c", "d"), directions, 0)
    labels = {
        "a": {"subject": "left", "difficulty": "same"},
        "b": {"subject": "left", "difficulty": "same"},
        "c": {"subject": "right", "difficulty": "same"},
        "d": {"subject": "right", "difficulty": "same"},
    }

    result = analyze_module_labels(
        module,
        labels,  # type: ignore[arg-type]
        "subject",
        bootstrap_repeats=20,
        permutation_repeats=20,
        cluster_starts=3,
        seed=7,
    )

    assert result["pair_mean_difference"] > 1.9
    assert result["record_mean_contrast"] > 1.9
    assert result["clustering"]["ari"] == pytest.approx(1.0)
    assert result["clustering"]["nmi"] == pytest.approx(1.0)


def test_partition_metrics_match_identical_and_unrelated_partitions() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    unrelated = np.asarray([0, 1, 0, 1], dtype=np.int64)

    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)
    assert normalized_mutual_information(labels, labels) == pytest.approx(1.0)
    assert adjusted_rand_index(labels, unrelated) < 0
    assert normalized_mutual_information(labels, unrelated) == pytest.approx(0.0)


def test_benjamini_hochberg_preserves_order_and_monotonic_adjustment() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])

    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.008])
