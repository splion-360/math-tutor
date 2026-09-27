"""Check held-out gradient aggregation without loading model weights.
The tests protect validation coverage and subject-level arithmetic means."""

from __future__ import annotations

import pytest

from dynamic_lora.validation_gradient_heatmap import summarize_subject_layer_norms


def test_subject_layer_norms_average_per_example() -> None:
    rows = [
        {"record_id": "a", "subject": "Algebra", "layer_norms": {"layer_0": 2.0, "layer_1": 4.0}},
        {"record_id": "b", "subject": "Algebra", "layer_norms": {"layer_0": 4.0, "layer_1": 8.0}},
        {"record_id": "c", "subject": "Topology", "layer_norms": {"layer_0": 7.0, "layer_1": 9.0}},
    ]

    summary = summarize_subject_layer_norms(rows, expected_ids={"a", "b", "c"})

    assert summary["subjects"]["Algebra"] == {
        "count": 2,
        "mean_layer_norms": {"layer_0": 3.0, "layer_1": 6.0},
    }
    assert summary["subjects"]["Topology"]["count"] == 1
    assert summary["layers"] == ["layer_0", "layer_1"]


@pytest.mark.parametrize("ids", [{"a"}, {"a", "b", "c"}])
def test_subject_layer_norms_reject_missing_or_extra_ids(ids: set[str]) -> None:
    row = {"record_id": "a", "subject": "Algebra", "layer_norms": {"layer_0": 2.0}}

    with pytest.raises(ValueError, match="exactly once"):
        summarize_subject_layer_norms([row, row] if len(ids) == 1 else [row], expected_ids=ids)


def test_subject_layer_norms_reject_incomplete_layers() -> None:
    rows = [
        {"record_id": "a", "subject": "Algebra", "layer_norms": {"layer_0": 2.0}},
        {"record_id": "b", "subject": "Topology", "layer_norms": {"layer_1": 1.0}},
    ]

    with pytest.raises(ValueError, match="complete"):
        summarize_subject_layer_norms(rows, expected_ids={"a", "b"})
