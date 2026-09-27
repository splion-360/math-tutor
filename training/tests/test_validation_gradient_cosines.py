"""Verify exact cosine and subject summaries without loading Qwen.
The tests distinguish same-subject pairs from cross-subject opposition."""

from __future__ import annotations

import numpy as np
import pytest

from dynamic_lora.validation_gradient_cosines import cosine_matrix, summarize_cosines


def test_cosine_summary_uses_cross_subject_example_pairs() -> None:
    gradients = np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    result = summarize_cosines(
        cosine_matrix(gradients @ gradients.T), ["Algebra", "Algebra", "Topology"]
    )

    assert result["cross_subject_negative_rate"] == 1.0
    assert result["subject_pair_mean_cosine"]["Algebra"]["Topology"] == -1.0
    assert result["subject_pair_mean_cosine"]["Algebra"]["Algebra"] == 1.0
    assert result["subject_pair_mean_cosine"]["Topology"]["Topology"] is None


def test_cosine_summary_rejects_zero_gradient() -> None:
    with pytest.raises(ValueError, match="complete"):
        summarize_cosines(cosine_matrix(np.asarray([[0.0, 0.0], [0.0, 1.0]])), ["A", "B"])
