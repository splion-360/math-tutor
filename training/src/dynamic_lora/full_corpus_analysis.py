"""Analyze identified gradient directions from a fixed model checkpoint.
The summary separates within-topic from between-topic negative cosine pairs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def summarize_gradient_directions(
    rows: list[dict[str, Any]], topics_by_id: dict[str, str]
) -> dict[str, dict[str, int | float | None]]:
    """Compare all nonzero per-example signatures within each module.

    Args:
        rows: One identified projected gradient per example and monitored module.
        topics_by_id: Source topic labels used only to stratify analysis.

    Returns:
        Pair counts, negative-cosine rates, and topic-stratified counts by module.

    Raises:
        ValueError: If IDs, signatures, or topic metadata are incomplete.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["record_id"] not in topics_by_id:
            raise ValueError(f"missing topic for gradient record {row['record_id']}")
        grouped[row["layer"]].append(row)
    result: dict[str, dict[str, int | float | None]] = {}
    for layer, layer_rows in sorted(grouped.items()):
        ids = [str(row["record_id"]) for row in layer_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate gradient record in {layer}")
        matrix = np.asarray([row["signature"] for row in layer_rows], dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"invalid gradient signatures in {layer}")
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms > 0
        valid = matrix[nonzero] / norms[nonzero, None]
        valid_ids = [record_id for record_id, keep in zip(ids, nonzero, strict=True) if keep]
        upper = np.triu_indices(len(valid_ids), k=1)
        cosines = (valid @ valid.T)[upper]
        topic_values = np.asarray([topics_by_id[record_id] for record_id in valid_ids])
        within = topic_values[upper[0]] == topic_values[upper[1]]
        negative = cosines < 0
        pair_count = int(cosines.size)
        result[layer] = {
            "examples": len(layer_rows),
            "zero_gradient_count": int((~nonzero).sum()),
            "pair_count": pair_count,
            "negative_pairs": int(negative.sum()),
            "negative_rate": float(negative.mean()) if pair_count else None,
            "mean_cosine": float(cosines.mean()) if pair_count else None,
            "within_topic_pairs": int(within.sum()),
            "within_topic_negative_pairs": int((negative & within).sum()),
            "between_topic_pairs": int((~within).sum()),
            "between_topic_negative_pairs": int((negative & ~within).sum()),
        }
    return result
