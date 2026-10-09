"""Measure whether projected LoRA gradients align with corpus labels.
The analysis keeps subject and code-length difficulty claims statistically separate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


class RecordLabels(TypedDict):
    """Labels joined to one recovered gradient-signature record."""

    subject: str
    difficulty: str


@dataclass(frozen=True)
class SignatureModule:
    """Normalized projected gradients for one LoRA module.

    Attributes:
        record_ids: Stable corpus identifiers in matrix row order.
        directions: Unit-length projected gradients.
        zero_gradient_count: Rows omitted because their projected norm was zero.
    """

    record_ids: tuple[str, ...]
    directions: NDArray[np.float64]
    zero_gradient_count: int


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local analysis input.

    Args:
        path: File whose bytes identify the analysis input.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_record_labels(path: Path) -> dict[str, RecordLabels]:
    """Load unique subject and difficulty labels from prepared corpus records.

    Args:
        path: Prepared Bespoke-Manim JSON Lines file.

    Returns:
        Labels keyed by stable record identifier.

    Raises:
        ValueError: If a record is malformed, duplicated, or lacks a label.
    """
    labels: dict[str, RecordLabels] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            values = {name: row[name] for name in ("id", "subject", "difficulty")}
        except KeyError as error:
            raise ValueError(f"missing label field on line {line_number}") from error
        if any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in values.values()
        ):
            raise ValueError(f"invalid label field on line {line_number}")
        record_id = values["id"]
        subject = values["subject"]
        difficulty = values["difficulty"]
        if record_id in labels:
            raise ValueError(f"duplicate corpus record {record_id}")
        labels[record_id] = {"subject": subject, "difficulty": difficulty}
    if not labels:
        raise ValueError("corpus label file is empty")
    return labels


def load_signature_modules(
    path: Path, labels: dict[str, RecordLabels]
) -> tuple[dict[str, SignatureModule], int]:
    """Load and normalize projected gradients after validating their label join.

    Args:
        path: JSON Lines stream of projected gradients.
        labels: Corpus metadata keyed by stable record identifier.

    Returns:
        Modules keyed by layer/projection name and their shared checkpoint step.

    Raises:
        ValueError: If signatures are malformed, duplicated, missing labels, or
            do not cover the same records at one checkpoint step.
    """
    grouped: dict[str, dict[str, list[float]]] = {}
    steps: set[int] = set()
    dimensions: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            module = str(row["layer"])
            record_id = str(row["record_id"])
            signature = [float(value) for value in row["signature"]]
            step = int(row["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid signature row on line {line_number}") from error
        if not module or not record_id or not signature:
            raise ValueError(f"empty signature field on line {line_number}")
        if record_id not in labels:
            raise ValueError(f"missing labels for gradient record {record_id}")
        module_rows = grouped.setdefault(module, {})
        if record_id in module_rows:
            raise ValueError(f"duplicate gradient record {record_id} in {module}")
        module_rows[record_id] = signature
        steps.add(step)
        dimensions.add(len(signature))
    if not grouped:
        raise ValueError("gradient-signature file is empty")
    if len(steps) != 1:
        raise ValueError("gradient signatures must come from one checkpoint step")
    if len(dimensions) != 1:
        raise ValueError("gradient signatures must have one projection dimension")

    expected_ids = set(next(iter(grouped.values())))
    modules: dict[str, SignatureModule] = {}
    for module, rows in sorted(grouped.items()):
        if set(rows) != expected_ids:
            raise ValueError(f"inconsistent record coverage in {module}")
        record_ids = tuple(sorted(rows))
        matrix = np.asarray([rows[record_id] for record_id in record_ids], dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"invalid gradient signatures in {module}")
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms > 0
        modules[module] = SignatureModule(
            record_ids=tuple(
                record_id for record_id, keep in zip(record_ids, nonzero, strict=True) if keep
            ),
            directions=matrix[nonzero] / norms[nonzero, None],
            zero_gradient_count=int((~nonzero).sum()),
        )
    return modules, steps.pop()


def _encode_labels(values: list[str]) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    categories = tuple(sorted(set(values)))
    indices = {value: index for index, value in enumerate(categories)}
    return np.asarray([indices[value] for value in values], dtype=np.int64), categories


def _record_contrasts(
    cosines: NDArray[np.float64], encoded: NDArray[np.int64]
) -> NDArray[np.float64]:
    contrasts = np.full(len(encoded), np.nan, dtype=np.float64)
    total = cosines.sum(axis=1)
    for label in np.unique(encoded):
        members = np.flatnonzero(encoded == label)
        if len(members) < 2 or len(members) == len(encoded):
            continue
        same_sums = cosines[:, members].sum(axis=1)
        contrasts[members] = (same_sums[members] - 1.0) / (len(members) - 1) - (
            total[members] - same_sums[members]
        ) / (len(encoded) - len(members))
    return contrasts


def _pair_statistics(
    cosines: NDArray[np.float64], encoded: NDArray[np.int64]
) -> dict[str, int | float]:
    upper = np.triu_indices(len(encoded), k=1)
    pair_cosines = cosines[upper]
    within_mask = encoded[upper[0]] == encoded[upper[1]]
    within = pair_cosines[within_mask]
    across = pair_cosines[~within_mask]
    pooled_scale = math.sqrt((float(within.var()) + float(across.var())) / 2)
    difference = float(within.mean() - across.mean())
    return {
        "within_pair_count": int(within.size),
        "across_pair_count": int(across.size),
        "within_mean_cosine": float(within.mean()),
        "across_mean_cosine": float(across.mean()),
        "pair_mean_difference": difference,
        "standardized_pair_mean_difference": difference / pooled_scale if pooled_scale else 0.0,
    }


def _bootstrap_interval(
    values: NDArray[np.float64], repeats: int, rng: np.random.Generator
) -> list[float]:
    valid = values[np.isfinite(values)]
    if not valid.size:
        raise ValueError("no records have both within-label and across-label comparisons")
    samples = rng.choice(valid, size=(repeats, len(valid)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return [float(lower), float(upper)]


def _permutation_test(
    cosines: NDArray[np.float64], encoded: NDArray[np.int64], repeats: int, rng: np.random.Generator
) -> dict[str, float | list[float]]:
    observed = float(np.nanmean(_record_contrasts(cosines, encoded)))
    null = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        null[index] = float(np.nanmean(_record_contrasts(cosines, rng.permutation(encoded))))
    probability = (1 + int(np.count_nonzero(np.abs(null) >= abs(observed)))) / (repeats + 1)
    return {
        "observed_record_mean_contrast": observed,
        "two_sided_p_value": probability,
        "null_mean": float(null.mean()),
        "null_interval_95": [float(value) for value in np.quantile(null, [0.025, 0.975])],
    }


def _spherical_kmeans(
    directions: NDArray[np.float64], cluster_count: int, seed: int, max_iterations: int = 100
) -> tuple[NDArray[np.int64], float]:
    rng = np.random.default_rng(seed)
    centers = [directions[int(rng.integers(len(directions)))]]
    while len(centers) < cluster_count:
        similarities = directions @ np.asarray(centers).T
        distances = np.maximum(0.0, 1.0 - similarities.max(axis=1))
        probabilities = distances / distances.sum() if distances.sum() else None
        selected = int(rng.choice(len(directions), p=probabilities))
        centers.append(directions[selected])
    center_matrix = np.asarray(centers)
    assignments = np.full(len(directions), -1, dtype=np.int64)
    for _ in range(max_iterations):
        similarities = directions @ center_matrix.T
        updated = similarities.argmax(axis=1).astype(np.int64)
        if np.array_equal(assignments, updated):
            break
        assignments = updated
        for cluster in range(cluster_count):
            members = directions[assignments == cluster]
            if not len(members):
                weakest = int(np.argmin(similarities.max(axis=1)))
                center_matrix[cluster] = directions[weakest]
                continue
            center = members.mean(axis=0)
            norm = np.linalg.norm(center)
            center_matrix[cluster] = center / norm if norm else members[0]
    objective = float(np.mean(np.max(directions @ center_matrix.T, axis=1)))
    return assignments, objective


def _contingency(left: NDArray[np.int64], right: NDArray[np.int64]) -> NDArray[np.int64]:
    table = np.zeros((int(left.max()) + 1, int(right.max()) + 1), dtype=np.int64)
    np.add.at(table, (left, right), 1)
    return table


def adjusted_rand_index(left: NDArray[np.int64], right: NDArray[np.int64]) -> float:
    """Compute the adjusted Rand index for two complete partitions.

    Args:
        left: Integer cluster assignments.
        right: Integer reference labels.

    Returns:
        Chance-adjusted pair agreement in the conventional [-1, 1] range.
    """
    table = _contingency(left, right)
    joint = float((table * (table - 1) / 2).sum())
    row_counts = table.sum(axis=1)
    column_counts = table.sum(axis=0)
    row_pairs = float((row_counts * (row_counts - 1) / 2).sum())
    column_pairs = float((column_counts * (column_counts - 1) / 2).sum())
    total_pairs = len(left) * (len(left) - 1) / 2
    expected = row_pairs * column_pairs / total_pairs if total_pairs else 0.0
    maximum = (row_pairs + column_pairs) / 2
    return (joint - expected) / (maximum - expected) if maximum != expected else 1.0


def normalized_mutual_information(left: NDArray[np.int64], right: NDArray[np.int64]) -> float:
    """Compute arithmetic-mean normalized mutual information.

    Args:
        left: Integer cluster assignments.
        right: Integer reference labels.

    Returns:
        Mutual information normalized by the mean partition entropy.
    """
    table = _contingency(left, right).astype(np.float64)
    total = table.sum()
    probabilities = table / total
    left_probabilities = probabilities.sum(axis=1)
    right_probabilities = probabilities.sum(axis=0)
    rows, columns = np.nonzero(probabilities)
    mutual_information = float(
        sum(
            probabilities[row, column]
            * math.log(
                probabilities[row, column] / (left_probabilities[row] * right_probabilities[column])
            )
            for row, column in zip(rows, columns, strict=True)
        )
    )
    left_entropy = -float(sum(value * math.log(value) for value in left_probabilities if value))
    right_entropy = -float(sum(value * math.log(value) for value in right_probabilities if value))
    denominator = (left_entropy + right_entropy) / 2
    return mutual_information / denominator if denominator else 1.0


def _cluster_agreement(
    directions: NDArray[np.float64],
    encoded: NDArray[np.int64],
    starts: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    candidates = [
        _spherical_kmeans(directions, len(np.unique(encoded)), seed + offset)
        for offset in range(starts)
    ]
    assignments, objective = max(candidates, key=lambda candidate: candidate[1])
    observed_ari = adjusted_rand_index(assignments, encoded)
    observed_nmi = normalized_mutual_information(assignments, encoded)
    rng = np.random.default_rng(seed + 10_000)
    null_ari = np.empty(permutations, dtype=np.float64)
    null_nmi = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = rng.permutation(encoded)
        null_ari[index] = adjusted_rand_index(assignments, shuffled)
        null_nmi[index] = normalized_mutual_information(assignments, shuffled)
    ari_probability = (1 + int(np.count_nonzero(null_ari >= observed_ari))) / (permutations + 1)
    nmi_probability = (1 + int(np.count_nonzero(null_nmi >= observed_nmi))) / (permutations + 1)
    return {
        "cluster_count": int(len(np.unique(encoded))),
        "starts": starts,
        "selected_objective": objective,
        "ari": observed_ari,
        "nmi": observed_nmi,
        "ari_permutation_p_value": ari_probability,
        "nmi_permutation_p_value": nmi_probability,
        "ari_null_mean": float(null_ari.mean()),
        "ari_null_interval_95": [float(value) for value in np.quantile(null_ari, [0.025, 0.975])],
        "nmi_null_mean": float(null_nmi.mean()),
        "nmi_null_interval_95": [float(value) for value in np.quantile(null_nmi, [0.025, 0.975])],
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjust a family of p-values with the Benjamini-Hochberg procedure.

    Args:
        p_values: Unadjusted probabilities in their reporting order.

    Returns:
        False-discovery-rate adjusted q-values in the original order.

    Raises:
        ValueError: If the family is empty or contains invalid probabilities.
    """
    if not p_values or any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be a non-empty family in [0, 1]")
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[int(index)] * len(p_values) / rank)
        adjusted[int(index)] = running
    return [float(value) for value in adjusted]


def analyze_module_labels(
    module: SignatureModule,
    labels: dict[str, RecordLabels],
    label_name: Literal["subject", "difficulty"],
    bootstrap_repeats: int,
    permutation_repeats: int,
    cluster_starts: int,
    seed: int,
) -> dict[str, Any]:
    """Measure pairwise separation and cluster agreement for one label axis.

    Args:
        module: Unit projected-gradient directions for one LoRA module.
        labels: Subject and difficulty values keyed by record identifier.
        label_name: Either ``subject`` or ``difficulty``.
        bootstrap_repeats: Record-contrast resamples used for the interval.
        permutation_repeats: Label shuffles used for null comparisons.
        cluster_starts: Independent spherical k-means initializations.
        seed: Deterministic random seed for this module and label axis.

    Returns:
        Label counts, cosine effects, permutation result, and cluster agreement.

    Raises:
        ValueError: If the requested label is unsupported or has fewer than two groups.
    """
    if label_name not in {"subject", "difficulty"}:
        raise ValueError(f"unsupported label axis {label_name}")
    values = [labels[record_id][label_name] for record_id in module.record_ids]
    encoded, categories = _encode_labels(values)
    if len(categories) < 2:
        raise ValueError(f"{label_name} needs at least two represented groups")
    cosines = np.clip(module.directions @ module.directions.T, -1.0, 1.0)
    contrasts = _record_contrasts(cosines, encoded)
    counts = {category: values.count(category) for category in categories}
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "group_counts": counts,
        **_pair_statistics(cosines, encoded),
        "record_mean_contrast": float(np.nanmean(contrasts)),
        "record_bootstrap_interval_95": _bootstrap_interval(contrasts, bootstrap_repeats, rng),
        "permutation": _permutation_test(
            cosines, encoded, permutation_repeats, np.random.default_rng(seed + 1_000)
        ),
        "clustering": _cluster_agreement(
            module.directions,
            encoded,
            cluster_starts,
            permutation_repeats,
            seed + 2_000,
        ),
    }
    return result


def analyze_gradient_label_alignment(
    modules: dict[str, SignatureModule],
    labels: dict[str, RecordLabels],
    *,
    bootstrap_repeats: int = 2_000,
    permutation_repeats: int = 2_000,
    cluster_starts: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze subject and difficulty association across every recovered module.

    Args:
        modules: Projected gradients keyed by layer/projection module.
        labels: Subject and difficulty metadata keyed by record identifier.
        bootstrap_repeats: Record-level contrast bootstrap resamples.
        permutation_repeats: Label shuffles for effect and clustering nulls.
        cluster_starts: Spherical k-means initializations per module and axis.
        seed: Root seed from which deterministic module seeds are derived.

    Returns:
        Per-module results and analysis-level counts.

    Raises:
        ValueError: If repetition counts are non-positive or modules are absent.
    """
    if not modules:
        raise ValueError("at least one signature module is required")
    if min(bootstrap_repeats, permutation_repeats, cluster_starts) < 1:
        raise ValueError("analysis repetition counts must be positive")
    results: dict[str, Any] = {}
    for index, (name, module) in enumerate(sorted(modules.items())):
        results[name] = {
            "examples": len(module.record_ids),
            "projection_dimension": int(module.directions.shape[1]),
            "zero_gradient_count": module.zero_gradient_count,
            "subject": analyze_module_labels(
                module,
                labels,
                "subject",
                bootstrap_repeats,
                permutation_repeats,
                cluster_starts,
                seed + index * 10,
            ),
            "difficulty": analyze_module_labels(
                module,
                labels,
                "difficulty",
                bootstrap_repeats,
                permutation_repeats,
                cluster_starts,
                seed + index * 10 + 1,
            ),
        }
    label_summaries: dict[str, Any] = {}
    for label_name in ("subject", "difficulty"):
        names = list(results)
        p_values = [
            float(results[name][label_name]["permutation"]["two_sided_p_value"]) for name in names
        ]
        q_values = benjamini_hochberg(p_values)
        for name, p_value, q_value in zip(names, p_values, q_values, strict=True):
            results[name][label_name]["permutation"]["benjamini_hochberg_q_value"] = q_value
            results[name][label_name]["permutation"]["bonferroni_p_value"] = min(
                1.0, p_value * len(names)
            )
        effects = [float(results[name][label_name]["record_mean_contrast"]) for name in names]
        ari = [float(results[name][label_name]["clustering"]["ari"]) for name in names]
        nmi = [float(results[name][label_name]["clustering"]["nmi"]) for name in names]
        label_summaries[label_name] = {
            "effect_range": [min(effects), max(effects)],
            "effect_median": float(np.median(effects)),
            "modules_with_bh_q_at_most_0_05": sum(value <= 0.05 for value in q_values),
            "modules_with_bonferroni_p_at_most_0_05": sum(
                min(1.0, value * len(names)) <= 0.05 for value in p_values
            ),
            "ari_range": [min(ari), max(ari)],
            "nmi_range": [min(nmi), max(nmi)],
        }
    return {
        "module_count": len(results),
        "bootstrap_repeats": bootstrap_repeats,
        "permutation_repeats": permutation_repeats,
        "cluster_starts": cluster_starts,
        "seed": seed,
        "label_summaries": label_summaries,
        "modules": results,
    }
