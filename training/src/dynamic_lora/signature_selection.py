"""Choose signature modules from a saved no-update base probe.
This module validates probe provenance before the Qwen model is loaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dynamic_lora.placement_run import validate_placement_probe


def select_base_probe_modules(
    path: Path,
    *,
    train_hash: str,
    holdout_hash: str,
    max_seq_length: int,
    seed: int,
    target_modules: tuple[str, ...],
) -> tuple[str, ...]:
    """Return selected module keys from a matching saved base probe.

    Args:
        path: Saved no-update base-probe metadata.
        train_hash: SHA-256 of the current training data.
        holdout_hash: SHA-256 of the current evaluation data.
        max_seq_length: Current complete-example token ceiling.
        seed: Current experiment seed.
        target_modules: LoRA projection types attached by this run.

    Returns:
        Probe-selected module keys, preserving their rank order.

    Raises:
        ValueError: If provenance or module selection does not match.
    """
    metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    validate_placement_probe(
        metadata,
        train_hash=train_hash,
        holdout_hash=holdout_hash,
        max_seq_length=max_seq_length,
    )
    if metadata.get("probe_config", {}).get("seed") != seed:
        raise ValueError("base probe seed does not match")
    modules = metadata.get("probe", {}).get("selected_modules")
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(module, str) and module for module in modules)
        or len(set(modules)) != len(modules)
    ):
        raise ValueError("base probe has invalid selected modules")
    if any(module.rpartition(".")[2] not in target_modules for module in modules):
        raise ValueError("base probe selected modules are not LoRA target modules")
    return tuple(modules)
