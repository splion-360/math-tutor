"""Choose matched LoRA placements and a fixed supervised validation split.
All choices derive from the training probe and never inspect evaluation outcomes."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, TypedDict

from dynamic_lora.lora_parameters import is_lora_parameter, lora_layer_key

MODULE_KINDS = ("down_proj", "q_proj", "v_proj")
MODULE_KEY = re.compile(r"^layer_(\d+)\.(down_proj|q_proj|v_proj)$")


class PlacementManifest(TypedDict):
    """Fixed arms and disjoint record IDs for one placement experiment."""

    arms: dict[str, list[str]]
    training_ids: list[str]
    validation_ids: list[str]
    probe_ids: list[str]
    split_seed: int
    random_seeds: list[int]


def peft_target_modules(selected: list[str]) -> list[str]:
    """Convert stable probe keys into exact PEFT module suffixes.

    Args:
        selected: Four layer-and-projection keys from a placement arm.

    Returns:
        Model module paths that PEFT must adapt.

    Raises:
        ValueError: If a key is not a supported projection or names are repeated.
    """
    if len(selected) != len(set(selected)):
        raise ValueError("placement modules must be distinct")
    targets: list[str] = []
    for name in selected:
        match = MODULE_KEY.fullmatch(name)
        if match is None:
            raise ValueError(f"unsupported placement module: {name}")
        layer, kind = match.groups()
        parent = "mlp" if kind == "down_proj" else "self_attn"
        targets.append(f"model.layers.{layer}.{parent}.{kind}")
    return targets


def assert_exact_trainable_modules(
    named_parameters: Iterable[tuple[str, Any]], selected: list[str]
) -> int:
    """Fail if PEFT trains any weight outside the selected modules.

    Args:
        named_parameters: Model parameter names paired with requires_grad-bearing values.
        selected: Stable module keys for the intended arm.

    Returns:
        Count of trainable LoRA parameter tensors.

    Raises:
        ValueError: If selected modules are missing or unrelated weights are trainable.
    """
    expected = set(selected)
    observed: Counter[str] = Counter()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if not is_lora_parameter(name):
            raise ValueError(f"unexpected trainable non-LoRA parameter: {name}")
        key = lora_layer_key(name)
        if key not in expected:
            raise ValueError(f"unexpected trainable LoRA module: {key}")
        observed[key] += 1
    if set(observed) != expected or any(count < 2 for count in observed.values()):
        raise ValueError(f"trainable LoRA modules do not match placement: {dict(observed)}")
    return sum(observed.values())


def build_placement_manifest(
    probe_metadata: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    split_seed: int,
) -> PlacementManifest:
    """Preselect four matched arms and reserve 10% per difficulty for validation.

    Args:
        probe_metadata: Completed base-weight probe result with per-module energy.
        records: Validated training records in the order probed.
        split_seed: Seed for validation hashing and random control selection.

    Returns:
        Module names for each arm and disjoint training/validation IDs.

    Raises:
        ValueError: If the probe cannot support matched, disjoint controls.
    """
    probe = probe_metadata["probe"]
    discovered: list[str] = list(probe["selected_modules"])
    energies: dict[str, float] = probe["module_energy"]
    if len(discovered) != 4 or len(set(discovered)) != 4:
        raise ValueError("probe must select four distinct modules")
    mix = Counter(name.rsplit(".", 1)[-1] for name in discovered)
    if mix != {"down_proj": 1, "q_proj": 2, "v_proj": 1}:
        raise ValueError("probe module mix does not match the placement experiment")
    if any(name not in energies for name in discovered):
        raise ValueError("probe is missing energy for a selected module")

    excluded = set(discovered)
    low: list[str] = []
    for kind in MODULE_KINDS:
        candidates = sorted(
            (name for name in energies if name.endswith(f".{kind}") and name not in excluded),
            key=lambda name: (energies[name], name),
        )
        if len(candidates) < mix[kind]:
            raise ValueError(f"not enough {kind} modules for low-energy control")
        low.extend(candidates[: mix[kind]])
    excluded.update(low)

    arms = {"discovered": discovered, "low_energy": low}
    random_seeds = [split_seed + 1000, split_seed + 2000]
    for arm_index, seed in enumerate(random_seeds, start=1):
        rng = random.Random(seed)
        selected: list[str] = []
        for kind in MODULE_KINDS:
            candidates = sorted(
                name for name in energies if name.endswith(f".{kind}") and name not in excluded
            )
            if len(candidates) < mix[kind]:
                raise ValueError(f"not enough {kind} modules for random controls")
            selected.extend(rng.sample(candidates, mix[kind]))
        arms[f"random_{arm_index}"] = selected
        excluded.update(selected)

    probe_indices: list[int] = probe["sample_indices"]
    if any(index < 0 or index >= len(records) for index in probe_indices):
        raise ValueError("probe sample index is outside the training dataset")
    probe_ids = [str(records[index]["id"]) for index in probe_indices]
    candidates_by_difficulty: dict[str, list[str]] = defaultdict(list)
    for record in records:
        record_id = str(record["id"])
        if record_id not in probe_ids:
            candidates_by_difficulty[str(record["difficulty"])].append(record_id)
    validation_ids: set[str] = set()
    for difficulty, candidates in sorted(candidates_by_difficulty.items()):
        count = max(1, round(sum(record["difficulty"] == difficulty for record in records) * 0.1))
        if len(candidates) < count:
            raise ValueError(f"not enough {difficulty} records for validation")
        ranked = sorted(
            candidates,
            key=lambda record_id: (
                hashlib.sha256(f"{split_seed}:{record_id}".encode()).hexdigest(),
                record_id,
            ),
        )
        validation_ids.update(ranked[:count])
    if len(candidates_by_difficulty) != 3:
        raise ValueError("validation split must cover all three difficulties")
    return {
        "arms": arms,
        "training_ids": [
            str(record["id"]) for record in records if record["id"] not in validation_ids
        ],
        "validation_ids": sorted(validation_ids),
        "probe_ids": probe_ids,
        "split_seed": split_seed,
        "random_seeds": random_seeds,
    }
