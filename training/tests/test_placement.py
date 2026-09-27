"""Test deterministic adapter-placement controls and validation partitions.
These tests keep module choice independent of held-out evaluation outcomes."""

from __future__ import annotations

from dynamic_lora.placement import (
    assert_exact_trainable_modules,
    build_placement_manifest,
    peft_target_modules,
)


def _probe() -> dict[str, object]:
    energy = {
        f"layer_{layer}.{module}": float(layer + 1)
        for module in ("down_proj", "q_proj", "v_proj")
        for layer in range(12)
    }
    return {
        "probe": {
            "sample_indices": [0, 3, 6],
            "selected_modules": [
                "layer_11.down_proj",
                "layer_11.q_proj",
                "layer_10.q_proj",
                "layer_11.v_proj",
            ],
            "module_energy": energy,
        }
    }


def _records() -> list[dict[str, str]]:
    return [
        {"id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty in ("foundational", "intermediate", "advanced")
        for index in range(20)
    ]


def test_manifest_preselects_matched_discovered_low_and_random_controls() -> None:
    manifest = build_placement_manifest(_probe(), _records(), split_seed=42)

    arms = manifest["arms"]
    assert list(arms) == ["discovered", "low_energy", "random_1", "random_2"]
    assert arms["discovered"] == [
        "layer_11.down_proj",
        "layer_11.q_proj",
        "layer_10.q_proj",
        "layer_11.v_proj",
    ]
    assert arms["low_energy"] == [
        "layer_0.down_proj",
        "layer_0.q_proj",
        "layer_1.q_proj",
        "layer_0.v_proj",
    ]
    expected_mix = {"down_proj": 1, "q_proj": 2, "v_proj": 1}
    for modules in arms.values():
        assert {kind: sum(name.endswith(kind) for name in modules) for kind in expected_mix} == (
            expected_mix
        )
        assert len(set(modules)) == 4
    assert not (set(arms["discovered"]) & set(arms["low_energy"]))
    assert manifest == build_placement_manifest(_probe(), _records(), split_seed=42)


def test_validation_partition_is_stratified_and_excludes_probe_examples() -> None:
    records = _records()

    manifest = build_placement_manifest(_probe(), records, split_seed=42)

    validation_ids = set(manifest["validation_ids"])
    assert len(validation_ids) == 6
    assert {records[index]["id"] for index in (0, 3, 6)}.isdisjoint(validation_ids)
    assert {
        difficulty: sum(item.startswith(difficulty) for item in validation_ids)
        for difficulty in ("foundational", "intermediate", "advanced")
    } == {"foundational": 2, "intermediate": 2, "advanced": 2}
    assert set(manifest["training_ids"]) == {record["id"] for record in records} - validation_ids


def test_manifest_rejects_incomplete_probe_pool() -> None:
    probe = _probe()
    probe["probe"]["module_energy"] = {"layer_0.q_proj": 1.0}  # type: ignore[index]

    try:
        build_placement_manifest(probe, _records(), split_seed=42)
    except ValueError as error:
        assert "missing" in str(error) or "enough" in str(error)
    else:
        raise AssertionError("incomplete probe pool was accepted")


def test_exact_peft_targets_are_checked_against_trainable_parameters() -> None:
    selected = ["layer_6.down_proj", "layer_34.q_proj"]
    assert peft_target_modules(selected) == [
        "model.layers.6.mlp.down_proj",
        "model.layers.34.self_attn.q_proj",
    ]

    class Parameter:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    model = [
        ("base_model.model.model.layers.6.mlp.down_proj.lora_A.default.weight", Parameter(True)),
        ("base_model.model.model.layers.6.mlp.down_proj.lora_B.default.weight", Parameter(True)),
        (
            "base_model.model.model.layers.34.self_attn.q_proj.lora_A.default.weight",
            Parameter(True),
        ),
        (
            "base_model.model.model.layers.34.self_attn.q_proj.lora_B.default.weight",
            Parameter(True),
        ),
        ("base_model.model.model.layers.9.self_attn.v_proj.weight", Parameter(False)),
    ]
    assert assert_exact_trainable_modules(model, selected) == 4
    model.append(("base_model.model.lm_head.weight", Parameter(True)))
    try:
        assert_exact_trainable_modules(model, selected)
    except ValueError as error:
        assert "unexpected trainable" in str(error)
    else:
        raise AssertionError("unexpected trainable weight was accepted")
