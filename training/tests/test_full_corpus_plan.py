"""Test full-corpus experiment partitions and coverage requirements.
The tests prevent short probes from being reported as full training runs."""

from __future__ import annotations

from dynamic_lora.full_corpus_plan import build_full_corpus_plan


def test_plan_uses_every_training_id_in_each_epoch_and_holds_out_100() -> None:
    records = [
        {"id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty, count in (("foundational", 336), ("intermediate", 332), ("advanced", 327))
        for index in range(count)
    ]

    plan = build_full_corpus_plan(records, seed=42, epochs=3)

    assert len(plan.training_ids) == 895
    assert len(plan.validation_ids) == 100
    assert len(set(plan.training_ids)) == 895
    assert set(plan.training_ids).isdisjoint(plan.validation_ids)
    assert set(plan.training_ids) | set(plan.validation_ids) == {r["id"] for r in records}
    assert plan.examples_per_epoch == 895
    assert plan.total_training_presentations == 2685


def test_plan_is_stable_when_source_rows_are_reordered() -> None:
    records = [
        {"id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty in ("foundational", "intermediate", "advanced")
        for index in range(20)
    ]

    first = build_full_corpus_plan(records, seed=42, epochs=3)
    reversed_plan = build_full_corpus_plan(list(reversed(records)), seed=42, epochs=3)

    assert set(first.training_ids) == set(reversed_plan.training_ids)
    assert set(first.validation_ids) == set(reversed_plan.validation_ids)
    assert first.total_training_presentations == 162


def test_plan_rejects_duplicate_ids_and_zero_epochs() -> None:
    import pytest

    records = [
        {"id": "one", "difficulty": "foundational"},
        {"id": "one", "difficulty": "intermediate"},
    ]
    with pytest.raises(ValueError, match="unique"):
        build_full_corpus_plan(records, seed=42, epochs=3)
    with pytest.raises(ValueError, match="epochs"):
        build_full_corpus_plan(records, seed=42, epochs=0)
