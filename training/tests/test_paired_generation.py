"""Test the deterministic preflight contract for paired lesson generation.
Budget selection and overlap checks must complete before any GPU work."""

from __future__ import annotations

import pytest

from dynamic_lora.paired_generation import audit_overlap, completion_allowance, summarize_lengths


def test_allowance_covers_every_corpus_target_with_declared_headroom() -> None:
    """The observed corpus maximum determines the shared cap, independently of outputs."""
    assert completion_allowance(2886) == 4096
    assert completion_allowance(1000) == 1536
    assert summarize_lengths(list(range(1, 101))) == {
        "count": 100,
        "min": 1,
        "p50": 50,
        "p95": 95,
        "p99": 99,
        "max": 100,
    }


@pytest.mark.parametrize("values", [[], [-1], [True]])
def test_invalid_measurements_cannot_determine_a_budget(values: list[int]) -> None:
    """Missing or invalid measurements are rejected before freezing settings."""
    with pytest.raises(ValueError):
        summarize_lengths(values)


@pytest.mark.parametrize(
    "prompt", ["Solve the simple linear equation now", " SOLVE  the simple linear equation NOW "]
)
def test_exact_prompt_overlap_is_rejected(prompt: str) -> None:
    """IDs alone cannot establish that a prompt is held out."""
    with pytest.raises(ValueError, match="prompts overlap"):
        audit_overlap(
            [{"id": "new", "prompt": prompt}],
            [{"id": "old", "prompt": "Solve the simple linear equation now"}],
        )


def test_near_duplicate_is_rejected() -> None:
    """A long prompt with only a trailing word changed cannot enter the pilot."""
    prompt = " ".join(f"word{i}" for i in range(50))
    with pytest.raises(ValueError, match="near-duplicate"):
        audit_overlap(
            [{"id": "new", "prompt": prompt + " please"}], [{"id": "old", "prompt": prompt}]
        )


def test_unrelated_prompt_records_the_nearest_match() -> None:
    """The audit records lexical similarity without claiming semantic novelty."""
    result = audit_overlap(
        [{"id": "new", "prompt": "Animate a geometric area proof"}],
        [{"id": "old", "prompt": "Show the roots of a polynomial"}],
    )
    assert result[0]["nearest_corpus_id"] == "old"
    assert result[0]["trigram_jaccard"] == 0


def test_freeze_preserves_inputs_and_rejects_mutation(tmp_path, monkeypatch) -> None:
    """The CPU preflight pins hashes, settings, references, and refuses plan overwrites."""
    import hashlib
    import json

    from dynamic_lora import paired_generation

    rows = [
        {
            "id": f"train-{i}",
            "difficulty": group,
            "split": "train",
            "topic": "test",
            "prompt": f"Training problem with unique number {i}",
            "manim_code": "pass",
            "source": {"name": "test", "reference": str(i)},
        }
        for i, group in enumerate(("foundational", "intermediate", "advanced"))
    ]
    examples = [
        {
            "id": f"pilot-{group}-{i}",
            "difficulty": group,
            "split": "evaluation",
            "topic": "test",
            "prompt": f"Distinct authored geometry lesson {group} {i}",
            "review_checks": ["Reference mathematical claim"],
            "source": {"name": "test", "reference": str(i)},
        }
        for group in ("foundational", "intermediate", "advanced")
        for i in range(5)
    ]
    corpus, prompts, evidence, output = [
        tmp_path / name
        for name in (
            "corpus.jsonl",
            "prompts.jsonl",
            "manifest.json",
            "plan.json",
        )
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prompts.write_text("".join(json.dumps(row) + "\n" for row in examples))
    manifest = {
        "dataset": {"prepared_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()},
        "checkpoint": {"sha256": "a" * 64},
        "artifacts": [{"path": "adapter_config.json"}],
        "limitations": [],
    }
    evidence.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        paired_generation,
        "tokenize_completion_record",
        lambda *a, **k: {
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
        },
    )
    kwargs = {
        "corpus_path": corpus,
        "prompts_path": prompts,
        "evidence_path": evidence,
        "output": output,
        "tokenizer": object(),
        "renderer_contract": {"image": "pinned", "timeout_seconds": 90},
    }
    plan = paired_generation.freeze_plan(**kwargs)
    assert plan["decoding"]["max_new_tokens"] == 256
    assert plan["renderer"]["image"] == "pinned"
    assert plan["examples"][0]["review_checks"] == ["Reference mathematical claim"]
    assert plan["corpus_sha256"] == manifest["dataset"]["prepared_sha256"]
    with pytest.raises(FileExistsError):
        paired_generation.freeze_plan(**kwargs)
    output.unlink()
    corpus.write_text(corpus.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        paired_generation.freeze_plan(**kwargs)
    assert not output.exists()
