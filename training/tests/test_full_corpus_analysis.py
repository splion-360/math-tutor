"""Test pairwise conflict summaries from identified gradient signatures.
The analysis treats zero gradients as missing directions and preserves topic identity."""

from __future__ import annotations

from dynamic_lora.full_corpus_analysis import summarize_gradient_directions


def test_conflict_summary_distinguishes_within_and_between_topic_pairs() -> None:
    signatures = [
        {"record_id": "a", "layer": "layer_0.q_proj", "signature": [1.0, 0.0]},
        {"record_id": "b", "layer": "layer_0.q_proj", "signature": [0.8, 0.0]},
        {"record_id": "c", "layer": "layer_0.q_proj", "signature": [-1.0, 0.0]},
    ]
    topics = {"a": "geometry", "b": "geometry", "c": "algebra"}

    result = summarize_gradient_directions(signatures, topics)

    layer = result["layer_0.q_proj"]
    assert layer["examples"] == 3
    assert layer["pair_count"] == 3
    assert layer["negative_pairs"] == 2
    assert layer["within_topic_negative_pairs"] == 0
    assert layer["between_topic_negative_pairs"] == 2


def test_zero_gradient_is_not_counted_as_a_direction() -> None:
    signatures = [
        {"record_id": "a", "layer": "layer_0.q_proj", "signature": [1.0, 0.0]},
        {"record_id": "b", "layer": "layer_0.q_proj", "signature": [0.0, 0.0]},
        {"record_id": "c", "layer": "layer_0.q_proj", "signature": [-1.0, 0.0]},
    ]

    result = summarize_gradient_directions(signatures, {key: key for key in "abc"})

    assert result["layer_0.q_proj"]["zero_gradient_count"] == 1
    assert result["layer_0.q_proj"]["pair_count"] == 1
    assert result["layer_0.q_proj"]["negative_pairs"] == 1
