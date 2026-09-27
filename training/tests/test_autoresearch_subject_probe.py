"""Check subject normalization for the cross-subject Manim probe.
The tests keep ambiguous publisher tags out of reported groups."""

from __future__ import annotations

from dynamic_lora.autoresearch_subject_probe import subject_for_tags


def test_subject_tag_aliases() -> None:
    """Map short and long subject tags to the same reported group."""
    assert subject_for_tags(["cs", "duration:10s"]) == "computer science"
    assert subject_for_tags(["computer-science", "cs"]) == "computer science"
    assert subject_for_tags(["mathematics"]) == "math"


def test_missing_or_conflicting_subject_tags() -> None:
    """Exclude rows that lack one unambiguous subject."""
    assert subject_for_tags(["animation", "duration:10s"]) is None
    assert subject_for_tags(["math", "physics"]) is None
