"""Verify stream-count requirements for the container render validator.
The tests cover silent and narrated media contracts independently."""

from __future__ import annotations

import pytest

from math_tutor.rendering.scenes.render_known import (
    instrumented_scene_source,
    validate_stream_counts,
)


def test_voiceover_render_requires_audio_stream() -> None:
    with pytest.raises(RuntimeError, match="exactly one audio stream"):
        validate_stream_counts(video_stream_count=1, audio_stream_count=0, require_audio=True)


def test_silent_render_allows_no_audio_stream() -> None:
    validate_stream_counts(video_stream_count=1, audio_stream_count=0, require_audio=False)


def test_instrumented_scene_wraps_stable_operations_and_writes_trace() -> None:
    source = instrumented_scene_source("GeneratedLesson")

    assert "class GeneratedLesson(_OriginalScene):" in source
    assert 'self._trace_recorder().record(self, kind="play")' in source
    assert 'self._trace_recorder().record(self, kind="wait")' in source
    assert 'recorder.record(self, kind="final")' in source
    assert "recorder.write()" in source
    compile(source, "instrumented_scene.py", "exec")
