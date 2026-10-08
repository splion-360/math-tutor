"""Verify stream-count requirements for the container render validator.
The tests cover silent and narrated media contracts independently."""

from __future__ import annotations

import pytest

from math_tutor.rendering.scenes.render_known import validate_stream_counts


def test_voiceover_render_requires_audio_stream() -> None:
    with pytest.raises(RuntimeError, match="exactly one audio stream"):
        validate_stream_counts(video_stream_count=1, audio_stream_count=0, require_audio=True)


def test_silent_render_allows_no_audio_stream() -> None:
    validate_stream_counts(video_stream_count=1, audio_stream_count=0, require_audio=False)
