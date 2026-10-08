"""Validate the known lesson inside the configured Manim container image.
This script is the container entry point for the renderer image smoke check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def validate_stream_counts(
    *,
    video_stream_count: int,
    audio_stream_count: int,
    require_audio: bool,
) -> None:
    """Require exactly one video stream and any requested audio stream."""
    if video_stream_count != 1:
        raise RuntimeError(f"expected one video stream, found {video_stream_count}")
    if require_audio and audio_stream_count != 1:
        raise RuntimeError(f"expected exactly one audio stream, found {audio_stream_count}")


def main() -> None:
    """Render and validate the bundled known scene inside the container."""
    import av

    scene_class = sys.argv[1] if len(sys.argv) >= 2 else "PythagoreanTheorem"
    require_audio = "--require-audio" in sys.argv[2:]
    command = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        "/work/output/media",
        "/work/scene.py",
        scene_class,
    ]
    subprocess.run(command, check=True)

    videos = list(Path("/work/output/media").rglob(f"{scene_class}.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"expected one rendered video, found {len(videos)}")

    with av.open(str(videos[0])) as container:
        video_streams = list(container.streams.video)
        validate_stream_counts(
            video_stream_count=len(video_streams),
            audio_stream_count=len(container.streams.audio),
            require_audio=require_audio,
        )
        try:
            next(container.decode(video_streams[0]))
        except StopIteration as error:
            raise RuntimeError("rendered video has no decodable frames") from error


if __name__ == "__main__":
    main()
