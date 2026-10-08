"""Validate the known lesson inside the configured Manim container image.
This script is the container entry point for the renderer image smoke check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent


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


def instrumented_scene_source(scene_class: str) -> str:
    """Return a wrapper that records geometry without editing generated source.

    Args:
        scene_class: Valid Python class name exported by the generated scene.

    Returns:
        Python source defining an instrumented class with the same name.

    Raises:
        ValueError: If the class name is not a Python identifier.
    """
    if not scene_class.isidentifier():
        raise ValueError("scene class is not a valid Python identifier")
    return dedent(
        f'''\
        """Instrument one generated Manim scene for deterministic geometry tracing.
        This render-only wrapper preserves the admitted source in /work/scene.py."""

        from __future__ import annotations

        import importlib.util
        import sys
        from pathlib import Path

        from manim import config

        sys.path.insert(0, "/work")
        from spatial_trace import SpatialTraceRecorder

        _SPEC = importlib.util.spec_from_file_location("generated_scene", "/work/scene.py")
        if _SPEC is None or _SPEC.loader is None:
            raise RuntimeError("could not load generated scene")
        _MODULE = importlib.util.module_from_spec(_SPEC)
        _SPEC.loader.exec_module(_MODULE)
        _OriginalScene = getattr(_MODULE, "{scene_class}")


        class {scene_class}(_OriginalScene):
            """Record object bounds after stable scene operations."""

            def _trace_recorder(self):
                recorder = getattr(self, "_spatial_trace_recorder", None)
                if recorder is None:
                    recorder = SpatialTraceRecorder(
                        Path("/work/output/spatial_trace.json"),
                        frame_width=float(config.frame_width),
                        frame_height=float(config.frame_height),
                    )
                    self._spatial_trace_recorder = recorder
                return recorder

            def play(self, *args, **kwargs):
                result = super().play(*args, **kwargs)
                self._trace_recorder().record(self, kind="play")
                return result

            def wait(self, *args, **kwargs):
                result = super().wait(*args, **kwargs)
                self._trace_recorder().record(self, kind="wait")
                return result

            def tear_down(self):
                recorder = self._trace_recorder()
                recorder.record(self, kind="final")
                recorder.write()
                return super().tear_down()
        '''
    )


def main() -> None:
    """Render and validate the bundled known scene inside the container."""
    import av

    scene_class = sys.argv[1] if len(sys.argv) >= 2 else "PythagoreanTheorem"
    require_audio = "--require-audio" in sys.argv[2:]
    instrumented_scene = Path("/work/output/instrumented_scene.py")
    instrumented_scene.write_text(
        instrumented_scene_source(scene_class),
        encoding="utf-8",
    )
    command = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        "/work/output/media",
        str(instrumented_scene),
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
