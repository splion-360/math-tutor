"""Inspect media streams and combine rendered video with narration and captions.
FFmpeg operations stay behind a narrow assembler used by narration orchestration."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from math_tutor.domain import NarrationStatus
from math_tutor.rendering.narration import MediaBundle, SynthesizedNarration

CommandRunner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


class MediaAssemblyError(RuntimeError):
    """A sanitized media-stage failure safe for job diagnostics."""


def run_command(command: list[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    """Run one media command with captured text output and a timeout."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def probe_audio_duration(
    audio_path: Path,
    *,
    command_runner: CommandRunner = run_command,
    timeout_seconds: float = 10.0,
) -> float:
    """Measure a positive audio duration with ffprobe.

    Args:
        audio_path: Audio artifact to inspect.
        command_runner: Injectable ffprobe command boundary.
        timeout_seconds: Maximum command duration.

    Returns:
        Measured duration in seconds.

    Raises:
        MediaAssemblyError: If ffprobe cannot return a valid positive duration.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = command_runner(command, timeout_seconds)
        duration = float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        raise MediaAssemblyError("could not measure synthesized audio duration") from None
    if result.returncode != 0 or not math.isfinite(duration) or duration <= 0:
        raise MediaAssemblyError("could not measure synthesized audio duration")
    return duration


class FfmpegMediaAssembler:
    """Create narration timelines, captions, audio, and final muxed videos."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner = run_command,
        duration_probe: Callable[[Path], float] = probe_audio_duration,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Configure FFmpeg execution and media-duration inspection.

        Args:
            command_runner: Injectable FFmpeg command boundary.
            duration_probe: Function used to measure video and audio artifacts.
            timeout_seconds: Maximum duration for each FFmpeg command.

        Raises:
            ValueError: If the timeout is not positive.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._run_command = command_runner
        self._duration_probe = duration_probe
        self._timeout_seconds = timeout_seconds

    def assemble(
        self,
        *,
        silent_video: Path,
        narration: SynthesizedNarration,
        output_dir: Path,
    ) -> MediaBundle:
        """Combine a silent video with synthesized narration artifacts.

        Args:
            silent_video: Existing rendered video without narration.
            narration: Ordered synthesized narration segments.
            output_dir: Directory for the assembled media.

        Returns:
            Paths and diagnostics for the assembled media.

        Raises:
            MediaAssemblyError: If an input is missing or an FFmpeg command fails.
        """
        if not silent_video.is_file():
            raise MediaAssemblyError("silent video is unavailable")
        output_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = output_dir / "audio-timeline.json"
        captions_path = output_dir / "captions.vtt"
        concatenated_audio = output_dir / "narration.mp3"
        narrated_video = output_dir / "narrated.mp4"

        timeline_path.write_text(
            json.dumps(
                _timeline_payload(narration, relative_to=timeline_path.parent),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        captions_path.write_text(_webvtt(narration), encoding="utf-8")

        concat_command = ["ffmpeg", "-y"]
        for segment in narration.segments:
            concat_command.extend(("-i", str(segment.audio_path)))
        concat_command.extend(
            (
                "-filter_complex",
                f"concat=n={len(narration.segments)}:v=0:a=1",
                "-c:a",
                "libmp3lame",
                str(concatenated_audio),
            )
        )
        self._execute(concat_command, "audio concatenation failed")

        silent_video_duration = self._duration_probe(silent_video)
        audio_duration = self._duration_probe(concatenated_audio)
        caption_duration = sum(segment.duration_seconds for segment in narration.segments)
        target_duration = max(silent_video_duration, audio_duration, caption_duration)
        video_extension = target_duration - silent_video_duration
        mux_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(concatenated_audio),
        ]
        if video_extension > 0:
            mux_command.extend(
                (
                    "-filter_complex",
                    (
                        "[0:v]tpad=stop_mode=clone:"
                        f"stop_duration={video_extension:.6f}[video];[1:a]apad[audio]"
                    ),
                    "-map",
                    "[video]",
                    "-map",
                    "[audio]",
                    "-c:v",
                    "libx264",
                )
            )
        else:
            mux_command.extend(
                (
                    "-filter_complex",
                    "[1:a]apad[audio]",
                    "-map",
                    "0:v:0",
                    "-map",
                    "[audio]",
                    "-c:v",
                    "copy",
                )
            )
        mux_command.extend(("-c:a", "aac", "-shortest", str(narrated_video)))
        self._execute(mux_command, "video muxing failed")

        return MediaBundle(
            silent_video_path=silent_video,
            video_path=narrated_video,
            narration_status=NarrationStatus.READY,
            captions_path=captions_path,
            timeline_path=timeline_path,
            diagnostics={
                "narration_provider": narration.provider,
                "narration_model": narration.model_id,
                "narration_segments": len(narration.segments),
                "narration_duration_seconds": sum(
                    segment.duration_seconds for segment in narration.segments
                ),
                "silent_video_duration_seconds": silent_video_duration,
                "assembled_target_duration_seconds": target_duration,
            },
        )

    def _execute(self, command: list[str], message: str) -> None:
        try:
            result = self._run_command(command, self._timeout_seconds)
        except (OSError, subprocess.SubprocessError):
            raise MediaAssemblyError(message) from None
        if result.returncode != 0:
            raise MediaAssemblyError(message)
        output_path = Path(command[-1])
        if not output_path.is_file():
            raise MediaAssemblyError(message)


def _timeline_payload(
    narration: SynthesizedNarration,
    *,
    relative_to: Path,
) -> dict[str, object]:
    return {
        "schema_version": narration.schema_version,
        "lesson_id": narration.lesson_id,
        "provider": narration.provider,
        "model_id": narration.model_id,
        "segments": [
            {
                "id": segment.id,
                "cue": segment.cue,
                "text": segment.text,
                "audio_file": Path(
                    os.path.relpath(segment.audio_path, start=relative_to)
                ).as_posix(),
                "duration_seconds": segment.duration_seconds,
                "sha256": segment.sha256,
            }
            for segment in narration.segments
        ],
        "total_duration_seconds": sum(segment.duration_seconds for segment in narration.segments),
    }


def _webvtt(narration: SynthesizedNarration) -> str:
    lines = ["WEBVTT", ""]
    start = 0.0
    for segment in narration.segments:
        end = start + segment.duration_seconds
        lines.extend(
            (
                f"{_timestamp(start)} --> {_timestamp(end)}",
                segment.text,
                "",
            )
        )
        start = end
    return "\n".join(lines).rstrip() + "\n"


def _timestamp(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
