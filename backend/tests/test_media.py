"""Verify FFmpeg media assembly, timelines, captions, and bounded errors.
The tests replace external commands with deterministic local fakes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from math_tutor.rendering.media import (
    FfmpegMediaAssembler,
    MediaAssemblyError,
    probe_audio_duration,
)
from math_tutor.rendering.narration import (
    NarrationPlan,
    NarrationSegment,
    NarrationStatus,
    SynthesizedNarration,
    SynthesizedSegment,
)


def make_narration(tmp_path: Path) -> SynthesizedNarration:
    plan = NarrationPlan(
        lesson_id="lesson",
        segments=(
            NarrationSegment("intro", "Start here.", "intro-visible"),
            NarrationSegment("proof", "Now prove it.", "proof-visible"),
        ),
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    first = audio_dir / "000-intro.mp3"
    second = audio_dir / "001-proof.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    return SynthesizedNarration(
        lesson_id="lesson",
        provider="fake",
        model_id="fake-model",
        plan=plan,
        segments=(
            SynthesizedSegment("intro", "intro-visible", "Start here.", first, 1.25, "a" * 64),
            SynthesizedSegment("proof", "proof-visible", "Now prove it.", second, 2.75, "b" * 64),
        ),
    )


def test_probe_audio_duration_uses_ffprobe_measurement(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")
    commands: list[list[str]] = []

    def runner(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="4.820000\n", stderr="")

    assert probe_audio_duration(audio, command_runner=runner) == 4.82
    assert commands[0][0] == "ffprobe"
    assert commands[0][-1] == str(audio)


def test_assembler_writes_measured_timeline_captions_and_narrated_video(
    tmp_path: Path,
) -> None:
    narration = make_narration(tmp_path)
    silent_video = tmp_path / "silent.mp4"
    silent_video.write_bytes(b"video")
    commands: list[list[str]] = []

    def runner(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"assembled")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    bundle = FfmpegMediaAssembler(
        command_runner=runner,
        duration_probe=lambda path: 5.0 if path == silent_video else 4.0,
    ).assemble(
        silent_video=silent_video,
        narration=narration,
        output_dir=tmp_path / "media",
    )

    assert bundle.narration_status is NarrationStatus.READY
    assert bundle.video_path.read_bytes() == b"assembled"
    assert bundle.silent_video_path == silent_video
    assert bundle.captions_path is not None
    assert bundle.captions_path.read_text(encoding="utf-8") == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.250\nStart here.\n\n"
        "00:00:01.250 --> 00:00:04.000\nNow prove it.\n"
    )
    assert bundle.timeline_path is not None
    timeline = json.loads(bundle.timeline_path.read_text(encoding="utf-8"))
    assert timeline["schema_version"] == "audio-timeline.v1"
    assert [segment["duration_seconds"] for segment in timeline["segments"]] == [1.25, 2.75]
    assert timeline["segments"][0]["audio_file"] == "../audio/000-intro.mp3"
    assert timeline["total_duration_seconds"] == 4.0
    assert len(commands) == 2
    assert "concat=n=2:v=0:a=1" in commands[0]
    assert "[1:a]apad[audio]" in commands[1]
    assert "0:v:0" in commands[1]
    assert "copy" in commands[1]
    assert "-shortest" in commands[1]
    assert commands[1][-1].endswith("narrated.mp4")


def test_assembler_extends_video_when_narration_is_longer(tmp_path: Path) -> None:
    narration = make_narration(tmp_path)
    silent_video = tmp_path / "silent.mp4"
    silent_video.write_bytes(b"video")
    commands: list[list[str]] = []

    def runner(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"assembled")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    bundle = FfmpegMediaAssembler(
        command_runner=runner,
        duration_probe=lambda path: 2.0 if path == silent_video else 4.25,
    ).assemble(
        silent_video=silent_video,
        narration=narration,
        output_dir=tmp_path / "media",
    )

    mux_filter = commands[1][commands[1].index("-filter_complex") + 1]
    assert "tpad=stop_mode=clone:stop_duration=2.250000" in mux_filter
    assert "[1:a]apad[audio]" in mux_filter
    assert "[video]" in commands[1]
    assert "libx264" in commands[1]
    assert bundle.diagnostics["silent_video_duration_seconds"] == 2.0
    assert bundle.diagnostics["assembled_target_duration_seconds"] == 4.25


def test_assembler_reports_sanitized_command_failure(tmp_path: Path) -> None:
    narration = make_narration(tmp_path)
    silent_video = tmp_path / "silent.mp4"
    silent_video.write_bytes(b"video")

    def runner(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="provider-secret")

    with pytest.raises(MediaAssemblyError, match="audio concatenation failed") as captured:
        FfmpegMediaAssembler(command_runner=runner).assemble(
            silent_video=silent_video,
            narration=narration,
            output_dir=tmp_path / "media",
        )

    assert "provider-secret" not in str(captured.value)
