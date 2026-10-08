"""Render admitted Manim source inside an isolated Docker container.
The renderer returns media paths and operational diagnostics to lesson pipelines."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

from math_tutor.jobs import JobExecutionError, RenderOutcome, is_safe_job_id

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
DEFAULT_MANIM_IMAGE = (
    "manimcommunity/manim@sha256:ab5ad56cf685d89da96e5d459e0cde3743fbdf2141be4dcff6c26566b5ca3191"
)
VOICEOVER_MANIM_IMAGE = "math-tutor-manim-voiceover:local"
_DIGEST_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LOG_LIMIT = 32_000


class RenderError(JobExecutionError):
    """Base operational failure from isolated Manim rendering."""


class RenderTimedOut(RenderError):
    """Raised when the isolated renderer exceeds its configured timeout."""


class RenderFailed(RenderError):
    """Raised when rendering cannot produce acceptable media."""


def run_command(
    command: list[str],
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one renderer command with captured output and an optional environment."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=dict(environment) if environment is not None else None,
    )


class DockerManimRenderer:
    """Render static or generated Manim scenes in a pinned container image."""

    def __init__(
        self,
        artifact_root: Path,
        scene_path: Path,
        image: str = DEFAULT_MANIM_IMAGE,
        timeout_seconds: float = 90,
        command_runner: CommandRunner = run_command,
        network: str = "none",
        environment: Mapping[str, str] | None = None,
        require_audio: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not (_DIGEST_PINNED_IMAGE.fullmatch(image) or image == VOICEOVER_MANIM_IMAGE):
            raise ValueError("Manim image must be digest-pinned")
        if network not in {"none", "bridge"}:
            raise ValueError("renderer network must be 'none' or 'bridge'")
        resolved_environment = dict(environment or {})
        if any(not _ENVIRONMENT_NAME.fullmatch(name) for name in resolved_environment):
            raise ValueError("renderer environment contains an invalid variable name")
        self._artifact_root = artifact_root.resolve()
        self._scene_path = scene_path.resolve()
        self._validation_script = Path(__file__).parent / "scenes" / "render_known.py"
        self._spatial_trace_script = Path(__file__).parent / "scenes" / "spatial_trace.py"
        self._image = image
        self._timeout_seconds = timeout_seconds
        self._run_command = command_runner
        self._network = network
        self._environment = resolved_environment
        self._require_audio = require_audio

    def render(self, job_id: str) -> RenderOutcome:
        """Render the configured known scene for one job."""
        try:
            source = self._scene_path.read_text(encoding="utf-8")
        except OSError as error:
            detail = self._redact(str(error))
            raise self._failed(f"could not read scene source: {detail}", detail) from error
        return self.render_source(job_id, source, "PythagoreanTheorem")

    def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
        """Render admitted source for one generated lesson attempt."""
        if not is_safe_job_id(job_id):
            raise RenderFailed("job id is not safe for an artifact path or container name")
        if not scene_class.isidentifier():
            raise RenderFailed("scene class is not a valid Python identifier")
        job_dir = self._artifact_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        scene_snapshot = job_dir / "scene.py"
        validator_snapshot = job_dir / "render_known.py"
        spatial_trace_snapshot = job_dir / "spatial_trace.py"
        try:
            scene_snapshot.write_text(source, encoding="utf-8")
            shutil.copyfile(self._validation_script, validator_snapshot)
            shutil.copyfile(self._spatial_trace_script, spatial_trace_snapshot)
        except OSError as error:
            self._write_metadata(
                job_dir,
                status="failed",
                command=[],
                started_at=datetime.now(UTC),
                elapsed_seconds=0,
                exit_code=None,
                stdout="",
                stderr=self._redact(str(error)),
                scene_sha256=None,
                validator_sha256=None,
                scene_class=scene_class,
            )
            detail = self._redact(str(error))
            raise self._failed(f"could not snapshot render inputs: {detail}", detail) from error
        scene_digest = sha256(scene_snapshot.read_bytes()).hexdigest()
        validator_digest = sha256(validator_snapshot.read_bytes()).hexdigest()
        output_dir = job_dir / "output"
        output_dir.mkdir()
        container_name = f"math-tutor-render-{job_id}"
        command = self._render_command(
            scene_snapshot,
            validator_snapshot,
            spatial_trace_snapshot,
            output_dir,
            container_name,
            scene_class,
        )
        started_at = datetime.now(UTC)
        started = monotonic()

        try:
            if self._environment:
                result = self._run_command(
                    command,
                    self._timeout_seconds,
                    {**os.environ, **self._environment},
                )
            else:
                result = self._run_command(command, self._timeout_seconds)
        except subprocess.TimeoutExpired as error:
            elapsed = monotonic() - started
            cleanup_succeeded, cleanup_stderr = self._force_remove(container_name)
            stdout = self._redact(_as_text(error.output))
            stderr = self._redact(_as_text(error.stderr))
            cleanup_stderr = self._redact(cleanup_stderr)
            self._write_metadata(
                job_dir,
                status="timed_out",
                command=command,
                started_at=started_at,
                elapsed_seconds=elapsed,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                scene_sha256=scene_digest,
                validator_sha256=validator_digest,
                scene_class=scene_class,
                cleanup_succeeded=cleanup_succeeded,
                cleanup_stderr=cleanup_stderr,
            )
            cleanup_detail = "" if cleanup_succeeded else "; container cleanup failed"
            message = f"Manim render exceeded {self._timeout_seconds} seconds{cleanup_detail}"
            logs = "\n".join(part for part in (stdout, stderr, cleanup_stderr) if part)
            raise RenderTimedOut(
                message,
                diagnostics=self._failure_diagnostics(
                    logs,
                    cleanup_succeeded=cleanup_succeeded,
                ),
            ) from error
        except (OSError, subprocess.SubprocessError) as error:
            elapsed = monotonic() - started
            self._write_metadata(
                job_dir,
                status="failed",
                command=command,
                started_at=started_at,
                elapsed_seconds=elapsed,
                exit_code=None,
                stdout="",
                stderr=self._redact(str(error)),
                scene_sha256=scene_digest,
                validator_sha256=validator_digest,
                scene_class=scene_class,
            )
            detail = self._redact(str(error))
            raise self._failed(f"could not start Manim container: {detail}", detail) from error

        elapsed = monotonic() - started
        stdout = self._redact(_trim(result.stdout))
        stderr = self._redact(_trim(result.stderr))
        if result.returncode != 0:
            self._write_metadata(
                job_dir,
                status="failed",
                command=command,
                started_at=started_at,
                elapsed_seconds=elapsed,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
                scene_sha256=scene_digest,
                validator_sha256=validator_digest,
                scene_class=scene_class,
            )
            detail = stderr or stdout or "no renderer output"
            raise self._failed(
                f"Manim exited with code {result.returncode}: {detail}",
                "\n".join(part for part in (stdout, stderr) if part),
            )

        videos = list((output_dir / "media").rglob(f"{scene_class}.mp4"))
        if len(videos) != 1:
            self._write_metadata(
                job_dir,
                status="failed",
                command=command,
                started_at=started_at,
                elapsed_seconds=elapsed,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
                scene_sha256=scene_digest,
                validator_sha256=validator_digest,
                scene_class=scene_class,
            )
            message = f"expected one rendered video, found {len(videos)}"
            raise self._failed(message, "\n".join(part for part in (stdout, stderr) if part))

        video_path = videos[0]
        safe_video = False
        try:
            resolved_video = video_path.resolve(strict=True)
            resolved_video.relative_to(output_dir.resolve())
            safe_video = not video_path.is_symlink() and resolved_video.is_file()
        except (OSError, ValueError):
            resolved_video = video_path
        if not safe_video:
            self._write_metadata(
                job_dir,
                status="failed",
                command=command,
                started_at=started_at,
                elapsed_seconds=elapsed,
                exit_code=result.returncode,
                stdout=stdout,
                stderr="unsafe rendered video path",
                scene_sha256=scene_digest,
                validator_sha256=validator_digest,
                scene_class=scene_class,
            )
            raise self._failed("unsafe rendered video path", "unsafe rendered video path")

        trace_path = output_dir / "spatial_trace.json"
        safe_trace = False
        try:
            resolved_trace = trace_path.resolve(strict=True)
            resolved_trace.relative_to(output_dir.resolve())
            safe_trace = not trace_path.is_symlink() and resolved_trace.is_file()
        except (OSError, ValueError):
            resolved_trace = trace_path
        if not safe_trace:
            self._write_metadata(
                job_dir,
                status="failed",
                command=command,
                started_at=started_at,
                elapsed_seconds=elapsed,
                exit_code=result.returncode,
                stdout=stdout,
                stderr="spatial trace is missing or unsafe",
                scene_sha256=scene_digest,
                validator_sha256=validator_digest,
                scene_class=scene_class,
            )
            raise self._failed(
                "spatial trace is missing or unsafe",
                "spatial trace is missing or unsafe",
            )

        self._write_metadata(
            job_dir,
            status="ready",
            command=command,
            started_at=started_at,
            elapsed_seconds=elapsed,
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            scene_sha256=scene_digest,
            validator_sha256=validator_digest,
            scene_class=scene_class,
        )
        return RenderOutcome(
            video_path=resolved_video,
            renderer=f"docker:{self._image}",
            elapsed_seconds=elapsed,
            logs="\n".join(part for part in (stdout, stderr) if part),
            spatial_trace_path=resolved_trace,
        )

    def _render_command(
        self,
        scene_snapshot: Path,
        validator_snapshot: Path,
        spatial_trace_snapshot: Path,
        output_dir: Path,
        container_name: str,
        scene_class: str,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            self._network,
            "--cpus",
            "1.0",
            "--memory",
            "1g",
            "--memory-swap",
            "1g",
            "--pids-limit",
            "256",
            "--ulimit",
            "fsize=536870912",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--env",
            "HOME=/tmp",
            *[option for name in self._environment for option in ("--env", name)],
            "--volume",
            f"{scene_snapshot.resolve()}:/work/scene.py:ro",
            "--volume",
            f"{validator_snapshot.resolve()}:/work/render_known.py:ro",
            "--volume",
            f"{spatial_trace_snapshot.resolve()}:/work/spatial_trace.py:ro",
            "--volume",
            f"{output_dir.resolve()}:/work/output:rw",
            "--workdir",
            "/work/output",
            self._image,
            "python",
            "/work/render_known.py",
            scene_class,
        ]
        if self._require_audio:
            command.append("--require-audio")
        return command

    def _force_remove(self, container_name: str) -> tuple[bool, str]:
        try:
            result = self._run_command(["docker", "rm", "-f", container_name], 10)
        except (OSError, subprocess.SubprocessError) as error:
            return False, self._redact(str(error))
        detail = self._redact(_trim(result.stderr) or _trim(result.stdout))
        return result.returncode == 0, detail

    def _redact(self, value: str) -> str:
        for secret in self._environment.values():
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value

    def _write_metadata(
        self,
        job_dir: Path,
        *,
        status: str,
        command: list[str],
        started_at: datetime,
        elapsed_seconds: float,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        scene_sha256: str | None,
        validator_sha256: str | None,
        scene_class: str,
        cleanup_succeeded: bool | None = None,
        cleanup_stderr: str = "",
    ) -> None:
        metadata: dict[str, Any] = {
            "status": status,
            "image": self._image,
            "scene": "scene.py",
            "scene_sha256": scene_sha256,
            "validator": "render_known.py",
            "validator_sha256": validator_sha256,
            "spatial_trace_recorder": "spatial_trace.py",
            "spatial_trace_recorder_sha256": _optional_sha256(job_dir / "spatial_trace.py"),
            "scene_class": scene_class,
            "command": command,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "timeout_seconds": self._timeout_seconds,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "cleanup_succeeded": cleanup_succeeded,
            "cleanup_stderr": cleanup_stderr,
        }
        (job_dir / "render.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _failure_diagnostics(self, logs: str, **extra: object) -> dict[str, object]:
        return {
            "renderer": f"docker:{self._image}",
            "logs": _trim(logs),
            "metadata_file": "render.json",
            **extra,
        }

    def _failed(self, message: str, logs: str) -> RenderFailed:
        return RenderFailed(message, diagnostics=self._failure_diagnostics(logs))


def _trim(value: str | None) -> str:
    return (value or "")[-_LOG_LIMIT:]


def _as_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return _trim(value.decode(errors="replace"))
    return _trim(value)


def _optional_sha256(path: Path) -> str | None:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else None
