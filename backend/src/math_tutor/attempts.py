"""Describe rendered attempts and persist their immutable evidence bundles.
Validation consumes attempt values while orchestration owns the artifact store."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from math_tutor.jobs import RenderOutcome, is_safe_job_id


@dataclass(frozen=True)
class RenderedAttempt:
    """Inputs and outputs needed to validate one rendered lesson attempt.

    Args:
        number: Zero-based attempt number within the lesson job.
        artifact_dir: Directory containing immutable evidence for this attempt.
        prompt: Prompt sent to the generator for this attempt.
        source: Admitted Manim source used for rendering.
        scene_class: Manim scene class selected for rendering.
        outcome: Rendered media and renderer provenance.
        narration_required: Whether the final video must contain an audio stream.
        captions_required: Whether a non-empty WebVTT artifact is required.
        original_prompt_path: Immutable copy of the original user prompt.
        generation_model: Model identifier recorded for this attempt.
        generation_provider: Provider identifier recorded for this attempt.
        inference_path: Routing path used to produce this attempt.
        infrastructure_retry_count: Render retries that reused the admitted source.
    """

    number: int
    artifact_dir: Path
    prompt: str
    source: str
    scene_class: str
    outcome: RenderOutcome
    narration_required: bool
    captions_required: bool
    original_prompt_path: Path | None = None
    generation_model: str | None = None
    generation_provider: str | None = None
    inference_path: str = "unknown"
    infrastructure_retry_count: int = 0


class AttemptArtifactStore:
    """Persist immutable inputs, outputs, manifests, and final attempt selection."""

    def __init__(self, *, artifact_root: Path, job_id: str, original_prompt: str) -> None:
        """Create the artifact directories for one lesson job.

        Args:
            artifact_root: Root directory for all lesson artifacts.
            job_id: Safe job identifier used as the directory name.
            original_prompt: Unmodified user request for the lesson.

        Raises:
            ValueError: If the job identifier is unsafe.
            FileExistsError: If the job artifact directory already exists.
        """
        if not is_safe_job_id(job_id):
            raise ValueError("job id is not safe for an artifact path")
        self.job_dir = artifact_root.resolve() / job_id
        self.job_dir.mkdir(parents=True, exist_ok=False)
        self.original_prompt_path = self.job_dir / "prompt.txt"
        self.original_prompt_path.write_text(original_prompt, encoding="utf-8")
        self._attempts_dir = self.job_dir / "attempts"
        self._attempts_dir.mkdir()

    def start_attempt(self, number: int, prompt: str) -> Path:
        """Create a separate immutable directory for one generation attempt.

        Args:
            number: Zero-based attempt number.
            prompt: Generation prompt used for this attempt.

        Returns:
            Newly created attempt directory.

        Raises:
            FileExistsError: If the attempt directory already exists.
        """
        attempt_dir = self._attempts_dir / str(number)
        attempt_dir.mkdir()
        (attempt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        return attempt_dir

    def write_attempt_manifest(
        self,
        attempt: RenderedAttempt,
        *,
        expected_checks: tuple[str, ...],
        validation_status: str,
    ) -> Path:
        """Write the complete evidence manifest for a rendered attempt.

        Args:
            attempt: Rendered attempt whose evidence must be indexed.
            expected_checks: Stable checks expected for this attempt.
            validation_status: Final status reported by validation.

        Returns:
            Path to the written attempt manifest.
        """
        manifest = self._base_manifest(
            attempt_dir=attempt.artifact_dir,
            number=attempt.number,
            status=validation_status,
            generation_model=attempt.generation_model,
            generation_provider=attempt.generation_provider,
            inference_path=attempt.inference_path,
            renderer=attempt.outcome.renderer,
            infrastructure_retry_count=attempt.infrastructure_retry_count,
            expected_checks=expected_checks,
        )
        manifest.update(
            {
                "source": self._optional_file_evidence(
                    attempt.artifact_dir / "extracted_scene.py"
                ),
                "render": {
                    "renderer": attempt.outcome.renderer,
                    "elapsed_seconds": attempt.outcome.elapsed_seconds,
                    "logs": self._optional_file_evidence(
                        attempt.artifact_dir / "render.log"
                    ),
                },
                "media": {
                    "video": self._file_evidence(attempt.outcome.video_path),
                    "silent_video": self._optional_file_evidence(
                        attempt.outcome.silent_video_path
                    ),
                    "captions": self._optional_file_evidence(
                        attempt.outcome.captions_path
                    ),
                    "narration_required": attempt.narration_required,
                    "captions_required": attempt.captions_required,
                },
            }
        )
        return self._write_manifest(attempt.artifact_dir, manifest)

    def write_failed_attempt_manifest(
        self,
        *,
        attempt_dir: Path,
        number: int,
        status: str,
        generation_model: str | None,
        generation_provider: str | None,
        inference_path: str,
        expected_checks: tuple[str, ...],
        failure: dict[str, object],
    ) -> Path:
        """Write available evidence for an attempt that failed before publication.

        Args:
            attempt_dir: Evidence directory containing the available artifacts.
            number: Zero-based attempt number.
            status: Stable terminal status for this attempt.
            generation_model: Model identifier, when generation started.
            generation_provider: Provider identifier, when generation started.
            inference_path: Routing path used for generation.
            expected_checks: Checks that would have run after rendering.
            failure: Structured failure evidence safe for the manifest.

        Returns:
            Path to the written attempt manifest.
        """
        retry_count = failure.get("infrastructure_retry_count", 0)
        if not isinstance(retry_count, int):
            retry_count = 0
        manifest = self._base_manifest(
            attempt_dir=attempt_dir,
            number=number,
            status=status,
            generation_model=generation_model,
            generation_provider=generation_provider,
            inference_path=inference_path,
            renderer=None,
            infrastructure_retry_count=retry_count,
            expected_checks=expected_checks,
        )
        manifest.update(
            {
                "source": self._optional_file_evidence(
                    attempt_dir / "extracted_scene.py"
                ),
                "render": None,
                "media": None,
                "failure": failure,
            }
        )
        return self._write_manifest(attempt_dir, manifest)

    def select_attempt(
        self,
        attempt: RenderedAttempt,
        *,
        validation_status: str,
        infrastructure_retry_count: int,
    ) -> RenderOutcome:
        """Record and return the only attempt accepted for publication.

        Args:
            attempt: Validated rendered attempt selected for publication.
            validation_status: Passing status recorded in the selection summary.
            infrastructure_retry_count: Total render retries across the lesson job.

        Returns:
            Render outcome augmented with validation and repair diagnostics.
        """
        video_evidence = self._file_evidence(attempt.outcome.video_path)
        summary = {
            "attempt_number": attempt.number,
            "attempt_count": attempt.number + 1,
            "repair_count": attempt.number,
            "infrastructure_retry_count": infrastructure_retry_count,
            "validation_status": validation_status,
            "video_path": video_evidence["path"],
            "video_sha256": video_evidence["sha256"],
            "attempt_manifest": str((attempt.artifact_dir / "attempt.json").resolve()),
        }
        (self.job_dir / "selected_attempt.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return replace(
            attempt.outcome,
            validation_diagnostics={
                "attempt_count": attempt.number + 1,
                "repair_count": attempt.number,
                "infrastructure_retry_count": infrastructure_retry_count,
                "selected_attempt": attempt.number,
                "validation_status": validation_status,
            },
        )

    def _base_manifest(
        self,
        *,
        attempt_dir: Path,
        number: int,
        status: str,
        generation_model: str | None,
        generation_provider: str | None,
        inference_path: str,
        renderer: str | None,
        infrastructure_retry_count: int,
        expected_checks: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "schema_version": "rendered-attempt.v1",
            "attempt_number": number,
            "original_prompt": self._file_evidence(self.original_prompt_path),
            "generation_prompt": self._file_evidence(attempt_dir / "prompt.txt"),
            "raw_response": self._optional_file_evidence(attempt_dir / "raw_response.txt"),
            "provider_response": self._optional_file_evidence(
                attempt_dir / "provider_response.json"
            ),
            "generation": self._optional_file_evidence(attempt_dir / "generation.json"),
            "provenance": {
                "model": generation_model,
                "provider": generation_provider,
                "inference_path": inference_path,
                "renderer": renderer,
                "infrastructure_retry_count": infrastructure_retry_count,
            },
            "expected_checks": list(expected_checks),
            "validation_status": status,
        }

    @staticmethod
    def _write_manifest(attempt_dir: Path, manifest: dict[str, object]) -> Path:
        path = attempt_dir / "attempt.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def _file_evidence(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    @classmethod
    def _optional_file_evidence(cls, path: Path | None) -> dict[str, object] | None:
        return cls._file_evidence(path) if path is not None and path.is_file() else None
