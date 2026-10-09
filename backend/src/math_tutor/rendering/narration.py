"""Define narration plans and coordinate narration with rendered lesson media.
Provider and assembler protocols keep orchestration independent of external services."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from math_tutor.domain import NarrationStatus
from math_tutor.jobs import JobExecutionError, JobRenderer, PartialOutcome, RenderOutcome


@dataclass(frozen=True)
class NarrationSegment:
    """One narration line and the visual cue it explains."""

    id: str
    text: str
    cue: str

    def __post_init__(self) -> None:
        for name, value in (("id", self.id), ("text", self.text), ("cue", self.cue)):
            if not value.strip():
                raise ValueError(f"narration segment {name} must not be blank")


@dataclass(frozen=True)
class NarrationPlan:
    """Ordered narration segments for one lesson."""

    lesson_id: str
    segments: tuple[NarrationSegment, ...]
    schema_version: str = field(default="narration-plan.v1", init=False)

    def __post_init__(self) -> None:
        if not self.lesson_id.strip():
            raise ValueError("lesson_id must not be blank")
        if not self.segments:
            raise ValueError("narration plan needs at least one segment")
        segment_ids = [segment.id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("narration segment ids must be unique")


@dataclass(frozen=True)
class PlannedNarration:
    """Narration plan and provider evidence produced by a language model."""

    plan: NarrationPlan
    model: str
    raw_response: str = field(repr=False)
    provider_response: str = field(repr=False)


@dataclass(frozen=True)
class SynthesizedSegment:
    """Audio artifact and provenance for one narration segment."""

    id: str
    cue: str
    text: str
    audio_path: Path
    duration_seconds: float
    sha256: str

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("sha256 must be a 64-character hexadecimal digest") from error


@dataclass(frozen=True)
class SynthesizedNarration:
    """Synthesized narration that preserves the source plan order."""

    lesson_id: str
    provider: str
    model_id: str
    segments: tuple[SynthesizedSegment, ...]
    plan: NarrationPlan = field(repr=False, compare=False)
    schema_version: str = field(default="audio-timeline.v1", init=False)

    def __post_init__(self) -> None:
        if self.lesson_id != self.plan.lesson_id:
            raise ValueError("synthesized narration lesson must match its plan")
        expected = [segment.id for segment in self.plan.segments]
        actual = [segment.id for segment in self.segments]
        if actual != expected:
            raise ValueError("synthesized segments must match plan order")


@dataclass(frozen=True)
class MediaBundle:
    """Assembled lesson media and optional narration artifacts."""

    silent_video_path: Path
    video_path: Path
    narration_status: NarrationStatus
    captions_path: Path | None = None
    timeline_path: Path | None = None
    diagnostics: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


class NarrationProvider(Protocol):
    """Synthesize the segments in a narration plan."""

    def synthesize(
        self,
        plan: NarrationPlan,
        output_dir: Path,
    ) -> SynthesizedNarration:
        """Create audio artifacts for every segment in a plan."""
        ...


class MediaAssembler(Protocol):
    """Combine a silent render with synthesized narration artifacts."""

    def assemble(
        self,
        *,
        silent_video: Path,
        narration: SynthesizedNarration,
        output_dir: Path,
    ) -> MediaBundle:
        """Combine silent video and narration into a media bundle."""
        ...


class NarrationPlanner(Protocol):
    """Generate a bounded narration plan from a rendered lesson description."""

    def create_plan(
        self,
        *,
        lesson_id: str,
        prompt: str,
        source: str,
        target_duration_seconds: float,
    ) -> PlannedNarration:
        """Return narration text for one rendered lesson."""
        ...


class NarrationOutcomeProcessor:
    """Generate, synthesize, and attach narration after Manim rendering."""

    def __init__(
        self,
        *,
        planner: NarrationPlanner,
        provider: NarrationProvider,
        assembler: MediaAssembler,
        duration_probe: Callable[[Path], float],
    ) -> None:
        self._planner = planner
        self._provider = provider
        self._assembler = assembler
        self._duration_probe = duration_probe

    def process(
        self,
        *,
        job_id: str,
        prompt: str,
        source: str,
        outcome: RenderOutcome,
        artifact_dir: Path,
    ) -> RenderOutcome:
        """Attach model-planned narration to a completed silent render.

        Args:
            job_id: Parent lesson job identifier.
            prompt: Original lesson request.
            source: Admitted Manim source used for the render.
            outcome: Completed silent-render outcome.
            artifact_dir: Immutable attempt directory for narration evidence.

        Returns:
            Render outcome pointing to narrated media and captions.

        Raises:
            JobExecutionError: If planning, synthesis, or assembly fails.
        """
        try:
            video_duration = self._duration_probe(outcome.video_path)
            planned = self._planner.create_plan(
                lesson_id=job_id,
                prompt=prompt,
                source=source,
                target_duration_seconds=video_duration,
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "narration-plan.json").write_text(
                json.dumps(
                    {
                        "schema_version": planned.plan.schema_version,
                        "lesson_id": planned.plan.lesson_id,
                        "model": planned.model,
                        "segments": [
                            {
                                "id": segment.id,
                                "text": segment.text,
                                "cue": segment.cue,
                            }
                            for segment in planned.plan.segments
                        ],
                        "target_duration_seconds": video_duration,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (artifact_dir / "narration-response.txt").write_text(
                planned.raw_response,
                encoding="utf-8",
            )
            (artifact_dir / "narration-provider-response.json").write_text(
                planned.provider_response,
                encoding="utf-8",
            )
            narration = self._provider.synthesize(planned.plan, artifact_dir / "audio")
            bundle = self._assembler.assemble(
                silent_video=outcome.video_path,
                narration=narration,
                output_dir=artifact_dir / "narration",
            )
        except JobExecutionError:
            raise
        except Exception as error:
            raise JobExecutionError(
                "Narration could not be attached to the lesson",
                diagnostics={
                    "failure_stage": "narration",
                    "failure_kind": "operational",
                    "narration_status": NarrationStatus.UNAVAILABLE.value,
                },
            ) from error
        return replace(
            outcome,
            video_path=bundle.video_path,
            silent_video_path=bundle.silent_video_path,
            captions_path=bundle.captions_path,
            narration_status=bundle.narration_status,
            narration_diagnostics={
                **dict(bundle.diagnostics),
                "narration_plan_model": planned.model,
                "narration_target_duration_seconds": video_duration,
            },
        )


class SourceRenderer(Protocol):
    """Render one admitted Manim source artifact."""

    def render_source(
        self,
        job_id: str,
        source: str,
        scene_class: str,
    ) -> RenderOutcome:
        """Render admitted source and return its media outcome."""
        ...


class NarratedSourceRenderer:
    """Synthesize narration outside generated code, then render and mux it."""

    def __init__(
        self,
        *,
        renderer: SourceRenderer,
        provider: NarrationProvider,
        assembler: MediaAssembler,
        artifact_root: Path,
    ) -> None:
        self._renderer = renderer
        self._provider = provider
        self._assembler = assembler
        self._artifact_root = artifact_root

    def render_source(
        self,
        job_id: str,
        source: str,
        scene_class: str,
    ) -> RenderOutcome:
        """Render a voiceover scene without exposing credentials to its container.

        Args:
            job_id: Safe job identifier used for narration artifacts.
            source: Admitted VoiceoverScene source containing literal narration text.
            scene_class: Renderable class name from source admission.

        Returns:
            Render outcome containing the muxed narration and caption artifacts.
        """
        plan = _voiceover_plan(job_id, source)
        job_dir = self._artifact_root / job_id
        narration = self._provider.synthesize(plan, job_dir / "audio")
        render_source = _silent_scene_source(source, narration)
        outcome = self._renderer.render_source(job_id, render_source, scene_class)
        bundle = self._assembler.assemble(
            silent_video=outcome.video_path,
            narration=narration,
            output_dir=job_dir / "narration",
        )
        return replace(
            outcome,
            video_path=bundle.video_path,
            silent_video_path=bundle.silent_video_path,
            captions_path=bundle.captions_path,
            narration_status=bundle.narration_status,
            narration_diagnostics=bundle.diagnostics,
        )


def _voiceover_plan(job_id: str, source: str) -> NarrationPlan:
    tree = ast.parse(source)
    collector = _VoiceoverCollector()
    collector.visit(tree)
    if not 3 <= len(collector.segments) <= 6:
        raise ValueError("voiceover source must contain 3 to 6 narration blocks")
    return NarrationPlan(
        lesson_id=job_id,
        segments=tuple(
            NarrationSegment(
                id=f"segment-{index:02d}",
                text=text,
                cue=f"voiceover-block-{index:02d}",
            )
            for index, text in enumerate(collector.segments, start=1)
        ),
    )


class _VoiceoverCollector(ast.NodeVisitor):
    """Collect literal narration text in source order."""

    def __init__(self) -> None:
        self.segments: list[str] = []

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        call = _voiceover_call(node)
        if call is not None:
            text = next(
                (
                    keyword.value.value
                    for keyword in call.keywords
                    if keyword.arg == "text"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                None,
            )
            if text is None or not text.strip():
                raise ValueError("voiceover narration text must be a non-empty literal")
            self.segments.append(text)
        self.generic_visit(node)


def _silent_scene_source(source: str, narration: SynthesizedNarration) -> str:
    tree = ast.parse(source)
    transformer = _VoiceoverToScene(
        tuple(segment.duration_seconds for segment in narration.segments)
    )
    transformed = transformer.visit(tree)
    if transformer.transformed_blocks != len(narration.segments):
        raise ValueError("narration segments do not match voiceover blocks")
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed) + "\n"


class _VoiceoverToScene(ast.NodeTransformer):
    """Replace voiceover-only syntax with a credential-free Manim scene."""

    def __init__(self, durations: tuple[float, ...]) -> None:
        self._durations = durations
        self.transformed_blocks = 0

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:  # noqa: N802
        if node.module in {
            "manim_voiceover",
            "manim_voiceover.services.elevenlabs",
        }:
            return None
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:  # noqa: N802
        node.bases = [
            ast.Name(id="Scene", ctx=ast.Load())
            if isinstance(base, ast.Name) and base.id == "VoiceoverScene"
            else base
            for base in node.bases
        ]
        return cast(ast.ClassDef, self.generic_visit(node))

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:  # noqa: N802
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "set_speech_service"
        ):
            return None
        return cast(ast.Expr, self.generic_visit(node))

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:  # noqa: N802
        if _voiceover_call(node) is None:
            return cast(ast.With, self.generic_visit(node))
        if self.transformed_blocks >= len(self._durations):
            raise ValueError("voiceover block has no synthesized narration segment")
        tracker = node.items[0].optional_vars
        if not isinstance(tracker, ast.Name):
            raise ValueError("voiceover block must bind a tracker name")
        duration = self._durations[self.transformed_blocks]
        self.transformed_blocks += 1
        rewriter = _TrackerDurationRewriter(tracker.id, duration)
        transformed_body: list[ast.stmt] = []
        for statement in node.body:
            rewritten = rewriter.visit(statement)
            visited = self.visit(rewritten)
            if isinstance(visited, list):
                transformed_body.extend(visited)
            elif visited is not None:
                transformed_body.append(visited)
        return transformed_body


class _TrackerDurationRewriter(ast.NodeTransformer):
    """Replace one voiceover tracker's duration with measured audio length."""

    def __init__(self, tracker: str, duration: float) -> None:
        self._tracker = tracker
        self._duration = duration

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:  # noqa: N802
        if (
            node.attr == "duration"
            and isinstance(node.value, ast.Name)
            and node.value.id == self._tracker
        ):
            return ast.copy_location(ast.Constant(value=self._duration), node)
        return cast(ast.expr, self.generic_visit(node))


def _voiceover_call(node: ast.With) -> ast.Call | None:
    if len(node.items) != 1:
        return None
    context = node.items[0].context_expr
    if not (
        isinstance(context, ast.Call)
        and isinstance(context.func, ast.Attribute)
        and context.func.attr == "voiceover"
    ):
        return None
    return context


class NarratingRenderer:
    """Add optional narration to one renderer with a fixed lesson plan."""

    def __init__(
        self,
        *,
        renderer: JobRenderer,
        provider: NarrationProvider,
        assembler: MediaAssembler,
        plan_factory: Callable[[], NarrationPlan],
        artifact_root: Path,
    ) -> None:
        self._renderer = renderer
        self._provider = provider
        self._assembler = assembler
        self._plan_factory = plan_factory
        self._artifact_root = artifact_root

    def render(self, job_id: str) -> RenderOutcome | PartialOutcome:
        """Render a lesson and attach narration when synthesis succeeds."""
        outcome = self._renderer.render(job_id)
        if isinstance(outcome, PartialOutcome):
            return outcome
        return _attach_narration(
            job_id=job_id,
            outcome=outcome,
            plan=self._plan_factory(),
            provider=self._provider,
            assembler=self._assembler,
            artifact_root=self._artifact_root,
        )


def _attach_narration(
    *,
    job_id: str,
    outcome: RenderOutcome,
    plan: NarrationPlan,
    provider: NarrationProvider,
    assembler: MediaAssembler,
    artifact_root: Path,
) -> RenderOutcome:
    try:
        job_dir = artifact_root / job_id
        narration = provider.synthesize(plan, job_dir / "audio")
        bundle = assembler.assemble(
            silent_video=outcome.video_path,
            narration=narration,
            output_dir=job_dir / "narration",
        )
    except Exception as error:
        return replace(
            outcome,
            silent_video_path=outcome.video_path,
            narration_status=NarrationStatus.UNAVAILABLE,
            narration_diagnostics={
                "narration_error": type(error).__name__,
                "narration_error_cause": (
                    type(error.__cause__).__name__ if error.__cause__ is not None else None
                ),
            },
        )
    return replace(
        outcome,
        video_path=bundle.video_path,
        silent_video_path=bundle.silent_video_path,
        captions_path=bundle.captions_path,
        narration_status=bundle.narration_status,
        narration_diagnostics=bundle.diagnostics,
    )
