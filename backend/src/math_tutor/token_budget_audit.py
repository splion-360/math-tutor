from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

Phase = Literal["prompt", "draft", "normalization", "output"]
_TOKENISH = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class PhaseCount:
    phase: Phase
    tokens: int
    source: str


@dataclass(frozen=True)
class TokenBudgetRecord:
    job_id: str
    artifact_directory: str
    route: str
    prompt: PhaseCount
    draft: PhaseCount
    normalization: PhaseCount
    output: PhaseCount
    total_provider_tokens: int
    finish_reason: str | None
    finish_reasons: tuple[str, ...]
    failure_stage: str | None
    failure_artifacts: tuple[str, ...]


def audit_artifact_root(artifact_root: Path) -> dict[str, object]:
    records = list(iter_token_budget_records(artifact_root))
    return build_token_budget_report(records, artifact_root=artifact_root)


def iter_token_budget_records(artifact_root: Path) -> Iterable[TokenBudgetRecord]:
    attempts_by_job = _load_attempt_records(artifact_root)
    artifact_root = _artifact_attempt_root(artifact_root)
    job_dirs = sorted(
        path
        for path in artifact_root.iterdir()
        if path.is_dir() and (path / "prompt.txt").is_file()
    )
    base_job_ids = [
        path.name
        for path in job_dirs
        if not (
            path.name.endswith("-normalized")
            or path.name.endswith("-base")
            or path.name.endswith("-silent")
        )
    ]
    if base_job_ids:
        for job_id in base_job_ids:
            if _has_related_job_artifacts(artifact_root, job_id):
                yield _record_for_job(artifact_root, job_id, attempts_by_job.get(job_id))
            else:
                yield _record_for_direct_job(artifact_root / job_id, attempts_by_job.get(job_id))
        return

    for path in job_dirs:
        yield _record_for_direct_job(path, attempts_by_job.get(path.name))


def build_token_budget_report(
    records: Sequence[TokenBudgetRecord],
    *,
    artifact_root: Path,
) -> dict[str, object]:
    phase_summaries = {
        phase: _summarize_counts([getattr(record, phase).tokens for record in records])
        for phase in ("prompt", "draft", "normalization", "output")
    }
    failures = [
        {
            "job_id": record.job_id,
            "failure_stage": record.failure_stage,
            "artifact_directory": record.artifact_directory,
            "failure_artifacts": list(record.failure_artifacts),
            "tokens": {
                "prompt": asdict(record.prompt),
                "draft": asdict(record.draft),
                "normalization": asdict(record.normalization),
                "output": asdict(record.output),
                "total_provider_tokens": record.total_provider_tokens,
            },
        }
        for record in records
        if record.failure_stage is not None
    ]
    return {
        "artifact_root": str(artifact_root),
        "job_count": len(records),
        "phase_summaries": phase_summaries,
        "failure_examples": failures,
        "recommended_limits": _recommended_limits(records),
        "records": [_record_json(record) for record in records],
        "notes": [
            "Provider usage from generation metadata is authoritative when present.",
            "Estimated counts use a lightweight tokenizer only for missing provider usage.",
            "Recommended limits document observed pressure points and do not change runtime "
            "behavior.",
        ],
    }


def write_markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Token Budget Audit",
        "",
        f"Artifact root: `{report['artifact_root']}`",
        f"Jobs audited: {report['job_count']}",
        "",
        "## Phase Counts",
        "",
        "| Phase | Count | Min | Mean | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    summaries = report["phase_summaries"]
    assert isinstance(summaries, dict)
    for phase in ("prompt", "draft", "normalization", "output"):
        summary = summaries[phase]
        assert isinstance(summary, dict)
        lines.append(
            "| "
            + " | ".join(
                [
                    phase,
                    str(summary["count"]),
                    str(summary["min"]),
                    f"{float(summary['mean']):.1f}" if summary["mean"] is not None else "n/a",
                    str(summary["p95"]),
                    str(summary["max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failure Examples", ""])
    failures = report["failure_examples"]
    assert isinstance(failures, list)
    if failures:
        for failure in failures:
            assert isinstance(failure, dict)
            tokens = failure["tokens"]
            assert isinstance(tokens, dict)
            lines.append(
                "- "
                f"`{failure['job_id']}` failed at `{failure['failure_stage']}` "
                f"with prompt={_phase_tokens(tokens, 'prompt')}, "
                f"draft={_phase_tokens(tokens, 'draft')}, "
                f"normalization={_phase_tokens(tokens, 'normalization')}, "
                f"output={_phase_tokens(tokens, 'output')}. "
                f"Artifacts: `{failure['artifact_directory']}`"
            )
    else:
        lines.append("No failed jobs were found in the audited artifacts.")
    lines.extend(["", "## Recommended Limits", ""])
    recommendations = report["recommended_limits"]
    assert isinstance(recommendations, list)
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")
    lines.extend(["", "## Notes", ""])
    notes = report["notes"]
    assert isinstance(notes, list)
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _record_for_job(
    artifact_root: Path,
    job_id: str,
    attempt: dict[str, object] | None,
) -> TokenBudgetRecord:
    job_dir = artifact_root / job_id
    specialist = _read_json(job_dir / "specialist_generation.json")
    specialist_status = str(specialist.get("status", ""))
    if specialist_status == "generated":
        normalizer_job_id = f"{job_id}-normalized"
    else:
        normalizer_job_id = f"{job_id}-base"
    normalizer_dir = artifact_root / normalizer_job_id
    generation = _read_json(normalizer_dir / "generation.json")
    if not generation and normalizer_job_id.endswith("-normalized"):
        normalizer_job_id = f"{job_id}-base"
        normalizer_dir = artifact_root / normalizer_job_id
        generation = _read_json(normalizer_dir / "generation.json")
    base_dir = artifact_root / f"{job_id}-base"
    production_dirs = _unique_paths(
        [
            normalizer_dir,
            artifact_root / f"{normalizer_job_id}-silent",
            base_dir,
            artifact_root / f"{job_id}-base-silent",
        ]
    )
    production_generations = _generation_metadata(production_dirs)
    final_dir, final_generation = _final_generation(production_generations, normalizer_dir)

    failure_stage = _failure_stage(
        attempt,
        [specialist, *[metadata for _, metadata in production_generations]],
        [job_dir, *production_dirs],
    )
    return TokenBudgetRecord(
        job_id=job_id,
        artifact_directory=str(job_dir),
        route="specialist_guided" if specialist else "base_model",
        prompt=_provider_or_estimated(
            "prompt",
            specialist if specialist else generation,
            "prompt_tokens",
            job_dir / "prompt.txt",
        ),
        draft=_provider_or_estimated(
            "draft",
            specialist,
            "completion_tokens",
            job_dir / "specialist_response.txt",
        ),
        normalization=_provider_or_estimated(
            "normalization",
            generation,
            "prompt_tokens",
            normalizer_dir / "prompt.txt",
        ),
        output=_provider_or_estimated(
            "output",
            final_generation,
            "completion_tokens",
            final_dir / "raw_response.txt",
        ),
        total_provider_tokens=_int_field(specialist, "total_tokens")
        + _sum_provider_tokens(metadata for _, metadata in production_generations),
        finish_reason=_str_field(final_generation, "finish_reason")
        or _str_field(specialist, "finish_reason"),
        finish_reasons=_finish_reasons([specialist, *[m for _, m in production_generations]]),
        failure_stage=failure_stage,
        failure_artifacts=_failure_artifacts(job_dir, *production_dirs),
    )


def _has_related_job_artifacts(artifact_root: Path, job_id: str) -> bool:
    return (
        (artifact_root / job_id / "specialist_generation.json").is_file()
        or (artifact_root / f"{job_id}-normalized").is_dir()
        or (artifact_root / f"{job_id}-base").is_dir()
    )


def _record_for_direct_job(
    job_dir: Path,
    attempt: dict[str, object] | None,
) -> TokenBudgetRecord:
    generation = _read_json(job_dir / "generation.json")
    silent_dir = job_dir.parent / f"{job_dir.name}-silent"
    production_generations = _generation_metadata([job_dir, silent_dir])
    final_dir, final_generation = _final_generation(production_generations, job_dir)
    return TokenBudgetRecord(
        job_id=job_dir.name,
        artifact_directory=str(job_dir),
        route="base_model",
        prompt=_provider_or_estimated(
            "prompt",
            generation,
            "prompt_tokens",
            job_dir / "prompt.txt",
        ),
        draft=PhaseCount("draft", 0, "not_applicable"),
        normalization=PhaseCount("normalization", 0, "not_applicable"),
        output=_provider_or_estimated(
            "output",
            final_generation,
            "completion_tokens",
            final_dir / "raw_response.txt",
        ),
        total_provider_tokens=_sum_provider_tokens(
            metadata for _, metadata in production_generations
        ),
        finish_reason=_str_field(final_generation, "finish_reason"),
        finish_reasons=_finish_reasons([metadata for _, metadata in production_generations]),
        failure_stage=_failure_stage(
            attempt,
            [metadata for _, metadata in production_generations],
            [job_dir, silent_dir],
        ),
        failure_artifacts=_failure_artifacts(job_dir, silent_dir),
    )


def _load_attempt_records(artifact_root: Path) -> dict[str, dict[str, object]]:
    candidates = [
        artifact_root / "attempts.jsonl",
        artifact_root.parent / "attempts.jsonl",
    ]
    attempts: dict[str, dict[str, object]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict) and isinstance(item.get("example_id"), str):
                attempts[str(item["example_id"])] = item
    return attempts


def _artifact_attempt_root(artifact_root: Path) -> Path:
    attempts_root = artifact_root / "attempts"
    if attempts_root.is_dir():
        return attempts_root
    return artifact_root


def _provider_or_estimated(
    phase: Phase,
    metadata: dict[str, object],
    key: str,
    text_path: Path,
) -> PhaseCount:
    tokens = _int_field(metadata, key)
    if tokens > 0:
        return PhaseCount(phase, tokens, f"provider:{text_path.name}")
    if text_path.is_file():
        return PhaseCount(
            phase,
            _estimate_tokens(text_path.read_text(encoding="utf-8")),
            f"estimated:{text_path.name}",
        )
    return PhaseCount(phase, 0, "missing")


def _estimate_tokens(text: str) -> int:
    return len(_TOKENISH.findall(text))


def _generation_metadata(dirs: Sequence[Path]) -> list[tuple[Path, dict[str, object]]]:
    return [
        (directory, metadata)
        for directory in dirs
        if (metadata := _read_json(directory / "generation.json"))
    ]


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _final_generation(
    generations: Sequence[tuple[Path, dict[str, object]]],
    fallback_dir: Path,
) -> tuple[Path, dict[str, object]]:
    if generations:
        return generations[-1]
    return fallback_dir, {}


def _sum_provider_tokens(metadatas: Iterable[dict[str, object]]) -> int:
    return sum(_int_field(metadata, "total_tokens") for metadata in metadatas)


def _finish_reasons(metadatas: Sequence[dict[str, object]]) -> tuple[str, ...]:
    return tuple(
        reason
        for metadata in metadatas
        if (reason := _str_field(metadata, "finish_reason")) is not None
    )


def _failure_stage(
    attempt: dict[str, object] | None,
    metadatas: Sequence[dict[str, object]],
    artifact_dirs: Sequence[Path],
) -> str | None:
    if attempt is not None and isinstance(attempt.get("failure_stage"), str):
        return str(attempt["failure_stage"])
    for metadata in metadatas:
        status = metadata.get("status")
        if isinstance(status, str) and status not in {"", "generated"}:
            return str(metadata.get("failure_stage") or status.removesuffix("_failed"))
    for directory in artifact_dirs:
        render = _read_json(directory / "render.json")
        status = render.get("status")
        if isinstance(status, str) and status not in {"", "ready"}:
            return "render_timeout" if status == "timed_out" else "render"
    return None


def _failure_artifacts(*dirs: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for directory in dirs:
        for name in (
            "generation.json",
            "specialist_generation.json",
            "render.json",
            "raw_response.txt",
            "specialist_response.txt",
        ):
            path = directory / name
            if path.is_file():
                paths.append(str(path))
    return tuple(paths)


def _summarize_counts(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": fmean(ordered),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def _recommended_limits(records: Sequence[TokenBudgetRecord]) -> list[str]:
    if not records:
        return [
            "Run the audit against at least one saved artifact directory before setting limits."
        ]
    recommendations: list[str] = []
    for phase in ("prompt", "draft", "normalization", "output"):
        values = [getattr(record, phase).tokens for record in records]
        observed_max = max(values)
        p95 = sorted(values)[min(len(values) - 1, int(len(values) * 0.95))]
        recommendations.append(
            f"Keep {phase} comfortably above observed p95={p95} and below a documented "
            f"hard cap; observed max={observed_max}. Do not enforce until a live run confirms "
            "the cap leaves successful examples untouched."
        )
    pressure = [
        record.job_id
        for record in records
        if "length" in record.finish_reasons or record.output.tokens >= 3800
    ]
    if pressure:
        recommendations.append(
            "Investigate context/output pressure before tuning: "
            + ", ".join(sorted(pressure))
        )
    return recommendations


def _record_json(record: TokenBudgetRecord) -> dict[str, object]:
    value = asdict(record)
    value["failure_artifacts"] = list(record.failure_artifacts)
    return value


def _phase_tokens(tokens: dict[str, object], phase: str) -> int:
    value = tokens[phase]
    assert isinstance(value, dict)
    return int(value["tokens"])


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _int_field(value: dict[str, object], key: str) -> int:
    field = value.get(key, 0)
    return int(field) if isinstance(field, int | float) else 0


def _str_field(value: dict[str, object], key: str) -> str | None:
    field = value.get(key)
    return field if isinstance(field, str) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit saved Math Tutor generation artifacts for token pressure."
    )
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_artifact_root(args.artifact_root)
    content = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else write_markdown_report(report)
    )
    if args.output is not None:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
