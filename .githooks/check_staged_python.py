"""Validate staged Python files and run their repository Ruff checks.
Git paths stay NUL-delimited until they become explicit subprocess arguments."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

PROJECTS = ("backend", "training")


def staged_python_paths() -> tuple[str, ...]:
    """Return added, copied, modified, or renamed Python paths in the index."""
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            "--",
            "*.py",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(os.fsdecode(path) for path in completed.stdout.split(b"\0") if path)


def require_complete_files(paths: Iterable[str]) -> None:
    """Reject staged Python files whose working copies contain additional edits."""
    partial: list[str] = []
    for path in paths:
        completed = subprocess.run(["git", "diff", "--quiet", "--", path], check=False)
        if completed.returncode == 1:
            partial.append(path)
        elif completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)
    if partial:
        for path in partial:
            print(f"Partially staged Python file: {path}", file=os.sys.stderr)
        raise SystemExit(
            "Stage or restore each Python file completely before committing."
        )


def paths_by_project(paths: Iterable[str]) -> dict[str, list[str]]:
    """Group staged paths by the Python project whose Ruff configuration owns them."""
    grouped = {project: [] for project in PROJECTS}
    for path in paths:
        parsed = PurePosixPath(path)
        if parsed.parts and parsed.parts[0] in grouped:
            grouped[parsed.parts[0]].append(str(PurePosixPath(*parsed.parts[1:])))
    return grouped


def run_ruff(paths: Iterable[str], arguments: Sequence[str]) -> None:
    """Run one Ruff command per project with filenames passed as exact arguments."""
    for project, project_paths in paths_by_project(paths).items():
        if project_paths:
            subprocess.run(
                ["uv", "run", "ruff", *arguments, "--", *project_paths],
                cwd=Path(project),
                check=True,
            )


def main() -> None:
    """Validate index consistency and run the requested staged-file checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "lint", "format", "all"))
    mode = parser.parse_args().mode
    paths = staged_python_paths()
    require_complete_files(paths)
    if mode in {"lint", "all"}:
        run_ruff(paths, ("check",))
    if mode in {"format", "all"}:
        run_ruff(paths, ("format", "--check"))


if __name__ == "__main__":
    main()
