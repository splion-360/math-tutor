"""Configure shared pytest options and stable repository path fixtures.
The fixtures keep tests independent of their directory depth."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root shared by deployment tests."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def backend_root(repo_root: Path) -> Path:
    """Return the backend project root shared by rendering tests."""
    return repo_root / "backend"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-manim-docker",
        action="store_true",
        default=False,
        help="run integration tests that start the pinned Manim Docker image",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-manim-docker"):
        return
    skip_docker = pytest.mark.skip(reason="use --run-manim-docker to run container renders")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_docker)
