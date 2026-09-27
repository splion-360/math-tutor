from __future__ import annotations

import tomllib
from pathlib import Path

from dynamic_lora.constants import TRAIN_DEPENDENCY_CONSTRAINTS


def test_recorded_training_constraints_match_pyproject() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["optional-dependencies"]["train"]
    expected = {
        package: requirement.removeprefix(package)
        for requirement in dependencies
        for package in [requirement.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0]]
    }

    assert expected == TRAIN_DEPENDENCY_CONSTRAINTS
