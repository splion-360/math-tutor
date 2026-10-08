"""Check paired evaluation deployment in local and flattened Modal layouts.
Remote imports must not inspect paths that exist only in the owner's checkout."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("local", [True, False])
def test_deployment_mounts_local_sources_only_on_the_client(local: bool, monkeypatch: Any) -> None:
    """The same deployment imports with both repository and remote module paths."""
    mounts = []
    options = []

    class Image:
        def entrypoint(self, *args: Any) -> Image:
            return self

        def uv_pip_install(self, *args: Any) -> Image:
            return self

        def env(self, values: Any) -> Image:
            return self

        def add_local_dir(self, source: Path, **kwargs: Any) -> Image:
            mounts.append(source)
            return self

        def add_local_file(self, source: Path, **kwargs: Any) -> Image:
            mounts.append(source)
            return self

    class App:
        def __init__(self, *args: Any) -> None:
            pass

        def function(self, **kwargs: Any) -> Any:
            options.append(kwargs)
            return lambda function: function

        def local_entrypoint(self) -> Any:
            return lambda function: function

    fake = SimpleNamespace(
        is_local=lambda: local,
        App=App,
        Image=SimpleNamespace(from_registry=lambda *a, **k: Image()),
        Volume=SimpleNamespace(from_name=lambda *a, **k: object()),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    path = Path(__file__).parents[2] / "deployments/modal/paired_evaluation.py"
    namespace = {"__file__": str(path) if local else "/root/paired_evaluation.py"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert len(mounts) == (2 if local else 0)
    assert options[0]["gpu"] == "L4"
    assert options[0]["max_containers"] == 1
    assert options[0]["scaledown_window"] == 2
    assert options[0]["timeout"] == 3600
