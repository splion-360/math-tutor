"""Verify the pinned Modal visual-model deployment without deploying it.
The test records cold-start policy and the exact Hugging Face model revision."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from math_tutor.validation.visual_provider import QWEN3_VL_MODEL, QWEN3_VL_REVISION


def test_modal_visual_model_is_scale_to_zero_and_revision_pinned(monkeypatch) -> None:
    function_options: dict[str, object] = {}
    commands: list[list[str]] = []

    class FakeImage:
        @classmethod
        def from_registry(cls, *_args: object, **_kwargs: object) -> FakeImage:
            return cls()

        def entrypoint(self, *_args: object) -> FakeImage:
            return self

        def uv_pip_install(self, *_args: object) -> FakeImage:
            return self

        def env(self, *_args: object) -> FakeImage:
            return self

    class FakeApp:
        def __init__(self, *_args: object) -> None:
            pass

        def function(self, **kwargs: object):
            function_options.update(kwargs)
            return lambda function: function

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImage,
        Volume=SimpleNamespace(from_name=lambda *_args, **_kwargs: object()),
        concurrent=lambda **_kwargs: lambda function: function,
        web_server=lambda **_kwargs: lambda function: function,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command: commands.append(command),
    )

    namespace = runpy.run_path(
        Path(__file__).parents[2] / "deployments" / "modal" / "visual_validation.py"
    )
    namespace["serve"]()

    assert function_options["min_containers"] == 0
    assert function_options["max_containers"] == 1
    command = commands[0]
    assert command[command.index("serve") + 1] == QWEN3_VL_MODEL
    assert command[command.index("--revision") + 1] == QWEN3_VL_REVISION
