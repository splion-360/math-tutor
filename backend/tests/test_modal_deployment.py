from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def test_modal_module_does_not_check_the_local_adapter_source_during_remote_import(
    monkeypatch,
) -> None:
    installed_packages: list[str] = []
    function_options: dict[str, object] = {}
    web_server_options: dict[str, object] = {}

    class FakeImage:
        @classmethod
        def from_registry(cls, *_args: object, **_kwargs: object) -> FakeImage:
            return cls()

        def entrypoint(self, *_args: object) -> FakeImage:
            return self

        def uv_pip_install(self, *_args: object) -> FakeImage:
            installed_packages.extend(str(arg) for arg in _args)
            return self

        def env(self, *_args: object) -> FakeImage:
            return self

        def add_local_dir(self, *_args: object, **_kwargs: object) -> FakeImage:
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
        web_server=lambda **kwargs: web_server_options.update(kwargs)
        or (lambda function: function),
    )
    original_is_dir = Path.is_dir
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: False
        if str(path).endswith("training/artifacts/token_factory")
        else original_is_dir(path),
    )

    runpy.run_path(Path(__file__).parents[2] / "deployments" / "modal" / "inference.py")

    assert installed_packages == ["vllm==0.21.0"]
    assert function_options["min_containers"] == 1
    assert function_options["max_containers"] == 1
    assert web_server_options["requires_proxy_auth"] is True
