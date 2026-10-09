"""Verify shared-adapter Modal packaging without deploying GPU infrastructure.
The test replaces Modal objects while preserving checkpoint and runtime contracts."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def test_modal_module_packages_one_shared_adapter_and_scales_to_zero(
    monkeypatch,
    tmp_path: Path,
    repo_root: Path,
) -> None:
    installed_packages: list[str] = []
    local_files: list[tuple[Path, str]] = []
    function_options: dict[str, object] = {}
    web_server_options: dict[str, object] = {}
    commands: list[list[str]] = []

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

        def add_local_file(self, path: Path, *, remote_path: str) -> FakeImage:
            local_files.append((path, remote_path))
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
        web_server=lambda **kwargs: (
            web_server_options.update(kwargs) or (lambda function: function)
        ),
    )
    original_is_dir = Path.is_dir
    adapter_source = tmp_path / "adapters"
    monkeypatch.setenv("MATH_TUTOR_ADAPTER_SOURCE", str(adapter_source))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda command: commands.append(command),
    )
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: (
            False
            if str(path).endswith("training/artifacts/token_factory")
            else original_is_dir(path)
        ),
    )

    module = runpy.run_path(repo_root / "deployments" / "modal" / "inference.py")
    module["serve"]()

    assert installed_packages == ["vllm==0.21.0"]
    assert local_files == [
        (
            adapter_source / "adapter_model.safetensors",
            "/adapter/adapter_model.safetensors",
        ),
        (adapter_source / "adapter_config.json", "/adapter/adapter_config.json"),
    ]
    assert function_options["min_containers"] == 0
    assert function_options["max_containers"] == 1
    assert web_server_options["requires_proxy_auth"] is True
    command = commands[0]
    assert command[command.index("--revision") + 1] == ("1b4199c4f36b0cef378bfb12390c18780c18af4c")
    assert command[command.index("--lora-modules") + 1] == (
        "shared-lora-qwen3-4b-manim-v1=/adapter"
    )


def test_modal_module_imports_from_flattened_remote_path(
    monkeypatch,
    tmp_path: Path,
    repo_root: Path,
) -> None:
    local_files: list[tuple[Path, str]] = []

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

        def add_local_file(self, path: Path, *, remote_path: str) -> FakeImage:
            local_files.append((path, remote_path))
            return self

    class FakeApp:
        def __init__(self, *_args: object) -> None:
            pass

        def function(self, **_kwargs: object):
            return lambda function: function

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImage,
        Volume=SimpleNamespace(from_name=lambda *_args, **_kwargs: object()),
        concurrent=lambda **_kwargs: lambda function: function,
        web_server=lambda **_kwargs: lambda function: function,
    )
    deployed_module = tmp_path / "inference.py"
    deployed_module.write_text(
        (repo_root / "deployments" / "modal" / "inference.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.delenv("MATH_TUTOR_ADAPTER_SOURCE", raising=False)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    runpy.run_path(deployed_module)

    assert local_files == [
        (Path("/adapter/adapter_model.safetensors"), "/adapter/adapter_model.safetensors"),
        (Path("/adapter/adapter_config.json"), "/adapter/adapter_config.json"),
    ]
