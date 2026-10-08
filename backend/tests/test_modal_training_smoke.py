from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def test_modal_training_smoke_packages_training_code_and_artifact_volume(
    monkeypatch: Any,
) -> None:
    installed_packages: list[str] = []
    env_values: dict[str, str] = {}
    local_dirs: list[tuple[str, str]] = []
    local_files: list[tuple[str, str]] = []
    function_options: list[dict[str, object]] = []
    function_names: list[str] = []
    remote_calls: list[tuple[str, tuple[object, ...]]] = []
    secret_names: list[str] = []

    class FakeImage:
        @classmethod
        def from_registry(cls, *_args: object, **_kwargs: object) -> FakeImage:
            return cls()

        def entrypoint(self, *_args: object) -> FakeImage:
            return self

        def uv_pip_install(self, *_args: object) -> FakeImage:
            installed_packages.extend(str(arg) for arg in _args)
            return self

        def env(self, values: dict[str, str]) -> FakeImage:
            env_values.update(values)
            return self

        def add_local_dir(self, source: Path, *, remote_path: str) -> FakeImage:
            local_dirs.append((source.as_posix(), remote_path))
            return self

        def add_local_file(self, source: Path, *, remote_path: str) -> FakeImage:
            local_files.append((source.as_posix(), remote_path))
            return self

    class FakeApp:
        def __init__(self, *_args: object) -> None:
            pass

        def function(self, **kwargs: object) -> Any:
            function_options.append(kwargs)

            def decorate(function: Any) -> Any:
                function_names.append(function.__name__)
                function.remote = lambda *args: remote_calls.append((function.__name__, args)) or {}
                return function

            return decorate

        def local_entrypoint(self) -> Any:
            return lambda function: function

    def secret_from_name(name: str) -> str:
        secret_names.append(name)
        return f"secret:{name}"

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImage,
        Secret=SimpleNamespace(from_name=secret_from_name),
        Volume=SimpleNamespace(from_name=lambda name, **_kwargs: f"volume:{name}"),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    module = runpy.run_path(
        str(Path(__file__).parents[2] / "deployments" / "modal" / "training_smoke.py")
    )

    assert "transformers>=4.51,<5" in installed_packages
    assert "wandb>=0.18,<1" in installed_packages
    assert env_values["PYTHONPATH"] == "/workspace/training/src"
    assert env_values["SOURCE_VERSION"]
    mounted_dirs = [(Path(source).name, remote) for source, remote in local_dirs]
    assert ("src", "/workspace/training/src") in mounted_dirs
    assert ("fixtures", "/workspace/training/fixtures") in mounted_dirs
    assert ("configs", "/workspace/training/configs") in mounted_dirs
    if (Path(__file__).parents[2] / "training/data/bespoke_manim_train.jsonl").exists():
        assert local_files[0][1] == "/workspace/training/data/bespoke_manim_train.jsonl"
    assert (
        "evaluation",
        "/workspace/backend/data/evaluation",
    ) in mounted_dirs
    assert {options["gpu"] for options in function_options} == {"L4", "A100-40GB", "A100-80GB"}
    assert dict(zip(function_names, function_options, strict=True))["run_placement_arm"]["gpu"] == (
        "A100-80GB"
    )
    expected_volumes = {
        "/root/.cache/huggingface": "volume:math-tutor-huggingface-cache",
        "/artifacts": "volume:math-tutor-training-artifacts",
    }
    for options in function_options:
        assert options["volumes"] == expected_volumes
        assert options["secrets"] == ["secret:WANDB_API_KEY"]

    module["main"](full_probe=True)
    module["main"]()
    assert remote_calls == [
        ("run_full_probe", ("modal_probe_qwen3_4b.json",)),
        ("run_smoke", ("modal_smoke_qwen3_4b.json",)),
    ]
