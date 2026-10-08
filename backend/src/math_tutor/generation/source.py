"""Extract generated Python and enforce the supported Manim scene contract.
Static admission rejects unsafe or structurally invalid source before rendering.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from math_tutor.generation.errors import ExtractionError, SceneValidationError

_PYTHON_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ALLOWED_IMPORTS = frozenset({"manim", "math", "numpy"})
_VOICEOVER_IMPORTS = {
    "manim_voiceover": frozenset({"VoiceoverScene"}),
    "manim_voiceover.services.elevenlabs": frozenset({"ElevenLabsService"}),
}
_FORBIDDEN_CALLS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "vars",
    }
)


@dataclass(frozen=True)
class ExtractedScene:
    """Admitted Python source and its single renderable scene class."""

    source: str
    scene_class: str


def extract_and_validate_scene(
    response: str,
    *,
    voiceover: bool = False,
) -> ExtractedScene:
    """Extract one generated scene and enforce the production source contract.

    Args:
        response: Model response containing exactly one Python code fence.
        voiceover: Whether to require the narrated scene contract.

    Returns:
        Admitted source and the generated scene class name.

    Raises:
        ExtractionError: If the response does not contain exactly one code fence.
        SceneValidationError: If the source is invalid, unsafe, or structurally unsupported.
    """
    matches = _PYTHON_FENCE.findall(response)
    if len(matches) != 1:
        raise ExtractionError(
            "model response must contain exactly one Python code fence",
            diagnostics={"failure_stage": "extraction"},
        )
    source = matches[0].strip()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise SceneValidationError(
            "generated scene is not valid Python",
            diagnostics={"failure_stage": "parse", "line": getattr(error, "lineno", None)},
        ) from error

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    generated = [node for node in classes if node.name == "GeneratedLesson"]
    if len(generated) != 1:
        raise SceneValidationError(
            "generated code must define exactly one class named GeneratedLesson",
            diagnostics={"failure_stage": "validation"},
        )
    expected_base = "VoiceoverScene" if voiceover else "Scene"
    if len(classes) != 1 or not (
        len(generated[0].bases) == 1
        and isinstance(generated[0].bases[0], ast.Name)
        and generated[0].bases[0].id == expected_base
    ):
        raise SceneValidationError(
            f"GeneratedLesson must be the only class and inherit directly from {expected_base}",
            diagnostics={"failure_stage": "validation"},
        )

    _validate_safe_scene_tree(tree, voiceover=voiceover)

    if voiceover:
        _validate_voiceover_contract(tree)

    return ExtractedScene(source=source, scene_class="GeneratedLesson")


def extract_and_validate_raw_scene(response: str) -> ExtractedScene:
    """Validate one raw training-style Manim scene before isolated rendering.

    Args:
        response: Plain Python source or one fenced Python block from an adapter.

    Returns:
        Original source and the sole direct Manim scene subclass name.

    Raises:
        ExtractionError: If fencing is incomplete or ambiguous.
        SceneValidationError: If syntax, scene structure, or safety checks fail.
    """
    matches = _PYTHON_FENCE.findall(response)
    if len(matches) > 1 or ("```" in response and len(matches) != 1):
        raise ExtractionError(
            "raw scene must be plain Python or one Python code fence",
            diagnostics={"failure_stage": "extraction"},
        )
    source = (matches[0] if matches else response).strip()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise SceneValidationError(
            "generated scene is not valid Python",
            diagnostics={"failure_stage": "parse", "line": getattr(error, "lineno", None)},
        ) from error
    scene_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and len(node.bases) == 1
        and isinstance(node.bases[0], ast.Name)
        and node.bases[0].id in {"Scene", "MovingCameraScene", "ThreeDScene"}
    ]
    if len(scene_classes) != 1:
        raise SceneValidationError(
            "raw code must define exactly one direct Manim scene subclass",
            diagnostics={"failure_stage": "validation"},
        )
    _validate_safe_scene_tree(tree, voiceover=False)
    return ExtractedScene(source=source, scene_class=scene_classes[0].name)


def _validate_safe_scene_tree(tree: ast.Module, *, voiceover: bool) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    raise _unsafe_import(root)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.partition(".")[0]
            permitted = node.level == 0 and root in _ALLOWED_IMPORTS
            if voiceover and module in _VOICEOVER_IMPORTS:
                permitted = node.level == 0 and all(
                    alias.name in _VOICEOVER_IMPORTS[module]
                    and alias.name != "*"
                    and alias.asname is None
                    for alias in node.names
                )
            if not permitted:
                raise _unsafe_import(module or root)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            identifier = node.id if isinstance(node, ast.Name) else node.attr
            if _is_dunder_identifier(identifier):
                raise SceneValidationError(
                    "dunder identifiers are not allowed in generated scenes",
                    diagnostics={"failure_stage": "validation"},
                )
            if isinstance(node, ast.Name) and identifier in _FORBIDDEN_CALLS:
                raise SceneValidationError(
                    f"reference '{identifier}' is not allowed in generated scenes",
                    diagnostics={"failure_stage": "validation"},
                )
        elif isinstance(node, ast.Call):
            forbidden_call = _forbidden_call_name(node.func)
            if forbidden_call is not None:
                raise SceneValidationError(
                    f"call '{forbidden_call}' is not allowed in generated scenes",
                    diagnostics={"failure_stage": "validation"},
                )

def _validate_voiceover_contract(tree: ast.Module) -> None:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    service_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "set_speech_service"
    ]
    if len(service_calls) != 1:
        raise SceneValidationError(
            "generated voiceover scene must configure speech service exactly once",
            diagnostics={"failure_stage": "validation"},
        )
    service_argument = service_calls[0].args[0] if service_calls[0].args else None
    disables_transcription = (
        isinstance(service_argument, ast.Call)
        and isinstance(service_argument.func, ast.Name)
        and service_argument.func.id == "ElevenLabsService"
        and any(
            keyword.arg == "transcription_model"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is None
            for keyword in service_argument.keywords
        )
    )
    if not disables_transcription:
        raise SceneValidationError(
            "ElevenLabsService must set transcription_model=None",
            diagnostics={"failure_stage": "validation"},
        )
    uses_supported_model = isinstance(service_argument, ast.Call) and any(
        keyword.arg == "model"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "eleven_multilingual_v2"
        for keyword in service_argument.keywords
    )
    if not uses_supported_model:
        raise SceneValidationError(
            "ElevenLabsService must use model eleven_multilingual_v2",
            diagnostics={"failure_stage": "validation"},
        )
    voiceover_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "voiceover"
    ]
    if not 3 <= len(voiceover_calls) <= 6:
        raise SceneValidationError(
            "generated voiceover scene must contain 3 to 6 voiceover blocks",
            diagnostics={"failure_stage": "validation"},
        )
    uses_tracker_duration = any(
        isinstance(node, ast.Attribute)
        and node.attr == "duration"
        and isinstance(node.value, ast.Name)
        and node.value.id == "tracker"
        for node in ast.walk(tree)
    )
    if not uses_tracker_duration:
        raise SceneValidationError(
            "generated voiceover scene must pace animation with tracker.duration",
            diagnostics={"failure_stage": "validation"},
        )


def _unsafe_import(module: str) -> SceneValidationError:
    return SceneValidationError(
        f"import '{module}' is not allowed in generated scenes",
        diagnostics={"failure_stage": "validation"},
    )


def _is_dunder_identifier(value: str) -> bool:
    return value.startswith("__") and value.endswith("__")


def _forbidden_call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id if node.id in _FORBIDDEN_CALLS else None
    if isinstance(node, ast.Attribute):
        return node.attr if node.attr in _FORBIDDEN_CALLS else None
    if isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value if key.value in _FORBIDDEN_CALLS else None
    return None
