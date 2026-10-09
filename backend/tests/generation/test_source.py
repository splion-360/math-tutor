"""Verify extraction and deterministic admission of generated Manim source.
The tests cover supported scene contracts and rejected unsafe source patterns.
"""

from unittest.mock import patch

import pytest

from math_tutor.generation.errors import ExtractionError, SceneValidationError
from math_tutor.generation.source import (
    extract_and_validate_raw_scene,
    extract_and_validate_scene,
)

VALID_SCENE = """from manim import *

class GeneratedLesson(Scene):
    def construct(self):
        self.play(Write(MathTex(r"x^2")))
"""

VOICEOVER_SCENE = """from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService

class GeneratedLesson(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            ElevenLabsService(
                voice_id="voice-id",
                model="eleven_multilingual_v2",
                transcription_model=None,
            )
        )
        circle = Circle()
        with self.voiceover(text="Draw the circle.") as tracker:
            self.play(Create(circle), run_time=tracker.duration)
        with self.voiceover(text="Move it right.") as tracker:
            self.play(circle.animate.shift(RIGHT), run_time=tracker.duration)
        with self.voiceover(text="Now remove it.") as tracker:
            self.play(FadeOut(circle), run_time=tracker.duration)
"""


def test_extracts_one_python_fence_and_validates_generated_scene() -> None:
    extracted = extract_and_validate_scene(f"```python\n{VALID_SCENE}```")

    assert extracted.source == VALID_SCENE.rstrip()
    assert extracted.scene_class == "GeneratedLesson"


def test_raw_manim_scene_accepts_training_style_class_without_rewriting_it() -> None:
    source = VALID_SCENE.replace("GeneratedLesson", "TriangleProof")

    extracted = extract_and_validate_raw_scene(source)

    assert extracted.source == source.strip()
    assert extracted.scene_class == "TriangleProof"


def test_raw_manim_scene_ignores_known_qwen_tool_call_prefix() -> None:
    source = VALID_SCENE.replace("GeneratedLesson", "TriangleProof")
    response = f"<tool_call>\n\n<tool_call>\n\n{source}"

    extracted = extract_and_validate_raw_scene(response)

    assert extracted.source == source.strip()
    assert extracted.scene_class == "TriangleProof"


def test_raw_manim_scene_rejects_unsafe_import_before_rendering() -> None:
    source = VALID_SCENE.replace("from manim import *", "import os\nfrom manim import *")

    with pytest.raises(SceneValidationError, match="import 'os' is not allowed"):
        extract_and_validate_raw_scene(source)


def test_accepts_voiceover_scene_with_three_timed_narration_blocks() -> None:
    extracted = extract_and_validate_scene(
        f"```python\n{VOICEOVER_SCENE}```",
        voiceover=True,
    )

    assert extracted.source == VOICEOVER_SCENE.rstrip()


@pytest.mark.parametrize(
    ("source", "voiceover", "message"),
    [
        (VALID_SCENE, True, "inherit directly from VoiceoverScene"),
        (VOICEOVER_SCENE, False, "inherit directly from Scene"),
        (
            VOICEOVER_SCENE.replace(
                "from manim_voiceover import VoiceoverScene",
                "import os\nfrom manim_voiceover import VoiceoverScene",
            ),
            True,
            "import 'os' is not allowed",
        ),
        (
            VOICEOVER_SCENE.replace(
                "        with self.voiceover",
                "        open('/tmp/x')\n        with self.voiceover",
                1,
            ),
            True,
            "call 'open' is not allowed",
        ),
    ],
)
def test_voiceover_validation_rejects_wrong_mode_and_unsafe_code(
    source: str,
    voiceover: bool,
    message: str,
) -> None:
    with pytest.raises(SceneValidationError, match=message):
        extract_and_validate_scene(f"```python\n{source}```", voiceover=voiceover)


@pytest.mark.parametrize(
    "unsafe_import",
    [
        "from manim_voiceover import os",
        "from manim_voiceover.services.elevenlabs import os",
        "from manim_voiceover.services.elevenlabs import *",
        "from .manim_voiceover import VoiceoverScene",
    ],
)
def test_voiceover_validation_rejects_transitive_or_broad_imports(
    unsafe_import: str,
) -> None:
    source = VOICEOVER_SCENE.replace(
        "from manim_voiceover import VoiceoverScene",
        f"from manim_voiceover import VoiceoverScene\n{unsafe_import}",
    )

    with pytest.raises(SceneValidationError, match="not allowed"):
        extract_and_validate_scene(f"```python\n{source}```", voiceover=True)


def test_validation_rejects_aliasing_dynamic_execution() -> None:
    source = VALID_SCENE.replace(
        "    def construct(self):",
        "    def construct(self):\n        run = exec\n        run('print(1)')",
    )

    with pytest.raises(SceneValidationError, match="exec"):
        extract_and_validate_scene(f"```python\n{source}```")


def test_voiceover_validation_requires_three_to_six_blocks_and_tracker_duration() -> None:
    one_block = VOICEOVER_SCENE.split(
        '        with self.voiceover(text="Move it right.") as tracker:'
    )[0]
    one_block += "\n"

    with pytest.raises(SceneValidationError, match="3 to 6 voiceover blocks"):
        extract_and_validate_scene(f"```python\n{one_block}```", voiceover=True)

    without_duration = VOICEOVER_SCENE.replace("tracker.duration", "1")
    with pytest.raises(SceneValidationError, match="tracker.duration"):
        extract_and_validate_scene(f"```python\n{without_duration}```", voiceover=True)


def test_voiceover_validation_requires_literal_text_and_tracker_binding() -> None:
    dynamic_text = VOICEOVER_SCENE.replace(
        'with self.voiceover(text="Draw the circle.") as tracker:',
        "with self.voiceover(text=str(circle)) as tracker:",
    )
    without_tracker = VOICEOVER_SCENE.replace(
        'with self.voiceover(text="Draw the circle.") as tracker:',
        'with self.voiceover(text="Draw the circle."):',
    )

    with pytest.raises(SceneValidationError, match="non-empty literal"):
        extract_and_validate_scene(f"```python\n{dynamic_text}```", voiceover=True)
    with pytest.raises(SceneValidationError, match="bind a tracker"):
        extract_and_validate_scene(f"```python\n{without_tracker}```", voiceover=True)


def test_voiceover_validation_rejects_timing_outside_or_beyond_blocks() -> None:
    timed_prelude = VOICEOVER_SCENE.replace(
        "        circle = Circle()",
        "        circle = Circle()\n        self.play(Write(circle))",
    )
    extra_wait = VOICEOVER_SCENE.replace(
        "            self.play(Create(circle), run_time=tracker.duration)",
        (
            "            self.play(Create(circle), run_time=tracker.duration)\n"
            "            self.wait(1)"
        ),
    )

    with pytest.raises(SceneValidationError, match="inside voiceover blocks"):
        extract_and_validate_scene(f"```python\n{timed_prelude}```", voiceover=True)
    with pytest.raises(SceneValidationError, match="one play or wait"):
        extract_and_validate_scene(f"```python\n{extra_wait}```", voiceover=True)


def test_voiceover_validation_requires_direct_sequential_blocks_and_timing() -> None:
    looped_block = VOICEOVER_SCENE.replace(
        '        with self.voiceover(text="Draw the circle.") as tracker:\n'
        "            self.play(Create(circle), run_time=tracker.duration)",
        "        for _ in range(2):\n"
        '            with self.voiceover(text="Draw the circle.") as tracker:\n'
        "                self.play(Create(circle), run_time=tracker.duration)",
    )
    conditional_timing = VOICEOVER_SCENE.replace(
        "            self.play(Create(circle), run_time=tracker.duration)",
        (
            "            if circle:\n"
            "                self.play(Create(circle), run_time=tracker.duration)"
        ),
    )

    with pytest.raises(SceneValidationError, match="direct sequential statements"):
        extract_and_validate_scene(f"```python\n{looped_block}```", voiceover=True)
    with pytest.raises(SceneValidationError, match="one play or wait"):
        extract_and_validate_scene(f"```python\n{conditional_timing}```", voiceover=True)


def test_voiceover_validation_disables_optional_whisper_transcription() -> None:
    default_transcription = VOICEOVER_SCENE.replace(
        "                transcription_model=None,\n",
        "",
    )

    with pytest.raises(SceneValidationError, match="transcription_model=None"):
        extract_and_validate_scene(
            f"```python\n{default_transcription}```",
            voiceover=True,
        )


def test_voiceover_validation_rejects_deprecated_elevenlabs_model() -> None:
    deprecated_model = VOICEOVER_SCENE.replace(
        'model="eleven_multilingual_v2"',
        'model="eleven_monolingual_v1"',
    )

    with pytest.raises(SceneValidationError, match="eleven_multilingual_v2"):
        extract_and_validate_scene(
            f"```python\n{deprecated_model}```",
            voiceover=True,
        )


def test_rejects_ambiguous_multiple_code_fences() -> None:
    response = f"```python\n{VALID_SCENE}```\n```python\n{VALID_SCENE}```"

    with pytest.raises(ExtractionError, match="exactly one Python code fence"):
        extract_and_validate_scene(response)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import os\n" + VALID_SCENE, "import 'os' is not allowed"),
        (VALID_SCENE + "\nopen('/tmp/file', 'w')", "call 'open' is not allowed"),
        (
            VALID_SCENE + "\n__builtins__['open']('/tmp/file', 'w')",
            "not allowed",
        ),
        (
            VALID_SCENE + "\ngetattr(__builtins__, 'ev' + 'al')('1 + 1')",
            "not allowed",
        ),
        (VALID_SCENE + "\n# null byte:\x00", "not valid Python"),
        ("class GeneratedLesson(Scene)\n    pass", "not valid Python"),
        (
            "from manim import *\nclass WrongName(Scene):\n    pass",
            "class named GeneratedLesson",
        ),
    ],
)
def test_rejects_invalid_or_unsafe_generated_code(source: str, message: str) -> None:
    with pytest.raises(SceneValidationError, match=message):
        extract_and_validate_scene(f"```python\n{source}\n```")


def test_reports_parser_value_error_as_a_parse_failure() -> None:
    with (
        patch("math_tutor.generation.source.ast.parse", side_effect=ValueError("bad source")),
        pytest.raises(SceneValidationError, match="not valid Python") as caught,
    ):
        extract_and_validate_scene(f"```python\n{VALID_SCENE}```")

    assert caught.value.diagnostics == {"failure_stage": "parse", "line": None}
