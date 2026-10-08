"""Define generation, admission, and output-validation failures.
The shared hierarchy carries bounded diagnostics across generation modules.
"""

from math_tutor.jobs import JobExecutionError


class GeneratedLessonError(JobExecutionError):
    """Base failure for generated lesson processing."""


class ExtractionError(GeneratedLessonError):
    """Raised when a model response does not contain extractable source."""


class SceneValidationError(GeneratedLessonError):
    """Raised when generated source violates the static admission contract."""


class OutputValidationError(GeneratedLessonError):
    """Raised when a rendered attempt cannot be accepted for publication."""
