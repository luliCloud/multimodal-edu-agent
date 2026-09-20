import pytest

from backend.app.services.storyboard_schema import (
    MIN_SCENE_SECONDS,
    corrective_feedback,
    narration_seconds,
    strip_code_fence,
)


def test_narration_seconds_scales_with_words_above_a_floor() -> None:
    assert narration_seconds("word " * 30) == 12.0
    assert narration_seconds("six short words go right here") == MIN_SCENE_SECONDS


@pytest.mark.parametrize("fenced, expected", [
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('{"a": 1}', '{"a": 1}'),
    ("```", ""),
])
def test_strip_code_fence(fenced: str, expected: str) -> None:
    assert strip_code_fence(fenced) == expected


def test_corrective_feedback_includes_the_validation_problem() -> None:
    feedback = corrective_feedback(ValueError("scene 2 narration is too long"))
    assert "scene 2 narration is too long" in feedback
    assert "JSON object only" in feedback
