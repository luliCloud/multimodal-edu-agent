import pytest
from pydantic import ValidationError

from backend.app.services.storyboard_schema import (
    MIN_SCENE_SECONDS,
    SceneDraft,
    StoryboardDraft,
    narration_seconds,
    parse_storyboard,
    strip_code_fence,
)

NARRATION = "The little seed waits quietly beneath the dark garden soil."


def _scene(**overrides) -> dict:
    return {"visual_prompt": "A seed in dark soil.", "motion": "The camera pushes in.",
            "narration": NARRATION, **overrides}


def test_scene_strips_surrounding_whitespace() -> None:
    scene = SceneDraft(**_scene(visual_prompt="  A seed in dark soil.  "))
    assert scene.visual_prompt == "A seed in dark soil."


@pytest.mark.parametrize("field", ["visual_prompt", "motion", "narration"])
def test_scene_rejects_blank_field(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        SceneDraft(**_scene(**{field: "   "}))


@pytest.mark.parametrize("narration", ["Too short here.", "word " * 25])
def test_scene_rejects_narration_outside_budget(narration: str) -> None:
    with pytest.raises(ValidationError, match="expected 6-18"):
        SceneDraft(**_scene(narration=narration))


def test_storyboard_requires_the_planned_scene_count() -> None:
    payload = {"summary": "A seed becomes a flower.", "scenes": [_scene(), _scene()]}
    with pytest.raises(ValidationError, match="exactly 3 scenes"):
        StoryboardDraft.model_validate(payload, context={"scene_count": 3})


def test_storyboard_accepts_the_planned_scene_count() -> None:
    payload = {"summary": "A seed becomes a flower.", "scenes": [_scene(), _scene()]}
    assert len(StoryboardDraft.model_validate(payload, context={"scene_count": 2}).scenes) == 2


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


def test_parse_storyboard_reports_unparsable_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_storyboard("Sure! Here is your storyboard.", 1)
