"""Validated storyboard contract shared by the planner and its GPU worker.

The worker imports this module so it can check its own output while the model
is still resident, which makes a correction cost one more generation instead
of an 8 GB reload.
"""

import json

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

# Short-form narration is usually read at roughly 150 words per minute.
NARRATION_WORDS_PER_SECOND = 2.5
MIN_NARRATION_WORDS = 6
MAX_NARRATION_WORDS = 18
# Ask the model for a tighter band than the validator enforces, so ordinary
# variation does not spend a retry.
TARGET_NARRATION_WORDS = (8, 14)
MIN_SCENE_SECONDS = 3.0


class ScopeExceededError(ValueError):
    """The source document is longer than the planner supports."""


class StoryPlanningError(ValueError):
    """The planner could not produce a storyboard that satisfies the schema."""


def narration_seconds(narration: str) -> float:
    """Estimate how long this narration takes to read aloud."""
    return round(max(MIN_SCENE_SECONDS,
                     len(narration.split()) / NARRATION_WORDS_PER_SECOND), 1)


class SceneDraft(BaseModel):
    visual_prompt: str
    motion: str
    narration: str

    @field_validator("visual_prompt", "motion", "narration")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("narration")
    @classmethod
    def check_narration_budget(cls, value: str) -> str:
        words = len(value.split())
        if not MIN_NARRATION_WORDS <= words <= MAX_NARRATION_WORDS:
            raise ValueError(f"narration is {words} words, expected "
                             f"{MIN_NARRATION_WORDS}-{MAX_NARRATION_WORDS}")
        return value


class StoryboardDraft(BaseModel):
    summary: str
    scenes: list[SceneDraft]

    @field_validator("summary")
    @classmethod
    def require_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def check_scene_count(self, info: ValidationInfo) -> "StoryboardDraft":
        expected = (info.context or {}).get("scene_count")
        if expected is not None and len(self.scenes) != expected:
            raise ValueError(f"must return exactly {expected} scenes, "
                             f"got {len(self.scenes)}")
        return self


def strip_code_fence(text: str) -> str:
    """Drop a ```json fence that instruction-tuned models add despite the prompt."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return body.rsplit("```", 1)[0].strip()


def parse_storyboard(text: str, scene_count: int) -> StoryboardDraft:
    """Parse and validate one model response; raises ValueError if unusable."""
    try:
        payload = json.loads(strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"response was not valid JSON: {exc}") from exc
    return StoryboardDraft.model_validate(payload, context={"scene_count": scene_count})


def corrective_feedback(error: Exception) -> str:
    return ("Your previous answer was rejected:\n"
            f"{error}\n"
            "Return corrected JSON that fixes every problem listed above. "
            "Reply with the JSON object only.")
