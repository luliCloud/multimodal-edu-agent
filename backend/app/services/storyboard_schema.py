"""Shared planner errors and small validation helpers."""

NARRATION_WORDS_PER_SECOND = 2.5
MIN_SCENE_SECONDS = 3.0


class ScopeExceededError(ValueError):
    """The source document is outside the supported short-story scope."""


class StoryPlanningError(ValueError):
    """The planner could not produce a plan satisfying the contract."""


def narration_seconds(narration: str) -> float:
    """Estimate read-aloud time at roughly 150 words per minute."""
    return round(max(MIN_SCENE_SECONDS,
                     len(narration.split()) / NARRATION_WORDS_PER_SECOND), 1)


def strip_code_fence(text: str) -> str:
    """Drop a JSON Markdown fence occasionally added by instruction models."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return body.rsplit("```", 1)[0].strip()


def corrective_feedback(error: Exception) -> str:
    return (
        "Your previous answer was rejected:\n"
        f"{error}\n"
        "Return corrected JSON that fixes every problem listed above. "
        "Reply with the JSON object only."
    )
