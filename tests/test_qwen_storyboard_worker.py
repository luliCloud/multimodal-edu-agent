"""Cover the planner's retry contract without a GPU: `generate` is a stub."""

import json

import pytest

from backend.app.services.storyboard_schema import StoryPlanningError
from backend.scripts.qwen_storyboard_worker import build_messages, plan_with_retries

SENTENCES = [{"id": 1, "page": 1, "text": "A seed slept in dark soil."},
             {"id": 2, "page": 1, "text": "Rain fell on the seed."}]
GROUPS = [[1], [2]]
NARRATION = "The little seed waits quietly beneath the dark garden soil."


def _answer(scene_count: int, **overrides) -> str:
    scene = {"visual_prompt": "A seed in dark soil.", "motion": "The camera pushes in.",
             "narration": NARRATION, **overrides}
    return json.dumps({"summary": "A seed becomes a flower.",
                       "scenes": [scene] * scene_count})


def test_accepts_a_valid_first_answer() -> None:
    calls = []

    def generate(messages):
        calls.append(len(messages))
        return _answer(2)

    storyboard = plan_with_retries(generate, build_messages(SENTENCES, GROUPS), 2, 3)
    assert len(storyboard.scenes) == 2
    assert calls == [2]


def test_retries_with_the_validation_error_in_the_prompt() -> None:
    seen = []

    def generate(messages):
        seen.append(messages[-1]["content"])
        return _answer(2) if len(seen) == 2 else _answer(2, narration="Too short.")

    storyboard = plan_with_retries(generate, build_messages(SENTENCES, GROUPS), 2, 3)
    assert storyboard.scenes[0].narration == NARRATION
    assert "expected 6-18" in seen[1], "the retry prompt must carry the rejection reason"


def test_retries_when_the_model_does_not_return_json() -> None:
    answers = iter(["I cannot do that.", _answer(2)])
    storyboard = plan_with_retries(lambda messages: next(answers),
                                   build_messages(SENTENCES, GROUPS), 2, 3)
    assert storyboard.summary == "A seed becomes a flower."


def test_gives_up_after_the_attempt_budget() -> None:
    calls = []

    def generate(messages):
        calls.append(1)
        return _answer(1)

    with pytest.raises(StoryPlanningError, match="invalid after 3 attempts"):
        plan_with_retries(generate, build_messages(SENTENCES, GROUPS), 2, 3)
    assert len(calls) == 3


def test_prompt_states_the_scene_count_and_narration_contract() -> None:
    prompt = build_messages(SENTENCES, GROUPS)[1]["content"]
    assert "exactly 2 scenes" in prompt
    assert '"narration"' in prompt
    assert "Rain fell on the seed." in prompt
