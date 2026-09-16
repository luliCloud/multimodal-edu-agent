import pytest

import backend.app.services.story_planner as story_planner
from backend.app.services.storyboard_schema import StoryPlanningError


def _draft(count: int) -> dict:
    return {"summary": "The story changes from beginning to end.", "scenes": [
        {"visual_prompt": f"Scene {i}.", "motion": "Something visibly changes.",
         "narration": "The little seed waits quietly beneath the dark garden soil."}
        for i in range(count)]}


def test_qwen_plan_keeps_exact_source_evidence(monkeypatch) -> None:
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: _draft(3))
    pages = ["A seed slept in soil. Rain fell on the seed. A root grew down into soil."]
    plan = story_planner.plan_story(pages, "qwen")
    assert plan["scenes"][1]["text"] == "Rain fell on the seed."
    assert plan["scenes"][1]["source_sentence_ids"] == [2]
    assert plan["scenes"][1]["pages"] == [1]


def test_qwen_plan_rejects_missing_scene(monkeypatch) -> None:
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: _draft(2))
    with pytest.raises(ValueError, match="exactly 3 scenes"):
        story_planner.plan_story(["A seed slept in soil. Rain fell on the seed. "
                                  "A root grew down into soil."], "qwen")


def test_qwen_plan_assigns_every_sentence_and_ending(monkeypatch) -> None:
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: _draft(8))
    pages = [" ".join(f"Mia saw event {i}." for i in range(1, 18))]
    plan = story_planner.plan_story(pages, "qwen")
    assigned = [sentence_id for scene in plan["scenes"]
                for sentence_id in scene["source_sentence_ids"]]
    assert assigned == list(range(1, 18))
    assert "event 17" in plan["scenes"][-1]["text"]


def test_qwen_plan_accepts_non_seed_story(monkeypatch) -> None:
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: _draft(3))
    pages = ["Mia watched the rain. Mia jumped into a puddle. A rainbow appeared."]
    plan = story_planner.plan_story(pages, "qwen")
    assert plan["scenes"][-1]["text"] == "A rainbow appeared."


def test_qwen_plan_rejects_story_too_long() -> None:
    sentences = [{"id": i, "text": f"Event {i}."} for i in range(1, 34)]
    with pytest.raises(ValueError, match="up to 32"):
        story_planner.scene_groups(sentences)


def test_qwen_plan_carries_narration_and_its_estimated_length(monkeypatch) -> None:
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: _draft(3))
    plan = story_planner.plan_story(["A seed slept in soil. Rain fell on the seed. "
                                     "A root grew down into soil."], "qwen")
    assert plan["scenes"][0]["narration"].startswith("The little seed waits")
    assert plan["scenes"][0]["narration_seconds"] == 4.0


def test_qwen_plan_rejects_a_scene_with_no_narration(monkeypatch) -> None:
    draft = _draft(3)
    del draft["scenes"][1]["narration"]
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: draft)
    with pytest.raises(StoryPlanningError, match="narration"):
        story_planner.plan_story(["A seed slept in soil. Rain fell on the seed. "
                                  "A root grew down into soil."], "qwen")


def test_extractive_plan_narrates_the_source_text() -> None:
    plan = story_planner.plan_story(["A seed slept in soil. Rain fell on the seed."])
    scene = plan["scenes"][0]
    assert scene["narration"] == scene["text"]
    assert scene["narration_seconds"] >= 3.0
