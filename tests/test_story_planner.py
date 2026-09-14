import pytest

import backend.app.services.story_planner as story_planner


def test_qwen_plan_keeps_exact_source_evidence(monkeypatch) -> None:
    draft = {"summary": "A seed receives sun and water, grows roots and leaves, and flowers.",
             "scenes": [
                 {"source_sentence_ids": [1], "visual_prompt": "A seed rests in soil.",
                  "motion": "The seed settles into the soil."},
                 {"source_sentence_ids": [2], "visual_prompt": "Rain reaches the seed.",
                  "motion": "Raindrops fall toward the seed."},
                 {"source_sentence_ids": [3], "visual_prompt": "A root grows.",
                  "motion": "The root extends downward."},
             ]}
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: draft)
    pages = ["A seed slept in soil. Rain fell on the seed. A root grew down into soil."]
    plan = story_planner.plan_story(pages, "qwen")
    assert plan["scenes"][1]["text"] == "Rain fell on the seed."
    assert plan["scenes"][1]["source_sentence_ids"] == [2]
    assert plan["scenes"][1]["pages"] == [1]


def test_qwen_plan_rejects_nonexistent_source(monkeypatch) -> None:
    draft = {"summary": "A seed grows.", "scenes": [
        {"source_sentence_ids": [99], "visual_prompt": "A flower.", "motion": "It blooms."}
        for _ in range(3)]}
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: draft)
    with pytest.raises(ValueError, match="ungrounded"):
        story_planner.plan_story(["A little seed slept in dark soil."], "qwen")


def test_qwen_plan_requires_new_seeds_ending(monkeypatch) -> None:
    draft = {"summary": "A seed receives rain and grows into a flower with new seeds.",
             "scenes": [
                 {"source_sentence_ids": [1], "visual_prompt": "A seed in soil.",
                  "motion": "The camera moves closer."},
                 {"source_sentence_ids": [2], "visual_prompt": "Rain falls.",
                  "motion": "Drops fall onto soil."},
                 {"source_sentence_ids": [3], "visual_prompt": "A flower opens.",
                  "motion": "Petals open."},
             ]}
    monkeypatch.setattr(story_planner.scheduler, "run_on_gpu", lambda callback: draft)
    pages = ["A little seed slept in the soil. Rain fell onto the ground. "
             "A beautiful flower appeared in spring. New seeds appeared inside the flower."]
    with pytest.raises(ValueError, match="new-seeds ending"):
        story_planner.plan_story(pages, "qwen")
