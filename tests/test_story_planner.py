from pathlib import Path

import pytest

import backend.app.services.shorts_planner as shorts_planner
from backend.app.services.shorts_planner import STABLE_DEFAULT_STATE
from backend.app.services.pdf_keywords import extract_pdf_pages
from backend.app.services.story_planner import plan_story, scene_groups


def _draft() -> dict:
    return {
        "summary": "Mia watches rain, jumps, sees sunlight, and finds a rainbow.",
        "character": {
            "name": "Mia", "kind": "human child",
            "visual_identity": "young child with brown hair", "age": 6,
            "hair": "brown hair", "eyes": "brown eyes",
            "clothes": "yellow raincoat and red boots",
            "outfits": {"default": "yellow raincoat and red boots"},
            "states": {"default": "walking outside"},
            "inferred_fields": ["age", "hair", "eyes"],
        },
        "scenes": [
            {"action": "Mia watches rain through the window", "location": "home",
             "weather": "rain", "camera": "medium shot",
             "motion": "Raindrops slide down the window.",
             "narration": "Mia watches rain through the window quietly.", "outfit_id": "default"},
            {"action": "Mia jumps into a puddle", "location": "sidewalk",
             "weather": "rain", "camera": "full-body shot",
             "motion": "Mia jumps and water splashes outward.",
             "narration": "Mia jumps into a puddle with delight.", "outfit_id": "default"},
            {"action": "The clouds part to reveal sunlight", "location": "sidewalk",
             "weather": "clearing", "camera": "wide shot",
             "motion": "Clouds part as sunlight fills the sidewalk.",
             "narration": "Warm sunlight returns as the clouds part.", "outfit_id": "default"},
            {"action": "Mia sees a rainbow", "location": "sidewalk",
             "weather": "sunny", "camera": "wide sky shot",
             "motion": "A rainbow gradually brightens across the sky.",
             "narration": "Mia sees a bright rainbow across the sky.", "outfit_id": "default"},
        ],
    }


def _rain_pages() -> list[str]:
    path = Path(__file__).parents[1] / "test_pdfs/After the Rain.pdf"
    return extract_pdf_pages(path.read_bytes())


def test_four_scene_groups_cover_every_sentence_in_order() -> None:
    sentences = [{"id": i, "text": f"Event {i}."} for i in range(1, 18)]
    groups = scene_groups(sentences)
    assert len(groups) == 4
    assert [item for group in groups for item in group] == list(range(1, 18))


def test_four_scene_groups_reject_out_of_scope_documents() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        scene_groups([{"id": 1, "text": "One event."}])
    with pytest.raises(ValueError, match="up to 32"):
        scene_groups([{"id": i, "text": f"Event {i}."} for i in range(1, 34)])


def test_qwen_plan_exposes_global_character_and_four_grounded_scenes(monkeypatch) -> None:
    monkeypatch.setattr(shorts_planner.scheduler, "run_on_gpu", lambda callback: _draft())
    plan = plan_story(_rain_pages(), "qwen", title="After the Rain")
    assert plan["title"] == "After the Rain"
    assert plan["duration_seconds"] == 15
    assert len(plan["scenes"]) == 4
    assert plan["character"]["name"] == "Mia"
    assert plan["character"]["states"] == {
        "default": STABLE_DEFAULT_STATE
    }
    assert plan["character_reference_prompt"].startswith("GLOBAL_CHARACTER: Mia")
    assert [scene["index"] for scene in plan["scenes"]] == [1, 2, 3, 4]
    assert "GLOBAL_CHARACTER: Mia" in plan["scenes"][0]["visual_prompt"]
    assert "attached character reference" not in plan["scenes"][0]["visual_prompt"]
    assert "attached character reference" in plan["scenes"][0]["keyframe_prompt"]
    assert "rainbow" in plan["scenes"][-1]["text"].lower()
    assigned = [item for scene in plan["scenes"] for item in scene["source_sentence_ids"]]
    assert assigned == list(range(1, 18))


def test_qwen_plan_rejects_an_ungrounded_action(monkeypatch) -> None:
    draft = _draft()
    draft["scenes"][1]["action"] = "Mia rides a bicycle"
    monkeypatch.setattr(shorts_planner.scheduler, "run_on_gpu", lambda callback: draft)
    with pytest.raises(ValueError, match="Scene 2 action is not grounded"):
        plan_story(_rain_pages(), "qwen", title="After the Rain")


def test_extractive_fallback_uses_four_scenes_and_narration() -> None:
    plan = plan_story(_rain_pages(), "extractive", title="After the Rain")
    assert len(plan["scenes"]) == 4
    assert plan["scenes"][0]["narration"] == plan["scenes"][0]["text"]
    assert plan["scenes"][0]["narration_seconds"] >= 3.0
