"""Public planner facade for extractive fallback and adaptive Qwen plans."""

from backend.app.services.pdf_keywords import extract_video_keywords, extract_video_scenes
from backend.app.services.shorts_planner import (
    character_reference_prompt,
    plan_short_story,
    story_scene_groups,
    source_sentences,
)
from backend.app.services.storyboard_schema import narration_seconds


def scene_groups(sentences: list[dict]) -> list[list[int]]:
    """Compatibility wrapper for the adaptive 4-8 scene grouping policy."""
    return story_scene_groups(sentences)


def plan_story(pages: list[str], backend: str = "extractive",
               title: str = "Untitled PDF") -> dict:
    if backend == "extractive":
        scenes = extract_video_scenes(pages, limit=4)
        for scene in scenes:
            scene["keywords"] = [item["keyword"] for item in
                                 extract_video_keywords([scene["text"]], limit=5)]
            scene["narration"] = scene["text"]
            scene["narration_seconds"] = narration_seconds(scene["text"])
        return {
            "summary": None,
            "planner_backend": backend,
            "duration_seconds": 15,
            "scenes": scenes,
        }
    if backend != "qwen":
        raise ValueError(f"Unknown planner backend: {backend}")

    plan = plan_short_story(pages, title)
    result = plan.model_dump()
    result["planner_backend"] = backend
    result["character_reference_prompt"] = character_reference_prompt(plan)
    for scene in result["scenes"]:
        scene["index"] = scene["scene"]
    return result
