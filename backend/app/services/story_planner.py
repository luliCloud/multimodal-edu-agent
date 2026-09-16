"""Grounded story planning: exact PDF sentences remain the source of truth."""

import json
import os
import subprocess
import sys
from math import ceil
from pathlib import Path

from pydantic import ValidationError

from backend.app.services.gpu_scheduler import scheduler
from backend.app.services.pdf_keywords import extract_video_keywords, extract_video_scenes
from backend.app.services.storyboard_schema import (
    ScopeExceededError,
    StoryboardDraft,
    StoryPlanningError,
    narration_seconds,
)

MAX_SOURCE_SENTENCES = 32
REPO_ROOT = Path(__file__).resolve().parents[3]


def source_sentences(pages: list[str]) -> list[dict]:
    return [{"id": index, "page": scene["pages"][0], "text": scene["text"]}
            for index, scene in enumerate(extract_video_scenes(pages, limit=200), 1)]


def scene_groups(sentences: list[dict]) -> list[list[int]]:
    """Assign every source sentence to one time-ordered scene before prompting."""
    count = len(sentences)
    if count > MAX_SOURCE_SENTENCES:
        raise ScopeExceededError(f"Storyboard supports up to {MAX_SOURCE_SENTENCES} "
                                 "narrative sentences; split longer PDFs")
    scene_count = min(count, 8, max(3, ceil(count / 2)))
    return [[item["id"] for item in sentences[start:end]]
            for index in range(scene_count)
            for start, end in [(index * count // scene_count,
                                (index + 1) * count // scene_count)]]


def _worker_env() -> dict[str, str]:
    """The worker imports backend.* for the shared schema, so it needs the repo
    root on sys.path even when the API process was started somewhere else."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return env


def plan_story(pages: list[str], backend: str = "extractive") -> dict:
    if backend == "extractive":
        scenes = extract_video_scenes(pages)
        for scene in scenes:
            scene["keywords"] = [item["keyword"] for item in
                                 extract_video_keywords([scene["text"]], limit=5)]
            # The baseline does not rewrite: the narrator reads the source text.
            scene["narration"] = scene["text"]
            scene["narration_seconds"] = narration_seconds(scene["text"])
        return {"summary": None, "planner_backend": backend, "scenes": scenes}
    if backend != "qwen":
        raise ValueError(f"Unknown planner backend: {backend}")

    sentences = source_sentences(pages)
    if not sentences:
        return {"summary": None, "planner_backend": backend, "scenes": []}
    groups = scene_groups(sentences)

    def run_worker(gpu_id: int) -> dict:
        result = subprocess.run(
            [sys.executable, "-m", "backend.scripts.qwen_storyboard_worker", str(gpu_id)],
            input=json.dumps({"sentences": sentences, "scene_groups": groups}),
            text=True, capture_output=True,
            timeout=600, check=False, env=_worker_env(),
        )
        if result.returncode:
            raise StoryPlanningError(f"Story planner failed: {result.stderr[-1000:]}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise StoryPlanningError(f"Story planner returned invalid JSON: {exc}") from exc

    draft = scheduler.run_on_gpu(run_worker)
    try:
        storyboard = StoryboardDraft.model_validate(draft, context={"scene_count": len(groups)})
    except ValidationError as exc:
        raise StoryPlanningError(str(exc)) from exc

    by_id = {item["id"]: item for item in sentences}
    scenes = []
    for ids, draft_scene in zip(groups, storyboard.scenes):
        evidence = [by_id[item] for item in ids]
        text = " ".join(item["text"] for item in evidence)
        scenes.append({
            "index": len(scenes) + 1, "text": text,
            "pages": sorted({item["page"] for item in evidence}),
            "source_sentence_ids": ids,
            "visual_prompt": draft_scene.visual_prompt, "motion": draft_scene.motion,
            "narration": draft_scene.narration,
            "narration_seconds": narration_seconds(draft_scene.narration),
            "keywords": [item["keyword"] for item in extract_video_keywords([text], limit=5)],
        })
    return {"summary": storyboard.summary, "planner_backend": backend,
            "source_sentences": sentences, "scenes": scenes}
