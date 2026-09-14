"""Grounded story planning: exact PDF sentences remain the source of truth."""

import json
import os
import subprocess
import sys
from math import ceil

from backend.app.services.gpu_scheduler import scheduler
from backend.app.services.pdf_keywords import extract_video_keywords, extract_video_scenes


def source_sentences(pages: list[str]) -> list[dict]:
    return [{"id": index, "page": scene["pages"][0], "text": scene["text"]}
            for index, scene in enumerate(extract_video_scenes(pages, limit=200), 1)]


def scene_groups(sentences: list[dict]) -> list[list[int]]:
    """Assign every source sentence to one time-ordered scene before prompting."""
    count = len(sentences)
    if count > 32:
        raise ValueError("Storyboard supports up to 32 narrative sentences; split longer PDFs")
    scene_count = min(count, 8, max(3, ceil(count / 2)))
    return [[item["id"] for item in sentences[start:end]]
            for index in range(scene_count)
            for start, end in [(index * count // scene_count,
                                (index + 1) * count // scene_count)]]


def plan_story(pages: list[str], backend: str = "extractive") -> dict:
    if backend == "extractive":
        scenes = extract_video_scenes(pages)
        for scene in scenes:
            scene["keywords"] = [item["keyword"] for item in
                                 extract_video_keywords([scene["text"]], limit=5)]
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
            timeout=600, check=False, env=os.environ.copy(),
        )
        if result.returncode:
            raise RuntimeError(f"Story planner failed: {result.stderr[-1000:]}")
        return json.loads(result.stdout)

    draft = scheduler.run_on_gpu(run_worker)
    summary = draft.get("summary")
    raw_scenes = draft.get("scenes")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(raw_scenes, list):
        raise ValueError("Story planner returned an invalid summary or scene list")
    if len(raw_scenes) != len(groups):
        raise ValueError(f"Story planner must return exactly {len(groups)} scenes")

    by_id = {item["id"]: item for item in sentences}
    scenes = []
    for ids, draft_scene in zip(groups, raw_scenes):
        if not isinstance(draft_scene, dict):
            raise ValueError("Story planner returned an invalid scene")
        visual = draft_scene.get("visual_prompt")
        motion = draft_scene.get("motion")
        if (not isinstance(visual, str) or not visual.strip() or
                not isinstance(motion, str) or not motion.strip()):
            raise ValueError("Story planner returned an invalid scene")
        evidence = [by_id[item] for item in ids]
        text = " ".join(item["text"] for item in evidence)
        scenes.append({
            "index": len(scenes) + 1, "text": text,
            "pages": sorted({item["page"] for item in evidence}),
            "source_sentence_ids": ids,
            "visual_prompt": visual.strip(), "motion": motion.strip(),
            "keywords": [item["keyword"] for item in extract_video_keywords([text], limit=5)],
        })
    return {"summary": summary.strip(), "planner_backend": backend,
            "source_sentences": sentences, "scenes": scenes}
