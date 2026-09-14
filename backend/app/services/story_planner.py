"""Grounded story planning: exact PDF sentences remain the source of truth."""

import json
import os
import re
import subprocess
import sys

from backend.app.services.gpu_scheduler import scheduler
from backend.app.services.pdf_keywords import extract_video_keywords, extract_video_scenes


def source_sentences(pages: list[str]) -> list[dict]:
    return [{"id": index, "page": scene["pages"][0], "text": scene["text"]}
            for index, scene in enumerate(extract_video_scenes(pages, limit=200), 1)]


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

    def run_worker(gpu_id: int) -> dict:
        result = subprocess.run(
            [sys.executable, "-m", "backend.scripts.qwen_storyboard_worker", str(gpu_id)],
            input=json.dumps({"sentences": sentences}), text=True, capture_output=True,
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
    if not 3 <= len(raw_scenes) <= 8:
        raise ValueError("Story planner must return 3 to 8 scenes")

    by_id = {item["id"]: item for item in sentences}
    scenes = []
    previous = 0
    for draft_scene in raw_scenes:
        ids = draft_scene.get("source_sentence_ids")
        visual = draft_scene.get("visual_prompt")
        motion = draft_scene.get("motion")
        if (not isinstance(ids, list) or not ids or len(ids) > 4 or
                any(type(item) is not int or item not in by_id for item in ids) or
                ids != sorted(set(ids)) or ids[0] <= previous or
                not isinstance(visual, str) or not visual.strip() or
                not isinstance(motion, str) or not motion.strip()):
            raise ValueError("Story planner returned an ungrounded or invalid scene")
        previous = ids[-1]
        evidence = [by_id[item] for item in ids]
        text = " ".join(item["text"] for item in evidence)
        scenes.append({
            "index": len(scenes) + 1, "text": text,
            "pages": sorted({item["page"] for item in evidence}),
            "source_sentence_ids": ids,
            "visual_prompt": visual.strip(), "motion": motion.strip(),
            "keywords": [item["keyword"] for item in extract_video_keywords([text], limit=5)],
        })
    original = " ".join(item["text"] for item in sentences).lower()
    summary_lower = summary.lower()
    scene_evidence = " ".join(scene["text"] for scene in scenes).lower()
    concepts = {
        "water": r"\bwater\b", "sun": r"\b(?:sun|sunlight)\b",
        "root": r"\broots?\b", "leaves": r"\b(?:leaf|leaves)\b",
        "flower": r"\bflowers?\b", "rain": r"\brain\b", "bee": r"\bbees?\b",
    }
    for name in ("water", "sun", "root", "leaves", "flower"):
        pattern = concepts[name]
        if re.search(pattern, original) and not re.search(pattern, summary_lower):
            raise ValueError(f"Story summary omitted a source milestone: {name}")
    for name in ("rain", "root", "leaves", "flower", "bee"):
        pattern = concepts[name]
        if re.search(pattern, original) and not re.search(pattern, scene_evidence):
            raise ValueError(f"Story scenes omitted a source milestone: {name}")
    if "new seeds" in original and "new seeds" not in scenes[-1]["text"].lower():
        raise ValueError("Story scenes omitted the new-seeds ending")
    return {"summary": summary.strip(), "planner_backend": backend,
            "source_sentences": sentences, "scenes": scenes}
