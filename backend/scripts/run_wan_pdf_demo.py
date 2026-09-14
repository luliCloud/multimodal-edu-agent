"""Generate one real model clip from a selected PDF scene."""

import argparse
import json
import logging
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.services.pdf_keywords import extract_pdf_pages
from backend.app.services.story_planner import plan_story
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.wan_video import WanVideoGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--plan-file", type=Path, help="Use a reviewed JSON plan instead of rerunning Qwen")
    parser.add_argument("--scene", type=int, action="append", help="1-based scene number; repeat to select multiple")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if get_settings().generator_backend != "wan":
        parser.error("Set GENERATOR_BACKEND=wan")
    planned = (json.loads(args.plan_file.read_text(encoding="utf-8")) if args.plan_file else
               plan_story(extract_pdf_pages(args.pdf.read_bytes()), get_settings().planner_backend))
    scenes = planned["scenes"]
    plan_path = Path("storage/plans") / f"{args.pdf.stem}-storyboard.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(planned, indent=2, ensure_ascii=False), encoding="utf-8")
    print("plan:", plan_path, flush=True)
    print("summary:", planned["summary"], flush=True)
    selected = args.scene or [1]
    if any(not 1 <= number <= len(scenes) for number in selected):
        parser.error(f"Scene must be between 1 and {len(scenes)}")
    segments = [SegmentRequest(
        title=f"Scene {number}", text=scenes[number - 1]["text"],
        keywords=scenes[number - 1]["keywords"],
        visual_prompt=scenes[number - 1].get("visual_prompt"),
        motion=scenes[number - 1].get("motion"),
    ) for number in selected]
    for number, segment in zip(selected, segments):
        print(f"scene {number} prompt:", WanVideoGenerator.build_prompt(segment), flush=True)
    job = LocalPipeline().submit(UploadRequest(title=args.pdf.stem, segments=segments))
    print("status:", job.status, "error:", job.error, flush=True)
    for video in job.videos:
        print("video:", video.path, flush=True)
    if job.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
