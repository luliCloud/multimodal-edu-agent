"""Generate one real model clip from a selected PDF scene."""

import argparse
import logging
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.services.pdf_keywords import extract_pdf_pages, extract_video_keywords, extract_video_scenes
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.wan_video import WanVideoGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--scene", type=int, default=1, help="1-based scene number")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if get_settings().generator_backend != "wan":
        parser.error("Set GENERATOR_BACKEND=wan")
    scenes = extract_video_scenes(extract_pdf_pages(args.pdf.read_bytes()))
    if not 1 <= args.scene <= len(scenes):
        parser.error(f"Scene must be between 1 and {len(scenes)}")
    scene = scenes[args.scene - 1]
    segment = SegmentRequest(
        title=f"Scene {args.scene}", text=scene["text"],
        keywords=[item["keyword"] for item in extract_video_keywords([scene["text"]], limit=5)],
    )
    print("prompt:", WanVideoGenerator.build_prompt(segment), flush=True)
    job = LocalPipeline().submit(UploadRequest(title=args.pdf.stem, segments=[segment]))
    print("status:", job.status, "error:", job.error, flush=True)
    for video in job.videos:
        print("video:", video.path, flush=True)
    if job.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
