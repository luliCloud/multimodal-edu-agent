"""Save an editable four-scene plan and image prompt manifest for a PDF."""

import argparse
import json
import re
from pathlib import Path

from backend.app.services.pdf_keywords import extract_pdf_pages
from backend.app.services.shorts_planner import (
    character_reference_prompt, plan_short_story, scene_keyframe_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    pages = extract_pdf_pages(args.pdf.read_bytes())
    title = next((line.strip() for line in pages[0].splitlines() if line.strip()), args.pdf.stem)
    slug = re.sub(r"[^a-z0-9]+", "_", args.pdf.stem.lower()).strip("_")
    output_dir = args.output_dir or Path("storage/shorts") / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_short_story(pages, title)
    plan_path = output_dir / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    prompts = {"reference": character_reference_prompt(plan),
               "keyframes": [scene_keyframe_prompt(plan, scene) for scene in plan.scenes]}
    prompt_path = output_dir / "image_prompts.json"
    prompt_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    print("plan:", plan_path)
    print("image prompts:", prompt_path)
    print("summary:", plan.summary)
    for scene in plan.scenes:
        print(scene.scene, scene.source_sentence_ids, scene.action, "|", scene.narration)


if __name__ == "__main__":
    main()
