"""Stage 1 demo: convert a PDF into a reviewable four-scene script bundle."""

import argparse
import json
import re
from pathlib import Path

from backend.app.services.pdf_keywords import extract_pdf_pages
from backend.app.services.shorts_planner import (
    character_reference_prompt, plan_short_story, scene_keyframe_prompt,
)


def default_output_dir(pdf: Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", pdf.stem.lower()).strip("_")
    return Path("storage/demo") / slug


def create_script_bundle(pdf: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or default_output_dir(pdf)
    pages = extract_pdf_pages(pdf.read_bytes())
    title = next((line.strip() for line in pages[0].splitlines() if line.strip()), pdf.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_short_story(pages, title)
    script_path = output_dir / "script.json"
    script_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    prompts = {"reference": character_reference_prompt(plan),
               "keyframes": [scene_keyframe_prompt(plan, scene) for scene in plan.scenes]}
    prompt_path = output_dir / "image_prompts.json"
    prompt_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    return script_path, prompt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one PDF to a reviewed script.json")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    script_path, prompt_path = create_script_bundle(args.pdf, args.output_dir)
    plan = json.loads(script_path.read_text(encoding="utf-8"))
    print("script:", script_path)
    print("image prompts:", prompt_path)
    print("summary:", plan["summary"])
    for scene in plan["scenes"]:
        print(scene["scene"], scene["source_sentence_ids"], scene["action"],
              "|", scene["narration"])


if __name__ == "__main__":
    main()
