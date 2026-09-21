"""One-command PDF -> script -> video demo for the single-GPU pipeline."""

import argparse
from pathlib import Path

from backend.app.models.shorts import ShortPlan
from backend.scripts.plan_short_pdf import create_script_bundle, default_output_dir
from backend.scripts.render_short_script import render_script_with_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete two-stage short-video demo")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--backend", choices=("mock", "cuda_probe", "wan"), default="mock")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--character-reference", type=Path,
        help="Optional canonical character PNG; human stories create one if omitted.",
    )
    parser.add_argument(
        "--keyframe-dir", type=Path,
        help="Optional directory of scene_NN_keyframe.png assets.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or default_output_dir(args.pdf)
    print("stage 1/2: PDF -> script", flush=True)
    script_path, prompt_path = create_script_bundle(args.pdf, output_dir)
    print("script:", script_path, flush=True)
    print("image prompts:", prompt_path, flush=True)
    print("stage 2/2: script -> video", flush=True)
    plan = ShortPlan.model_validate_json(script_path.read_text(encoding="utf-8"))
    is_human = any(word in plan.character.kind.lower()
                   for word in ("human", "person", "child", "girl", "boy"))
    character_reference = args.character_reference
    if character_reference is None and is_human:
        character_reference = output_dir / "assets" / "character_reference.png"
    job = render_script_with_progress(
        plan, args.backend, output_dir / "videos",
        final_output=output_dir / "final.mp4",
        character_reference=character_reference,
        keyframe_dir=args.keyframe_dir or output_dir / "assets",
    )
    print("status:", job.status.value, flush=True)
    for video in job.videos:
        print("video:", Path(video.path).resolve(), flush=True)
    if job.status.value != "completed":
        print("error:", job.error, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
