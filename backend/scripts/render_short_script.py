"""Stage 2 demo: render a reviewed script.json without rerunning Qwen."""

import argparse
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.models.shorts import ShortPlan
from backend.app.services.job_store import InMemoryJobStore
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.script_video import upload_request_from_script


def render_script(plan: ShortPlan, backend: str, output_dir: Path,
                  scene_numbers: list[int] | None = None):
    settings = Settings(generator_backend=backend, storage_dir=output_dir)
    pipeline = LocalPipeline(store=InMemoryJobStore(), settings=settings)
    return pipeline.submit(upload_request_from_script(plan, scene_numbers), run_inline=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scene clips and a combined video from script.json"
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("--backend", choices=("mock", "cuda_probe", "wan"), default="mock")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scene", type=int, action="append",
                        help="Render only this scene number; repeat for multiple scenes")
    args = parser.parse_args()
    plan = ShortPlan.model_validate_json(args.script.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.script.parent / "videos"
    job = render_script(plan, args.backend, output_dir, args.scene)
    print("backend:", args.backend)
    print("status:", job.status.value)
    if job.error:
        print("error:", job.error)
    for video in job.videos:
        label = "combined" if video.segment_id.endswith("-combined") else video.segment_id
        print(f"video ({label}):", Path(video.path).resolve())
    if job.status.value != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
