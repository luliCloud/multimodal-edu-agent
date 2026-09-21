"""Stage 2 demo: render a reviewed script.json without rerunning Qwen."""

import argparse
import os
import shutil
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.models.shorts import ShortPlan
from backend.app.services.job_store import InMemoryJobStore
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.script_video import upload_request_from_script
from backend.app.services.wan_video import WanVideoGenerator


def render_script(plan: ShortPlan, backend: str, output_dir: Path,
                  scene_numbers: list[int] | None = None,
                  character_reference: Path | None = None,
                  keyframe_dir: Path | None = None,
                  progress_callback: Callable[[str], None] | None = None):
    settings = Settings(generator_backend=backend, storage_dir=output_dir)
    pipeline = LocalPipeline(
        store=InMemoryJobStore(), settings=settings,
        progress_callback=progress_callback,
    )
    return pipeline.submit(
        upload_request_from_script(
            plan, scene_numbers, character_reference, keyframe_dir,
        ),
        run_inline=True,
    )


@contextmanager
def generation_heartbeat(interval_seconds: float = 15.0) -> Iterator[None]:
    """Print elapsed time while a synchronous model call is still running."""
    stopped = threading.Event()
    started = time.monotonic()

    def report() -> None:
        while not stopped.wait(interval_seconds):
            elapsed = int(time.monotonic() - started)
            print(f"[video] still working; elapsed {elapsed}s", flush=True)

    thread = threading.Thread(target=report, name="video-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1)


def publish_final_video(job, final_output: Path) -> Path | None:
    """Copy a combined video, or a single selected scene, to a stable demo path."""
    if job.status.value != "completed":
        return None
    scene_videos = [video for video in job.videos if video.media_type == "video/mp4"]
    combined = next(
        (video for video in reversed(scene_videos)
         if video.segment_id.endswith("-combined")),
        None,
    )
    source = combined or (scene_videos[0] if len(scene_videos) == 1 else None)
    if source is None:
        return None
    final_output.parent.mkdir(parents=True, exist_ok=True)
    if Path(source.path).resolve() != final_output.resolve():
        shutil.copy2(source.path, final_output)
    print(f"[video] final video -> {final_output.resolve()}", flush=True)
    return final_output


def render_script_with_progress(plan: ShortPlan, backend: str, output_dir: Path,
                                scene_numbers: list[int] | None = None,
                                final_output: Path | None = None,
                                character_reference: Path | None = None,
                                keyframe_dir: Path | None = None):
    selected = scene_numbers or [scene.scene for scene in plan.scenes]
    print(f"[video] script validated: {plan.title}", flush=True)
    print(f"[video] backend: {backend}; scenes: {selected}", flush=True)
    print(f"[video] output directory: {output_dir.resolve()}", flush=True)
    if backend == "wan" and character_reference is not None:
        state = "found" if character_reference.is_file() else "will be generated once"
        print(
            f"[video] visual character bible ({state}): "
            f"{character_reference.resolve()}",
            flush=True,
        )
        frames_dir = keyframe_dir or character_reference.parent
        found = [number for number in selected
                 if (frames_dir / f"scene_{number:02d}_keyframe.png").is_file()]
        layered = [number for number in selected
                   if (frames_dir / f"scene_{number:02d}_background.png").is_file()
                   and (frames_dir / f"scene_{number:02d}_character.png").is_file()]
        rig_enabled = (frames_dir / "simple_character.json").is_file()
        print(
            f"[video] scene keyframes: {len(found)}/{len(selected)} found in "
            f"{frames_dir.resolve()}; scenes {found}",
            flush=True,
        )
        print(
            f"[video] layered animation assets: {len(layered)}/{len(selected)}; "
            f"scenes {layered}",
            flush=True,
        )
        print(f"[video] articulated simple character rig: {rig_enabled}", flush=True)
    if backend == "wan":
        fps = int(os.getenv("WAN_FPS", "9"))
        frames = int(os.getenv(
            "WAN_NUM_FRAMES",
            str(WanVideoGenerator.frames_for_scene_count(len(selected), fps)),
        ))
        print(
            "[video] Wan configuration: "
            f"{os.getenv('WAN_WIDTH', '320')}x{os.getenv('WAN_HEIGHT', '576')}, "
            f"{frames} frames at {fps} fps, "
            f"{os.getenv('WAN_STEPS', '20')} steps; "
            f"keyframe motion={os.getenv('WAN_KEYFRAME_MOTION', 'stable')}; "
            "loading may take a minute",
            flush=True,
        )
    with generation_heartbeat():
        job = render_script(
            plan, backend, output_dir, scene_numbers, character_reference, keyframe_dir,
            progress_callback=lambda message: print(f"[video] {message}", flush=True),
        )
    if final_output is not None:
        publish_final_video(job, final_output)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scene clips and a combined video from script.json"
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("--backend", choices=("mock", "cuda_probe", "wan"), default="mock")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--final-output", type=Path,
                        help="Stable path for the final MP4 (default: beside script.json)")
    parser.add_argument("--scene", type=int, action="append",
                        help="Render only this scene number; repeat for multiple scenes")
    parser.add_argument(
        "--character-reference", type=Path,
        help=("One canonical character PNG shared by all scenes. For human stories, the "
              "default is assets/character_reference.png beside script.json; Wan creates "
              "it once when it is missing."),
    )
    parser.add_argument(
        "--no-character-reference", action="store_true",
        help="Disable visual identity conditioning and use the text-to-video model.",
    )
    parser.add_argument(
        "--keyframe-dir", type=Path,
        help="Directory containing scene_01_keyframe.png, scene_02_keyframe.png, etc.",
    )
    args = parser.parse_args()
    plan = ShortPlan.model_validate_json(args.script.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.script.parent / "videos"
    final_output = args.final_output or args.script.parent / "final.mp4"
    character_reference = None
    is_human = any(word in plan.character.kind.lower()
                   for word in ("human", "person", "child", "girl", "boy"))
    if not args.no_character_reference and is_human:
        character_reference = (
            args.character_reference
            or args.script.parent / "assets" / "character_reference.png"
        )
    job = render_script_with_progress(
        plan, args.backend, output_dir, args.scene, final_output, character_reference,
        args.keyframe_dir,
    )
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
