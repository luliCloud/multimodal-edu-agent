"""Stage 2 demo: render a reviewed script.json without rerunning Qwen."""

import argparse
import os
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


def render_script(plan: ShortPlan, backend: str, output_dir: Path,
                  scene_numbers: list[int] | None = None,
                  progress_callback: Callable[[str], None] | None = None):
    settings = Settings(generator_backend=backend, storage_dir=output_dir)
    pipeline = LocalPipeline(
        store=InMemoryJobStore(), settings=settings,
        progress_callback=progress_callback,
    )
    return pipeline.submit(upload_request_from_script(plan, scene_numbers), run_inline=True)


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


def render_script_with_progress(plan: ShortPlan, backend: str, output_dir: Path,
                                scene_numbers: list[int] | None = None):
    selected = scene_numbers or [scene.scene for scene in plan.scenes]
    print(f"[video] script validated: {plan.title}", flush=True)
    print(f"[video] backend: {backend}; scenes: {selected}", flush=True)
    print(f"[video] output directory: {output_dir.resolve()}", flush=True)
    if backend == "wan":
        print(
            "[video] Wan configuration: "
            f"{os.getenv('WAN_WIDTH', '576')}x{os.getenv('WAN_HEIGHT', '320')}, "
            f"{os.getenv('WAN_NUM_FRAMES', '33')} frames, "
            f"{os.getenv('WAN_STEPS', '20')} steps; loading may take a minute",
            flush=True,
        )
    with generation_heartbeat():
        return render_script(
            plan, backend, output_dir, scene_numbers,
            progress_callback=lambda message: print(f"[video] {message}", flush=True),
        )


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
    job = render_script_with_progress(plan, args.backend, output_dir, args.scene)
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
