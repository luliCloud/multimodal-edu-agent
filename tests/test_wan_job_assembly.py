import subprocess

import pytest

from backend.app.core.config import Settings
from backend.app.models.jobs import SegmentRequest, UploadRequest, VideoArtifact
from backend.app.services.gpu_scheduler import SingleGpuScheduler
from backend.app.services.job_store import InMemoryJobStore
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.wan_video import WanVideoGenerator
import backend.app.services.pipeline as pipeline_module


def test_wan_job_exposes_combined_video(tmp_path, monkeypatch) -> None:
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg", reason="needs the cuda or wan extra")
    clips = []
    for color in ("red", "green"):
        path = tmp_path / f"{color}.mp4"
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", f"color=c={color}:s=64x64:r=8:d=0.5", "-pix_fmt", "yuv420p",
             str(path)], check=True,
        )
        clips.append(path)

    generator = WanVideoGenerator(tmp_path)
    used = iter(clips)

    def fake_generate(segment_id, segment, gpu_id):
        return VideoArtifact(segment_id=segment_id, path=str(next(used)),
                             media_type="video/mp4", duration_seconds=1, gpu_id=gpu_id)

    monkeypatch.setattr(generator, "generate", fake_generate)
    monkeypatch.setattr(pipeline_module, "get_settings",
                        lambda: Settings(generator_backend="wan", storage_dir=tmp_path))
    monkeypatch.setattr(pipeline_module, "get_wan_generator", lambda output_dir: generator)
    scheduler = SingleGpuScheduler()
    try:
        job = LocalPipeline(InMemoryJobStore(), scheduler).submit(
            UploadRequest(title="Story", segments=[SegmentRequest(text="Seed"),
                                                   SegmentRequest(text="Flower")]),
            run_inline=True,
        )
    finally:
        scheduler.workers.shutdown()

    assert job.status == "completed", job.error
    assert len(job.videos) == 3
    combined = job.videos[-1]
    assert combined.segment_id == f"{job.doc_id}-combined"
    assert combined.url == f"/media/{job.doc_id}-combined.mp4"
    assert (tmp_path / f"{job.doc_id}-combined.mp4").is_file()
