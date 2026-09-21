from uuid import uuid4
from pathlib import Path
from collections.abc import Callable

from backend.app.core.config import Settings, get_settings
from backend.app.models.jobs import JobRecord, JobStatus, UploadRequest, VideoArtifact
from backend.app.services.job_store import InMemoryJobStore, job_store
from backend.app.services.mock_video import MockVideoGenerator
from backend.app.services.gpu_scheduler import SingleGpuScheduler, scheduler
from backend.app.services.cuda_probe_video import CudaProbeVideoGenerator
from backend.app.services.wan_video import WanVideoGenerator
from backend.app.services.video_assembly import assemble_mp4
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_wan_generator(output_dir) -> WanVideoGenerator:
    return WanVideoGenerator(output_dir)


class LocalPipeline:
    def __init__(self, store: InMemoryJobStore = job_store,
                 gpu_scheduler: SingleGpuScheduler = scheduler,
                 settings: Settings | None = None,
                 progress_callback: Callable[[str], None] | None = None) -> None:
        self.store = store
        self.scheduler = gpu_scheduler
        self.progress_callback = progress_callback
        settings = settings or get_settings()
        if settings.generator_backend == "cuda_probe":
            self.generator = CudaProbeVideoGenerator(settings.storage_dir, settings.mock_clip_seconds)
        elif settings.generator_backend == "wan":
            self.generator = get_wan_generator(settings.storage_dir)
        elif settings.generator_backend == "mock":
            self.generator = MockVideoGenerator(settings.storage_dir, settings.mock_clip_seconds)
        else:
            raise ValueError(f"Unknown generator backend: {settings.generator_backend}")

    def submit(self, request: UploadRequest, *, run_inline: bool = True) -> JobRecord:
        job = JobRecord(title=request.title, segments=request.segments)
        self.store.create(job)
        if run_inline:
            self.run(job.job_id)
        else:
            self.scheduler.workers.submit(self.run, job.job_id)
        return job

    def run(self, job_id: str) -> JobRecord:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")

        self.store.update_status(job_id, JobStatus.running, progress=0.0)
        try:
            total = len(job.segments)
            for index, segment in enumerate(job.segments, start=1):
                if self.progress_callback:
                    self.progress_callback(
                        f'scene {index}/{total}: generating "{segment.title}"'
                    )
                segment_id = f"{job.doc_id}-{uuid4().hex[:8]}"
                def generate(gpu_id: int):
                    if isinstance(self.generator, (CudaProbeVideoGenerator, WanVideoGenerator)):
                        artifact = self.generator.generate(segment_id, segment, gpu_id)
                    else:
                        artifact = self.generator.generate(segment_id, segment)
                    return artifact

                artifact = self.scheduler.run_on_gpu(generate)
                if artifact.media_type == "video/mp4":
                    artifact.url = f"/media/{Path(artifact.path).name}"
                self.store.add_video(job_id, artifact)
                self.store.update_status(job_id, JobStatus.running, progress=index / total)
                if self.progress_callback:
                    self.progress_callback(
                        f"scene {index}/{total}: complete -> {Path(artifact.path).resolve()}"
                    )
            scene_videos = [video for video in job.videos if video.media_type == "video/mp4"]
            if len(scene_videos) > 1 and len(scene_videos) == len(job.videos):
                if self.progress_callback:
                    self.progress_callback("assembling scene clips into one MP4")
                output = assemble_mp4(job.doc_id, scene_videos, self.generator.output_dir)
                self.store.add_video(job_id, VideoArtifact(
                    segment_id=f"{job.doc_id}-combined", path=str(output),
                    media_type="video/mp4",
                    duration_seconds=sum(video.duration_seconds for video in scene_videos),
                    url=f"/media/{output.name}",
                ))
                if self.progress_callback:
                    self.progress_callback(f"combined video complete -> {output.resolve()}")
            return self.store.update_status(job_id, JobStatus.completed, progress=1.0)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            return self.store.update_status(
                job_id,
                JobStatus.failed,
                error=str(exc),
            )
