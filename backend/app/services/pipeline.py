from uuid import uuid4

from backend.app.core.config import get_settings
from backend.app.models.jobs import JobRecord, JobStatus, UploadRequest
from backend.app.services.job_store import InMemoryJobStore, job_store
from backend.app.services.mock_video import MockVideoGenerator


class LocalPipeline:
    def __init__(self, store: InMemoryJobStore = job_store) -> None:
        self.store = store
        settings = get_settings()
        self.generator = MockVideoGenerator(
            settings.storage_dir,
            duration_seconds=settings.mock_clip_seconds,
        )

    def submit(self, request: UploadRequest, *, run_inline: bool = True) -> JobRecord:
        job = JobRecord(title=request.title, segments=request.segments)
        self.store.create(job)
        if run_inline:
            self.run(job.job_id)
        return job

    def run(self, job_id: str) -> JobRecord:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")

        self.store.update_status(job_id, JobStatus.running, progress=0.0)
        try:
            total = len(job.segments)
            for index, segment in enumerate(job.segments, start=1):
                segment_id = f"{job.doc_id}-{uuid4().hex[:8]}"
                artifact = self.generator.generate(segment_id, segment)
                self.store.add_video(job_id, artifact)
                self.store.update_status(job_id, JobStatus.running, progress=index / total)
            return self.store.update_status(job_id, JobStatus.completed, progress=1.0)
        except Exception as exc:
            return self.store.update_status(
                job_id,
                JobStatus.failed,
                error=str(exc),
            )

