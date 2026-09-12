from datetime import datetime, timezone
from threading import Lock

from backend.app.models.jobs import JobRecord, JobStatus, VideoArtifact


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._jobs[job.job_id] = job
            return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def videos_for_doc(self, doc_id: str) -> list[VideoArtifact]:
        with self._lock:
            for job in self._jobs.values():
                if job.doc_id == doc_id:
                    return list(job.videos)
            return []

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        progress: float | None = None,
        error: str | None = None,
    ) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            if progress is not None:
                job.progress = progress
            job.error = error
            job.updated_at = datetime.now(timezone.utc)
            return job

    def add_video(self, job_id: str, artifact: VideoArtifact) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            job.videos.append(artifact)
            job.updated_at = datetime.now(timezone.utc)
            return job


job_store = InMemoryJobStore()

