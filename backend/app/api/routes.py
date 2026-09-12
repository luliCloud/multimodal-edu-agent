from fastapi import APIRouter, HTTPException

from backend.app.models.jobs import JobRecord, UploadRequest, VideoArtifact
from backend.app.services.job_store import job_store
from backend.app.services.pipeline import LocalPipeline

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload", response_model=JobRecord)
def upload(request: UploadRequest) -> JobRecord:
    return LocalPipeline().submit(request, run_inline=True)


@router.get("/status/{job_id}", response_model=JobRecord)
def status(job_id: str) -> JobRecord:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/videos/{doc_id}", response_model=list[VideoArtifact])
def videos(doc_id: str) -> list[VideoArtifact]:
    return job_store.videos_for_doc(doc_id)

