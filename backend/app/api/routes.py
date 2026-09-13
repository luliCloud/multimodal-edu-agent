from fastapi import APIRouter, Body, HTTPException

from backend.app.models.jobs import JobRecord, SegmentRequest, UploadRequest, VideoArtifact
from backend.app.services.job_store import job_store
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.pdf_keywords import extract_pdf_pages, extract_video_keywords, extract_video_scenes

router = APIRouter()


def _pdf_plan(pdf: bytes) -> dict:
    try:
        pages = extract_pdf_pages(pdf)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    scenes = extract_video_scenes(pages)
    if not scenes:
        raise HTTPException(status_code=422, detail="PDF has no usable narrative text")
    title = next((line.strip() for line in pages[0].splitlines() if line.strip()), "Untitled PDF")
    return {"title": title, "page_count": len(pages),
            "keywords": extract_video_keywords(pages), "scenes": scenes}


@router.post("/pdf/keywords")
def pdf_keywords(pdf: bytes = Body(..., media_type="application/pdf")) -> dict:
    """Accept raw PDF bytes; return candidate topics with source page numbers."""
    try:
        pages = extract_pdf_pages(pdf)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"page_count": len(pages), "keywords": extract_video_keywords(pages)}


@router.post("/pdf/plan")
def pdf_plan(pdf: bytes = Body(..., media_type="application/pdf")) -> dict:
    return _pdf_plan(pdf)


@router.post("/pdf/jobs", response_model=JobRecord)
def pdf_jobs(pdf: bytes = Body(..., media_type="application/pdf")) -> JobRecord:
    plan = _pdf_plan(pdf)
    request = UploadRequest(title=plan["title"], segments=[
        SegmentRequest(title=f"Scene {scene['index']}", text=scene["text"])
        for scene in plan["scenes"]
    ])
    return LocalPipeline().submit(request, run_inline=False)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload", response_model=JobRecord)
def upload(request: UploadRequest) -> JobRecord:
    return LocalPipeline().submit(request, run_inline=False)


@router.get("/status/{job_id}", response_model=JobRecord)
def status(job_id: str) -> JobRecord:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/videos/{doc_id}", response_model=list[VideoArtifact])
def videos(doc_id: str) -> list[VideoArtifact]:
    return job_store.videos_for_doc(doc_id)
