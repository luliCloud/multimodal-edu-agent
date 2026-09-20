from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
import re

from backend.app.models.jobs import JobRecord, SegmentRequest, UploadRequest, VideoArtifact
from backend.app.models.shorts import ShortPlan
from backend.app.services.job_store import job_store
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.pdf_keywords import extract_pdf_pages, extract_video_keywords
from backend.app.services.story_planner import plan_story
from backend.app.services.storyboard_schema import ScopeExceededError, StoryPlanningError
from backend.app.services.script_video import upload_request_from_script
from backend.app.core.config import get_settings

router = APIRouter()


def _pdf_plan(pdf: bytes) -> dict:
    try:
        pages = extract_pdf_pages(pdf)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    title = next((line.strip() for line in pages[0].splitlines() if line.strip()), "Untitled PDF")
    try:
        planned = plan_story(pages, get_settings().planner_backend, title=title)
    except ScopeExceededError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except StoryPlanningError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    scenes = planned["scenes"]
    if not scenes:
        raise HTTPException(status_code=422, detail="PDF has no usable narrative text")
    for scene in scenes:
        scene.setdefault("keywords", [item["keyword"] for item in
                                      extract_video_keywords([scene["text"]], limit=5)])
    return {"title": title, "page_count": len(pages),
            "keywords": extract_video_keywords(pages), **planned}


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
        SegmentRequest(title=f"Scene {scene['index']}", text=scene["text"],
                       keywords=scene["keywords"], visual_prompt=scene.get("visual_prompt"),
                       motion=scene.get("motion"), narration=scene.get("narration"),
                       narration_seconds=scene.get("narration_seconds"))
        for scene in plan["scenes"]
    ])
    return LocalPipeline().submit(request, run_inline=False)


@router.post("/scripts/jobs", response_model=JobRecord)
def script_jobs(script: ShortPlan) -> JobRecord:
    """Generate video from an already reviewed script without rerunning the planner."""
    return LocalPipeline().submit(upload_request_from_script(script), run_inline=False)


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


@router.get("/media/{filename}")
def media(filename: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.mp4", filename):
        raise HTTPException(status_code=404, detail="Video not found")
    path = get_settings().storage_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")
