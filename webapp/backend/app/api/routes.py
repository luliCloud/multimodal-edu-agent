import mimetypes
from pathlib import PurePosixPath

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, Response

from webapp.backend.app.models import (
    BUSY_STATUSES, LibraryVideo, Project, ProjectAsset, ProjectStatus, ProjectSummary,
    RenderRequest, ScriptUpdate, ServerInfo,
)
from webapp.backend.app.services.pipeline_bridge import refresh_derived_fields, validate_script
from webapp.backend.app.services.projects import ProjectNotFound, dump_script
from webapp.backend.app.services.tasks import Conflict

router = APIRouter(prefix="/api")


def _services(request: Request):
    state = request.app.state
    return state.settings, state.projects, state.storage, state.tasks


def _get(request: Request, project_id: str) -> Project:
    try:
        return request.app.state.projects.get(project_id)
    except ProjectNotFound:
        raise HTTPException(status_code=404, detail="Project not found") from None


@router.get("/health", response_model=ServerInfo)
def health(request: Request) -> ServerInfo:
    settings, _, storage, _ = _services(request)
    return ServerInfo(planner_backend=settings.planner_backend,
                      render_backend=settings.render_backend,
                      tts_enabled=settings.tts_enabled,
                      video_dir=str(settings.video_dir), storage=storage.name)


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(request: Request) -> list[ProjectSummary]:
    return [ProjectSummary.of(project) for project in request.app.state.projects.list()]


@router.post("/projects", response_model=Project, status_code=status.HTTP_202_ACCEPTED)
async def create_project(request: Request, file: UploadFile = File(...)) -> Project:
    """Upload one PDF. Text extraction and scene planning continue in the background."""
    settings, projects, _, tasks = _services(request)
    filename = PurePosixPath(file.filename or "document.pdf").name
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="PDF exceeds the 10 MB limit")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="That file is not a PDF")
    project = Project(title=PurePosixPath(filename).stem.replace("_", " "),
                      filename=filename, size_bytes=len(data),
                      stage_message="Uploaded")
    projects.create(project, data)
    projects.log(project.id, f"Uploaded {filename}")
    return tasks.start_planning(project.id, fresh=True)


@router.get("/projects/{project_id}", response_model=Project)
def get_project(request: Request, project_id: str) -> Project:
    return _get(request, project_id)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(request: Request, project_id: str) -> Response:
    project = _get(request, project_id)
    _, projects, storage, _ = _services(request)
    projects.delete(project.id)
    storage.delete_prefix(project.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/projects/{project_id}/script", response_model=Project)
def save_script(request: Request, project_id: str, update: ScriptUpdate) -> Project:
    _, projects, _, _ = _services(request)
    script = refresh_derived_fields(update.script)
    issues = validate_script(script)

    def mutate(project: Project) -> None:
        if project.status in BUSY_STATUSES:
            raise Conflict("The project is busy; save again when the current step finishes")
        if update.base_revision is not None and update.base_revision != project.script_revision:
            raise Conflict("The script changed elsewhere; reload before saving")
        project.script = script
        project.title = script.title or project.title
        project.issues = issues
        project.script_revision += 1
        if project.status == ProjectStatus.failed:
            project.status = ProjectStatus.draft
        project.error = None
    try:
        return projects.update(_get(request, project_id).id, mutate)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/projects/{project_id}/replan", response_model=Project,
             status_code=status.HTTP_202_ACCEPTED)
def replan(request: Request, project_id: str) -> Project:
    """Discard the current script and convert the stored PDF again."""
    _, _, _, tasks = _services(request)
    try:
        return tasks.start_planning(_get(request, project_id).id)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/projects/{project_id}/render", response_model=Project,
             status_code=status.HTTP_202_ACCEPTED)
def render(request: Request, project_id: str, body: RenderRequest | None = None) -> Project:
    _, _, _, tasks = _services(request)
    try:
        return tasks.start_render(_get(request, project_id).id, body.scenes if body else None)
    except Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/projects/{project_id}/pdf")
def project_pdf(request: Request, project_id: str) -> FileResponse:
    project = _get(request, project_id)
    path = request.app.state.projects.pdf_path(project.id)
    return FileResponse(path, media_type="application/pdf", filename=project.filename,
                        content_disposition_type="inline")


@router.get("/projects/{project_id}/script.json")
def export_script(request: Request, project_id: str) -> Response:
    project = _get(request, project_id)
    if project.script is None:
        raise HTTPException(status_code=404, detail="No script yet")
    name = project.filename.rsplit(".", 1)[0] or "script"
    return Response(dump_script(project), media_type="application/json", headers={
        "Content-Disposition": f'attachment; filename="{name}-script.json"'})


_ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".json"}


def _asset_path(request: Request, project_id: str, name: str):
    _get(request, project_id)
    safe_name = PurePosixPath(name).name
    if safe_name != name or not safe_name or PurePosixPath(name).suffix.lower() not in _ASSET_SUFFIXES:
        raise HTTPException(status_code=400, detail="Invalid asset name")
    return request.app.state.projects.assets_dir(project_id) / safe_name


@router.get("/projects/{project_id}/assets", response_model=list[ProjectAsset])
def list_assets(request: Request, project_id: str) -> list[ProjectAsset]:
    _get(request, project_id)
    directory = request.app.state.projects.assets_dir(project_id)
    if not directory.is_dir():
        return []
    return [
        ProjectAsset(
            name=path.name,
            size_bytes=path.stat().st_size,
            url=f"/api/projects/{project_id}/assets/{path.name}",
        )
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in _ASSET_SUFFIXES
    ]


@router.post("/projects/{project_id}/assets", response_model=list[ProjectAsset])
async def upload_assets(request: Request, project_id: str,
                        files: list[UploadFile] = File(...)) -> list[ProjectAsset]:
    settings, projects, _, _ = _services(request)
    project = _get(request, project_id)
    if project.status in BUSY_STATUSES:
        raise HTTPException(status_code=409, detail="Wait for the current project step to finish")
    directory = projects.assets_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    uploaded: list[str] = []
    for file in files:
        raw_name = file.filename or ""
        name = PurePosixPath(raw_name).name
        if name != raw_name:
            raise HTTPException(status_code=400, detail="Invalid asset name")
        path = _asset_path(request, project_id, name)
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"{name} exceeds the upload limit")
        path.write_bytes(data)
        uploaded.append(name)
    # This marker prevents render-time auto-discovery from replacing an explicit choice.
    (directory / ".manual-assets").write_text("\n".join(uploaded), encoding="utf-8")
    projects.log(project_id, f"Uploaded {len(uploaded)} visual bible asset(s): {', '.join(uploaded)}")
    return list_assets(request, project_id)


@router.get("/projects/{project_id}/assets/{name}")
def project_asset(request: Request, project_id: str, name: str) -> FileResponse:
    path = _asset_path(request, project_id, name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name,
                        content_disposition_type="inline")


@router.get("/library", response_model=list[LibraryVideo])
def library(request: Request) -> list[LibraryVideo]:
    """Every video in the storage directory, including files copied there by hand."""
    _, projects, storage, _ = _services(request)
    titles = {project.id: project for project in projects.list()}
    items = []
    for item in storage.list():
        owner = titles.get(item.key.split("/", 1)[0]) if "/" in item.key else None
        name = item.key.rsplit("/", 1)[-1]
        kind = ("final" if name.startswith("final.") else "scene") if owner else "external"
        items.append(LibraryVideo(
            key=item.key, url=storage.url(item.key), name=name, size_bytes=item.size_bytes,
            modified_at=item.modified_at, project_id=owner.id if owner else None,
            project_title=owner.title if owner else None, kind=kind,
        ))
    return items


@router.get("/media/{key:path}")
def media(request: Request, key: str, download: bool = False) -> FileResponse:
    path = request.app.state.storage.local_path(key)
    if path is None:
        raise HTTPException(status_code=404, detail="Video not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    # FileResponse honours Range requests, so the browser can seek inside long videos.
    return FileResponse(path, media_type=media_type, filename=path.name,
                        content_disposition_type="attachment" if download else "inline")
