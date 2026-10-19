"""API contract between the FastAPI wrapper and the Next.js studio."""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProjectStatus(str, Enum):
    extracting = "extracting"   # reading PDF text
    planning = "planning"       # planner is designing scenes
    draft = "draft"             # script is ready for review and edits
    rendering = "rendering"     # video generation in progress
    rendered = "rendered"       # final video available
    failed = "failed"


BUSY_STATUSES = {ProjectStatus.extracting, ProjectStatus.planning, ProjectStatus.rendering}


class EditableScene(BaseModel):
    """One scene as the editor sees it; a superset of ShortScene that tolerates edits."""

    scene: int = 1
    action: str = ""
    location: str = ""
    weather: str = ""
    camera: str = ""
    motion: str = ""
    narration: str = ""
    narration_seconds: float = 3.0
    outfit_id: str = "default"
    state_id: str = "default"
    text: str = ""
    pages: list[int] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    source_sentence_ids: list[int] = Field(default_factory=list)
    visual_prompt: str | None = None
    keyframe_prompt: str | None = None
    # When true, visual_prompt was hand-written and is not regenerated from the fields above.
    custom_prompt: bool = False


class EditableScript(BaseModel):
    title: str = "Untitled"
    summary: str = ""
    character: dict | None = None
    style: dict | None = None
    scenes: list[EditableScene] = Field(default_factory=list)
    source_sentences: list[dict] = Field(default_factory=list)
    duration_seconds: int = 15
    planner_backend: str = "extractive"


class ValidationIssue(BaseModel):
    level: str  # "warning" | "info"
    message: str
    scene: int | None = None


class ProjectEvent(BaseModel):
    at: datetime = Field(default_factory=utcnow)
    message: str
    level: str = "info"


class ProjectVideo(BaseModel):
    key: str
    kind: str  # "final" | "scene"
    scene: int | None = None
    url: str
    duration_seconds: float | None = None
    size_bytes: int | None = None


class ProjectAsset(BaseModel):
    name: str
    size_bytes: int
    url: str


class RenderState(BaseModel):
    render_id: str = Field(default_factory=lambda: uuid4().hex[:10])
    backend: str
    scenes: list[int] = Field(default_factory=list)
    progress: float = 0.0
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    error: str | None = None


class SourcePreview(BaseModel):
    page_count: int = 0
    sentence_count: int = 0
    keywords: list[dict] = Field(default_factory=list)
    excerpt: str = ""


class Project(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    title: str
    filename: str
    size_bytes: int
    status: ProjectStatus = ProjectStatus.extracting
    stage_message: str = "Queued"
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    source: SourcePreview = Field(default_factory=SourcePreview)
    script: EditableScript | None = None
    script_revision: int = 0
    rendered_revision: int | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    render: RenderState | None = None
    videos: list[ProjectVideo] = Field(default_factory=list)
    events: list[ProjectEvent] = Field(default_factory=list)


class ProjectSummary(BaseModel):
    id: str
    title: str
    filename: str
    status: ProjectStatus
    stage_message: str
    scene_count: int
    created_at: datetime
    updated_at: datetime
    final_video_url: str | None = None

    @classmethod
    def of(cls, project: Project) -> "ProjectSummary":
        final = next((video for video in project.videos if video.kind == "final"), None)
        return cls(
            id=project.id, title=project.title, filename=project.filename,
            status=project.status, stage_message=project.stage_message,
            scene_count=len(project.script.scenes) if project.script else 0,
            created_at=project.created_at, updated_at=project.updated_at,
            final_video_url=final.url if final else None,
        )


class RenderRequest(BaseModel):
    scenes: list[int] | None = None


class ScriptUpdate(BaseModel):
    script: EditableScript
    # Optimistic concurrency: reject a save based on an older revision.
    base_revision: int | None = None


class LibraryVideo(BaseModel):
    key: str
    url: str
    name: str
    size_bytes: int
    modified_at: datetime
    project_id: str | None = None
    project_title: str | None = None
    kind: str = "external"


class ServerInfo(BaseModel):
    status: str = "ok"
    planner_backend: str
    render_backend: str
    tts_enabled: bool = False
    video_dir: str
    storage: str
