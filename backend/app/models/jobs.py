from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field
from uuid import uuid4


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class SegmentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    title: str | None = None


class UploadRequest(BaseModel):
    title: str
    segments: list[SegmentRequest] = Field(..., min_length=1)


class VideoArtifact(BaseModel):
    segment_id: str
    path: str
    media_type: str
    duration_seconds: int


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    segments: list[SegmentRequest]
    videos: list[VideoArtifact] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

