"""Web-layer settings. Pipeline settings (planner, Wan, Qwen) stay in backend.app.core."""

from functools import lru_cache
from os import getenv
from pathlib import Path

from pydantic import BaseModel

from backend.app.core.config import get_settings as get_pipeline_settings

REPO_ROOT = Path(__file__).resolve().parents[3]


def _path(name: str, default: str) -> Path:
    path = Path(getenv(name, default))
    return path if path.is_absolute() else REPO_ROOT / path


class WebSettings(BaseModel):
    app_name: str = "Storyreel Studio API"
    # Project metadata, uploaded PDFs and editable scripts.
    data_dir: Path = _path("WEBAPP_DATA_DIR", "storage/webapp/projects")
    # Rendered videos. Everything under this directory is served and listed in the library;
    # swap LocalDirectoryStorage for a cloud implementation to move it off disk.
    video_dir: Path = _path("WEBAPP_VIDEO_DIR", "storage/webapp/videos")
    # Scratch space the pipeline writes into before clips are moved to video storage.
    render_dir: Path = _path("WEBAPP_RENDER_DIR", "storage/webapp/render-scratch")
    # mock | cuda_probe | wan run the existing generators; directory picks up videos
    # rendered elsewhere (e.g. by render_short_script on a GPU box) from prerendered_dir.
    render_backend: str = getenv("RENDER_BACKEND", get_pipeline_settings().generator_backend)
    prerendered_dir: Path = _path("PRERENDERED_DIR", "storage/demo")
    planner_backend: str = get_pipeline_settings().planner_backend
    max_upload_bytes: int = 10 * 1024 * 1024
    cors_origins: list[str] = [
        origin.strip()
        for origin in getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]


@lru_cache
def get_settings() -> WebSettings:
    settings = WebSettings()
    for directory in (settings.data_dir, settings.video_dir, settings.render_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
