from functools import lru_cache
from pathlib import Path
from os import getenv
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Multimodal Edu Agent"
    environment: str = "local"
    storage_dir: Path = Path("storage/videos")
    generator_backend: str = getenv("GENERATOR_BACKEND", "mock")
    mock_clip_seconds: int = 3


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
