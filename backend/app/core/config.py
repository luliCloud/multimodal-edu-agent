from functools import lru_cache
from pathlib import Path
from os import getenv
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Multimodal Edu Agent"
    environment: str = "local"
    storage_dir: Path = Path(getenv("STORAGE_DIR", "storage/videos"))
    generator_backend: str = getenv("GENERATOR_BACKEND", "mock")
    planner_backend: str = getenv(
        "PLANNER_BACKEND", "qwen" if getenv("GENERATOR_BACKEND", "mock") == "wan" else "extractive"
    )
    planner_model_id: str = getenv("PLANNER_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    planner_max_attempts: int = int(getenv("PLANNER_MAX_ATTEMPTS", "3"))
    mock_clip_seconds: int = 3


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
