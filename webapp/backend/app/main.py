"""FastAPI wrapper around the PDF → script → video pipeline.

Run from the repository root (so both ``backend`` and ``webapp`` import):

    uvicorn webapp.backend.app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend.app.api.routes import router
from webapp.backend.app.config import WebSettings, get_settings
from webapp.backend.app.services.projects import ProjectStore
from webapp.backend.app.services.storage import LocalDirectoryStorage
from webapp.backend.app.services.tasks import TaskRunner


def create_app(settings: WebSettings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.tasks.shutdown()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.projects = ProjectStore(settings.data_dir)
    app.state.storage = LocalDirectoryStorage(settings.video_dir)
    app.state.tasks = TaskRunner(app.state.projects, app.state.storage, settings)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
