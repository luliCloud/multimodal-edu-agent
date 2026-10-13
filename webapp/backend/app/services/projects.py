"""File-backed project store: one directory per uploaded PDF.

    <data_dir>/<project_id>/project.json   metadata, editable script, render state
    <data_dir>/<project_id>/source.pdf     the uploaded document
    <data_dir>/<project_id>/assets/        optional character reference / keyframes for Wan
"""

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from threading import RLock

from webapp.backend.app.models import BUSY_STATUSES, Project, ProjectEvent, ProjectStatus, utcnow

MAX_EVENTS = 200
_ID = re.compile(r"^[a-f0-9]{12}$")


class ProjectNotFound(KeyError):
    pass


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._cache: dict[str, Project] = {}
        self._load_all()

    def _load_all(self) -> None:
        for path in self.root.glob("*/project.json"):
            try:
                project = Project.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if project.status in BUSY_STATUSES:
                # Background work does not survive a restart; surface that honestly.
                was_rendering = project.status == ProjectStatus.rendering and project.script
                project.status = ProjectStatus.draft if was_rendering else ProjectStatus.failed
                project.error = None if was_rendering else "Interrupted by a server restart"
                project.stage_message = ("Render interrupted by a server restart"
                                         if was_rendering else "Interrupted")
                if project.render and project.render.finished_at is None:
                    project.render.error = "Interrupted by a server restart"
                    project.render.finished_at = utcnow()
                self._write(project)
            self._cache[project.id] = project

    def directory(self, project_id: str) -> Path:
        if not _ID.match(project_id):
            raise ProjectNotFound(project_id)
        return self.root / project_id

    def pdf_path(self, project_id: str) -> Path:
        return self.directory(project_id) / "source.pdf"

    def assets_dir(self, project_id: str) -> Path:
        return self.directory(project_id) / "assets"

    def create(self, project: Project, pdf: bytes) -> Project:
        with self._lock:
            directory = self.directory(project.id)
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "source.pdf").write_bytes(pdf)
            self._cache[project.id] = project
            self._write(project)
            return project.model_copy(deep=True)

    def get(self, project_id: str) -> Project:
        with self._lock:
            if project_id not in self._cache:
                raise ProjectNotFound(project_id)
            return self._cache[project_id].model_copy(deep=True)

    def list(self) -> list[Project]:
        with self._lock:
            projects = [project.model_copy(deep=True) for project in self._cache.values()]
        return sorted(projects, key=lambda item: item.created_at, reverse=True)

    def update(self, project_id: str, mutate: Callable[[Project], None]) -> Project:
        """Apply ``mutate`` atomically and persist the result."""
        with self._lock:
            if project_id not in self._cache:
                raise ProjectNotFound(project_id)
            project = self._cache[project_id].model_copy(deep=True)
            mutate(project)
            project.updated_at = utcnow()
            project.events = project.events[-MAX_EVENTS:]
            self._cache[project_id] = project
            self._write(project)
            return project.model_copy(deep=True)

    def log(self, project_id: str, message: str, level: str = "info", **changes) -> Project:
        def mutate(project: Project) -> None:
            for field, value in changes.items():
                setattr(project, field, value)
            project.events.append(ProjectEvent(message=message, level=level))
        return self.update(project_id, mutate)

    def delete(self, project_id: str) -> None:
        with self._lock:
            if project_id not in self._cache:
                raise ProjectNotFound(project_id)
            del self._cache[project_id]
            shutil.rmtree(self.directory(project_id), ignore_errors=True)

    def _write(self, project: Project) -> None:
        directory = self.directory(project.id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "project.json.tmp"
        temporary.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(directory / "project.json")


def dump_script(project: Project) -> str:
    """Export the script in the CLI's script.json shape (render_short_script compatible)."""
    if project.script is None:
        return "{}"
    data = project.script.model_dump()
    for scene in data["scenes"]:
        scene.pop("custom_prompt", None)
    data.pop("planner_backend", None)
    return json.dumps(data, indent=2, ensure_ascii=False)
