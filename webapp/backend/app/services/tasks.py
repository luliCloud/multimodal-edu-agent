"""Background execution for the two long-running stages (planning and rendering).

Both run on a small thread pool inside the API process. GPU access is still serialized by
the pipeline's own SingleGpuScheduler, so Qwen and Wan never share the GPU. Moving this to
Celery later only means replacing ``executor.submit`` with a task dispatch.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from webapp.backend.app.config import WebSettings
from webapp.backend.app.models import (
    Project, ProjectEvent, ProjectStatus, RenderState, utcnow,
)
from webapp.backend.app.services.pipeline_bridge import (
    UserFacingError, import_library_assets, plan_script, read_pdf, render_videos,
    validate_script,
)
from webapp.backend.app.services.projects import ProjectStore
from webapp.backend.app.services.storage import VideoStorage

logger = logging.getLogger(__name__)


class Conflict(Exception):
    pass


class TaskRunner:
    def __init__(self, store: ProjectStore, storage: VideoStorage, settings: WebSettings) -> None:
        self.store = store
        self.storage = storage
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="webapp-task")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ planning

    def start_planning(self, project_id: str, *, fresh: bool = False) -> Project:
        def reset(project: Project) -> None:
            if not fresh and project.status in (ProjectStatus.planning, ProjectStatus.extracting,
                                  ProjectStatus.rendering):
                raise Conflict("This project is already being processed")
            project.status = ProjectStatus.extracting
            project.stage_message = "Queued for text extraction"
            project.error = None
        project = self.store.update(project_id, reset)
        self.executor.submit(self._plan, project_id)
        return project

    def _plan(self, project_id: str) -> None:
        backend = self.settings.planner_backend
        try:
            self.store.log(project_id, "Reading the PDF and extracting text",
                           stage_message="Extracting text")
            pdf = self.store.pdf_path(project_id).read_bytes()
            pages, title, preview = read_pdf(pdf)
            self.store.log(
                project_id,
                f"Extracted {preview.page_count} page(s) and {preview.sentence_count} "
                "narrative sentences",
                status=ProjectStatus.planning, source=preview, title=title,
                stage_message=("Designing scenes with Qwen (usually 1-3 minutes)"
                               if backend == "qwen" else "Grouping sentences into scenes"),
            )
            script = plan_script(pages, title, backend)
            issues = validate_script(script)

            def finish(project: Project) -> None:
                project.script = script
                project.title = script.title
                project.issues = issues
                project.script_revision += 1
                project.status = ProjectStatus.draft
                project.stage_message = "Script ready for review"
                project.events.append(_event(
                    f"Storyboard ready with {len(script.scenes)} scenes"))
            self.store.update(project_id, finish)
            assets_dir = self.store.assets_dir(project_id)
            if (self.settings.render_backend == "wan"
                    and not (assets_dir / ".manual-assets").is_file()):
                project = self.store.get(project_id)
                source, imported = import_library_assets(
                    assets_dir=assets_dir,
                    library_dir=self.settings.prerendered_dir,
                    names=[project.filename.rsplit(".", 1)[0], project.title],
                )
                if imported:
                    self.store.log(
                        project_id,
                        f"Loaded visual bible with {len(imported)} asset(s) from {source}",
                    )
        except UserFacingError as exc:
            self._fail(project_id, str(exc))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, logged in full here
            logger.exception("Planning failed for %s", project_id)
            self._fail(project_id, f"Unexpected planning error: {exc}")

    def _fail(self, project_id: str, message: str) -> None:
        try:
            self.store.log(project_id, message, level="error", status=ProjectStatus.failed,
                           error=message, stage_message="Conversion failed")
        except KeyError:
            pass  # deleted while running

    # ------------------------------------------------------------ rendering

    def start_render(self, project_id: str, scenes: list[int] | None) -> Project:
        backend = self.settings.render_backend

        def begin(project: Project) -> None:
            if project.status in (ProjectStatus.extracting, ProjectStatus.planning,
                                  ProjectStatus.rendering):
                raise Conflict("This project is busy; wait for the current step to finish")
            if project.script is None or not project.script.scenes:
                raise Conflict("There is no script to render yet")
            available = [scene.scene for scene in project.script.scenes]
            selected = sorted(set(scenes)) if scenes else available
            unknown = [number for number in selected if number not in available]
            if unknown:
                raise Conflict(f"Unknown scene numbers: {unknown}")
            project.render = RenderState(backend=backend, scenes=selected)
            project.status = ProjectStatus.rendering
            project.error = None
            project.stage_message = f"Rendering {len(selected)} scene(s) with {backend}"
            project.events.append(_event(
                f"Started {backend} render of scene(s) {', '.join(map(str, selected))}"))
        project = self.store.update(project_id, begin)
        self.executor.submit(self._render, project_id)
        return project

    def _render(self, project_id: str) -> None:
        project = self.store.get(project_id)
        render = project.render
        assert render is not None and project.script is not None

        def on_progress(value: float) -> None:
            def mutate(current: Project) -> None:
                if current.render and current.render.render_id == render.render_id:
                    current.render.progress = round(min(max(value, 0.0), 1.0), 3)
            self.store.update(project_id, mutate)

        def on_message(message: str) -> None:
            # Pipeline messages end in "-> /absolute/server/path"; keep only the description.
            message = re.sub(r"\s*->\s*/\S.*$", "", message).strip()
            message = message[:1].upper() + message[1:]
            self.store.log(project_id, message, stage_message=message)

        try:
            videos = render_videos(
                project_id=project_id, render_id=render.render_id, script=project.script,
                selected=render.scenes, backend=render.backend, filename=project.filename,
                storage=self.storage, render_dir=self.settings.render_dir,
                prerendered_dir=self.settings.prerendered_dir,
                assets_dir=self.store.assets_dir(project_id),
                on_progress=on_progress, on_message=on_message,
                tts_enabled=self.settings.tts_enabled,
                tts_python=self.settings.tts_python,
                tts_gender=self.settings.tts_gender,
                tts_seed=self.settings.tts_seed,
            )
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, UserFacingError):
                logger.exception("Render failed for %s", project_id)
            message = str(exc) or "Video generation failed"

            def failed(current: Project) -> None:
                if current.render:
                    current.render.error = message
                    current.render.finished_at = utcnow()
                # Back to an editable state so the user can adjust and retry.
                current.status = (ProjectStatus.rendered if current.videos
                                  else ProjectStatus.draft)
                current.error = message
                current.stage_message = "Render failed"
                current.events.append(_event(message, "error"))
            try:
                self.store.update(project_id, failed)
            except KeyError:
                pass
            return

        current_prefix = f"{project_id}/{render.render_id}"
        stale: list[str] = []

        def done(current: Project) -> None:
            stale.extend({video.key.rsplit("/", 1)[0] for video in current.videos}
                         - {current_prefix})
            current.videos = videos
            current.rendered_revision = current.script_revision
            if current.render:
                current.render.progress = 1.0
                current.render.finished_at = utcnow()
            current.status = ProjectStatus.rendered
            current.stage_message = "Video ready"
            current.events.append(_event("Video ready"))
        try:
            self.store.update(project_id, done)
        except KeyError:
            stale = [current_prefix]  # project was deleted mid-render
        for prefix in stale:
            self.storage.delete_prefix(prefix)

def _event(message: str, level: str = "info") -> ProjectEvent:
    return ProjectEvent(message=message, level=level)
