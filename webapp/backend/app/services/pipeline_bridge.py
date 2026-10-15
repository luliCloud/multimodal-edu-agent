"""Adapters between the web project model and the existing PDF → script → video pipeline.

Nothing here reimplements planning or generation; it calls the same functions the CLIs use
(``extract_pdf_pages``, ``plan_story``, ``upload_request_from_script``, ``LocalPipeline``).
"""

import re
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from backend.app.core.config import Settings as PipelineSettings
from backend.app.models.jobs import JobStatus, SegmentRequest, UploadRequest
from backend.app.models.shorts import CharacterSpec, ShortPlan, StyleSpec, narration_word_range
from backend.app.services.job_store import InMemoryJobStore
from backend.app.services.pdf_keywords import extract_pdf_pages, extract_video_keywords
from backend.app.services.pipeline import LocalPipeline
from backend.app.services.script_video import upload_request_from_script
from backend.app.services.shorts_planner import scene_visual_prompt, source_sentences
from backend.app.services.story_planner import plan_story
from backend.app.services.storyboard_schema import (
    ScopeExceededError, StoryPlanningError, narration_seconds,
)
from webapp.backend.app.models import (
    EditableScene, EditableScript, ProjectVideo, SourcePreview, ValidationIssue,
)
from webapp.backend.app.services.storage import VideoStorage

HUMAN_WORDS = ("human", "person", "child", "girl", "boy")


class UserFacingError(Exception):
    """An error whose message is safe and useful to show in the UI."""


# ---------------------------------------------------------------- Stage 1: PDF → script


def read_pdf(pdf: bytes) -> tuple[list[str], str, SourcePreview]:
    try:
        pages = extract_pdf_pages(pdf)
    except ValueError as exc:
        raise UserFacingError(str(exc)) from exc
    title = next((line.strip() for line in pages[0].splitlines() if line.strip()), "Untitled")
    sentences = source_sentences(pages)
    excerpt = " ".join(item["text"] for item in sentences)
    preview = SourcePreview(
        page_count=len(pages), sentence_count=len(sentences),
        keywords=extract_video_keywords(pages, limit=12),
        excerpt=excerpt[:1200],
    )
    return pages, title, preview


def plan_script(pages: list[str], title: str, planner_backend: str) -> EditableScript:
    try:
        planned = plan_story(pages, planner_backend, title=title)
    except ScopeExceededError as exc:
        raise UserFacingError(str(exc)) from exc
    except StoryPlanningError as exc:
        raise UserFacingError(f"The planner could not produce a valid storyboard: {exc}") from exc
    if not planned["scenes"]:
        raise UserFacingError("The PDF has no usable narrative text")
    scenes = []
    for number, raw in enumerate(planned["scenes"], 1):
        scene = EditableScene.model_validate({
            **{key: value for key, value in raw.items() if value is not None},
            "scene": number,
        })
        if not scene.keywords:
            scene.keywords = [item["keyword"] for item in
                              extract_video_keywords([scene.text], limit=5)]
        scenes.append(scene)
    script = EditableScript(
        title=planned.get("title") or title,
        summary=planned.get("summary") or "",
        character=planned.get("character"),
        style=planned.get("style"),
        scenes=scenes,
        source_sentences=planned.get("source_sentences") or [],
        duration_seconds=planned.get("duration_seconds", 15),
        planner_backend=planner_backend,
    )
    return refresh_derived_fields(script)


# ---------------------------------------------------------------- editing helpers


def _character(script: EditableScript) -> CharacterSpec | None:
    if not script.character:
        return None
    try:
        return CharacterSpec.model_validate(script.character)
    except ValidationError:
        return None


def compose_visual_prompt(script: EditableScript, scene: EditableScene) -> str | None:
    """Rebuild the scene prompt with the planner's own template when the fields allow it."""
    character = _character(script)
    if (character and scene.action and scene.outfit_id in character.outfits
            and scene.state_id in character.states):
        style = StyleSpec.model_validate(script.style or {})

        class _Plan:  # scene_visual_prompt only reads .character and .style
            pass
        plan = _Plan()
        plan.character, plan.style = character, style
        return scene_visual_prompt(plan, scene)
    parts = [f"{label}: {value.strip()}" for label, value in (
        ("Action", scene.action), ("Location", scene.location),
        ("Weather", scene.weather), ("Camera", scene.camera),
    ) if value.strip()]
    if not parts:
        return None
    style = script.style or {}
    style_text = "; ".join(str(value) for value in style.values() if value)
    prefix = f"GLOBAL_STYLE: {style_text}. " if style_text else ""
    return f"{prefix}SCENE_DESCRIPTION: {'. '.join(parts)}."


def refresh_derived_fields(script: EditableScript) -> EditableScript:
    """Renumber scenes and recompute timing and non-custom prompts after an edit."""
    for number, scene in enumerate(script.scenes, 1):
        scene.scene = number
        scene.narration_seconds = narration_seconds(scene.narration or scene.text or "x")
        if not scene.custom_prompt:
            scene.visual_prompt = compose_visual_prompt(script, scene)
            scene.keyframe_prompt = (
                scene.visual_prompt + " Use the attached character reference as the exact same character."
                if scene.visual_prompt and script.character else None
            )
    return script


def _short_plan(script: EditableScript) -> ShortPlan:
    return ShortPlan.model_validate(script.model_dump(exclude={"planner_backend"}))


def _pydantic_messages(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        message = re.sub(r"^Value error, ", "", error["msg"])
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location}: {message}" if location else message)
    return messages


def validate_script(script: EditableScript) -> list[ValidationIssue]:
    """Non-blocking review hints. Rendering works even when the strict contract is broken."""
    issues: list[ValidationIssue] = []
    if not script.scenes:
        return [ValidationIssue(level="warning", message="Add at least one scene to render.")]
    for scene in script.scenes:
        if not (scene.narration.strip() or scene.text.strip()):
            issues.append(ValidationIssue(level="warning", scene=scene.scene,
                                          message="Scene has neither narration nor source text."))
        if not (scene.visual_prompt or scene.text.strip()):
            issues.append(ValidationIssue(level="warning", scene=scene.scene,
                                          message="Describe the action so the video model has something to draw."))
    total = sum(scene.narration_seconds for scene in script.scenes)
    if total > script.duration_seconds * 1.2:
        issues.append(ValidationIssue(
            level="info",
            message=f"Narration runs about {total:.0f}s, longer than the "
                    f"{script.duration_seconds}s target.",
        ))
    if script.character:
        minimum, maximum = narration_word_range(len(script.scenes))
        for scene in script.scenes:
            words = len(scene.narration.split())
            if not minimum <= words <= maximum:
                issues.append(ValidationIssue(
                    level="info", scene=scene.scene,
                    message=f"Narration is {words} words; the short-video contract expects "
                            f"{minimum}-{maximum} for {len(script.scenes)} scenes.",
                ))
        try:
            _short_plan(script)
        except ValidationError as exc:
            known = {issue.message for issue in issues}
            for message in _pydantic_messages(exc):
                if "narration must contain" in message or message in known:
                    continue
                issues.append(ValidationIssue(level="info", message=message))
    return issues


# ---------------------------------------------------------------- Stage 2: script → video


class _ProgressStore(InMemoryJobStore):
    def __init__(self, on_progress: Callable[[float], None]) -> None:
        super().__init__()
        self._on_progress = on_progress

    def update_status(self, job_id, status, *, progress=None, error=None):
        job = super().update_status(job_id, status, progress=progress, error=error)
        if progress is not None:
            self._on_progress(progress)
        return job


def _upload_request(script: EditableScript, selected: list[int], backend: str,
                    assets_dir: Path) -> UploadRequest:
    try:
        plan = _short_plan(script)
    except ValidationError:
        plan = None
    if plan is not None:
        # The reviewed script satisfies the planner contract: use the CLI's exact path.
        is_human = any(word in plan.character.kind.lower() for word in HUMAN_WORDS)
        reference = (assets_dir / "character_reference.png"
                     if backend == "wan" and is_human else None)
        if reference is not None:
            assets_dir.mkdir(parents=True, exist_ok=True)
        return upload_request_from_script(plan, selected, reference, None)
    by_number = {scene.scene: scene for scene in script.scenes}
    return UploadRequest(title=script.title, segments=[
        SegmentRequest(
            title=f"Scene {number}",
            text=by_number[number].text.strip() or by_number[number].narration.strip()
            or by_number[number].action.strip() or f"Scene {number}",
            keywords=by_number[number].keywords,
            visual_prompt=by_number[number].visual_prompt,
            motion=by_number[number].motion or None,
            narration=by_number[number].narration or None,
            narration_seconds=by_number[number].narration_seconds,
        )
        for number in selected
    ])


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _prerendered(prerendered_dir: Path, names: list[str]) -> Path:
    checked = []
    for name in dict.fromkeys(slugify(name) for name in names if slugify(name)):
        for candidate in (prerendered_dir / name / "final.mp4", prerendered_dir / f"{name}.mp4"):
            checked.append(candidate)
            if candidate.is_file():
                return candidate
    listed = ", ".join(str(path) for path in checked) or str(prerendered_dir)
    raise UserFacingError(f"No pre-rendered video found. Looked for: {listed}")


def render_videos(*, project_id: str, render_id: str, script: EditableScript,
                  selected: list[int], backend: str, filename: str,
                  storage: VideoStorage, render_dir: Path, prerendered_dir: Path,
                  assets_dir: Path, on_progress: Callable[[float], None],
                  on_message: Callable[[str], None]) -> list[ProjectVideo]:
    prefix = f"{project_id}/{render_id}"

    def stored(key: str, path: Path, kind: str, scene: int | None,
               duration: float | None, move: bool) -> ProjectVideo:
        item = storage.put_file(key, path, move=move)
        return ProjectVideo(key=item.key, kind=kind, scene=scene, url=storage.url(item.key),
                            duration_seconds=duration, size_bytes=item.size_bytes)

    if backend == "directory":
        source = _prerendered(prerendered_dir, [Path(filename).stem, script.title])
        on_message(f"Found pre-rendered video at {source}")
        on_progress(1.0)
        return [stored(f"{prefix}/final.mp4", source, "final", None, None, move=False)]

    request = _upload_request(script, selected, backend, assets_dir)
    settings = PipelineSettings(generator_backend=backend, storage_dir=render_dir)
    try:
        pipeline = LocalPipeline(store=_ProgressStore(on_progress), settings=settings,
                                 progress_callback=on_message)
    except ValueError as exc:
        raise UserFacingError(str(exc)) from exc
    job = pipeline.submit(request, run_inline=True)
    if job.status != JobStatus.completed:
        raise UserFacingError(job.error or "Video generation failed")
    if any(video.media_type != "video/mp4" for video in job.videos):
        raise UserFacingError("ffmpeg is not installed, so the mock backend could not write MP4s")

    scene_videos = [video for video in job.videos if not video.segment_id.endswith("-combined")]
    combined = next((video for video in job.videos if video.segment_id.endswith("-combined")), None)
    results = []
    for number, video in zip(selected, scene_videos):
        results.append(stored(f"{prefix}/scene-{number:02d}.mp4", Path(video.path), "scene",
                              number, video.duration_seconds, move=True))
    if combined is not None:
        results.insert(0, stored(f"{prefix}/final.mp4", Path(combined.path), "final", None,
                                 combined.duration_seconds, move=True))
    elif results:
        only = storage.local_path(results[0].key)
        if only is not None:
            results.insert(0, stored(f"{prefix}/final.mp4", only, "final", None,
                                     results[0].duration_seconds, move=False))
    return results

