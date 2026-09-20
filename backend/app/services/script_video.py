"""Convert a reviewed short-video script into the existing video job contract."""

from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.models.shorts import ShortPlan


def upload_request_from_script(plan: ShortPlan,
                               scene_numbers: list[int] | None = None) -> UploadRequest:
    by_number = {scene.scene: scene for scene in plan.scenes}
    selected = scene_numbers or sorted(by_number)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one scene and do not repeat scene numbers")
    unknown = [number for number in selected if number not in by_number]
    if unknown:
        raise ValueError(f"Unknown scene numbers: {unknown}")
    segments = []
    for number in selected:
        scene = by_number[number]
        segments.append(SegmentRequest(
            title=f"Scene {number}",
            text=scene.text,
            keywords=scene.keywords,
            visual_prompt=scene.visual_prompt,
            motion=scene.motion,
            narration=scene.narration,
            narration_seconds=scene.narration_seconds,
        ))
    return UploadRequest(title=plan.title, segments=segments)
