"""Convert a reviewed short-video script into the existing video job contract."""

import hashlib
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.models.shorts import ShortPlan
from backend.app.services.shorts_planner import character_reference_prompt


def upload_request_from_script(plan: ShortPlan,
                               scene_numbers: list[int] | None = None,
                               character_reference_path: Path | None = None,
                               keyframe_dir: Path | None = None) -> UploadRequest:
    by_number = {scene.scene: scene for scene in plan.scenes}
    selected = scene_numbers or sorted(by_number)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one scene and do not repeat scene numbers")
    unknown = [number for number in selected if number not in by_number]
    if unknown:
        raise ValueError(f"Unknown scene numbers: {unknown}")
    segments = []
    is_human = any(word in plan.character.kind.lower()
                   for word in ("human", "person", "child", "girl", "boy"))
    reference_prompt = character_reference_prompt(plan) if is_human else None
    reference_id = None
    reference_image = None
    if reference_prompt and character_reference_path is not None:
        reference_id = "character-" + hashlib.sha256(
            plan.character.model_dump_json().encode("utf-8")
        ).hexdigest()[:12]
        reference_image = str(character_reference_path.resolve())
        keyframe_dir = keyframe_dir or character_reference_path.parent
    for number in selected:
        scene = by_number[number]
        keyframe = keyframe_dir / f"scene_{number:02d}_keyframe.png" if keyframe_dir else None
        segments.append(SegmentRequest(
            title=f"Scene {number}",
            text=scene.text,
            keywords=scene.keywords,
            visual_prompt=scene.visual_prompt,
            motion=scene.motion,
            narration=scene.narration,
            narration_seconds=scene.narration_seconds,
            reference_id=reference_id,
            reference_image=reference_image,
            reference_prompt=reference_prompt,
            keyframe_image=str(keyframe.resolve()) if keyframe and keyframe.is_file() else None,
        ))
    return UploadRequest(title=plan.title, segments=segments)
