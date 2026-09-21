"""Four-scene plan with global character/style and grounded scene evidence."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from backend.app.core.config import get_settings
from backend.app.models.shorts import (
    CharacterSpec,
    ShortPlan,
    ShortPlanDraft,
    ShortScene,
    StyleSpec,
)
from backend.app.services.gpu_scheduler import scheduler
from backend.app.services.pdf_keywords import extract_video_keywords, extract_video_scenes
from backend.app.services.storyboard_schema import (
    ScopeExceededError,
    StoryPlanningError,
    narration_seconds,
)

MAX_SOURCE_SENTENCES = 32
REPO_ROOT = Path(__file__).resolve().parents[3]
STABLE_DEFAULT_STATE = "same physical identity throughout"
GROUNDING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    "he", "she", "they", "him", "her", "them", "his", "their", "its", "one", "two",
    "three", "four", "little", "small", "tiny", "then", "soon", "again", "said",
}


def source_sentences(pages: list[str]) -> list[dict]:
    return [{"id": index, "page": scene["pages"][0], "text": scene["text"]}
            for index, scene in enumerate(extract_video_scenes(pages, limit=200), 1)]


def four_scene_groups(sentences: list[dict]) -> list[list[int]]:
    count = len(sentences)
    if count < 4:
        raise ScopeExceededError("A 4-scene short needs at least 4 narrative sentences")
    if count > MAX_SOURCE_SENTENCES:
        raise ScopeExceededError(
            f"Storyboard supports up to {MAX_SOURCE_SENTENCES} narrative sentences; "
            "split longer PDFs"
        )
    return [[item["id"] for item in sentences[index * count // 4:(index + 1) * count // 4]]
            for index in range(4)]


def story_scene_groups(sentences: list[dict]) -> list[list[int]]:
    """Use semantic lifecycle cuts when several visible transformations must stay separate."""
    text = [item["text"].lower() for item in sentences]

    def first(pattern: str) -> int | None:
        return next((index for index, sentence in enumerate(text)
                     if re.search(pattern, sentence)), None)

    anchors = [
        first(r"\b(?:sun|sunlight|rain|water)\b"),
        first(r"\broots?\b"),
        first(r"\bstems?\b"),
        first(r"\b(?:leaf|leaves)\b"),
        first(r"\bflowers?\b"),
        first(r"\bbees?\b"),
        first(r"\b(?:not a seed anymore|new seeds?)\b"),
    ]
    if (all(anchor is not None for anchor in anchors) and
            anchors == sorted(set(anchors)) and anchors[0] > 0):
        boundaries = [0, *anchors, len(sentences)]
        groups = [
            [item["id"] for item in sentences[start:end]]
            for start, end in zip(boundaries, boundaries[1:])
        ]
        if len(groups) == 8 and all(groups):
            return groups
    return four_scene_groups(sentences)


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    settings = get_settings()
    env["PLANNER_MODEL_ID"] = settings.planner_model_id
    env["PLANNER_MAX_ATTEMPTS"] = str(settings.planner_max_attempts)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return env


def grounding_words(text: str) -> set[str]:
    def normalize(word: str) -> str:
        irregular = {
            "has": "have", "had": "have", "gave": "give",
            "grew": "grow", "grown": "grow", "flew": "fly",
            "drank": "drink", "slept": "sleep", "bore": "bear",
            "downward": "down", "upward": "up",
        }
        if word in irregular:
            return irregular[word]
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("ing") and len(word) > 5:
            return word[:-3]
        if word.endswith("ed") and len(word) > 4:
            stem = word[:-2]
            return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
        if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
            return word[:-1]
        return word

    return {normalize(word) for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) >= 3 and word not in GROUNDING_STOPWORDS}


def _mark_inferred_character_fields(character: dict, source_text: str) -> dict:
    character = dict(character)
    source_words = grounding_words(source_text)
    inferred = set()
    for field in ("visual_identity", "age", "hair", "eyes", "clothes"):
        value = character.get(field)
        if value is not None:
            value_words = grounding_words(str(value))
            if not value_words or not value_words.issubset(source_words):
                inferred.add(field)
    outfit_words = grounding_words(" ".join(character.get("outfits", {}).values()))
    outfit_values = {value.lower() for value in character.get("outfits", {}).values()}
    if (outfit_words and outfit_values != {"not applicable"} and
            not outfit_words.issubset(source_words)):
        inferred.add("outfits")
    state_words = grounding_words(" ".join(character.get("states", {}).values()))
    default_identity_state = character.get("states") in (
        {"default": character.get("visual_identity")},
        {"default": STABLE_DEFAULT_STATE},
    )
    if state_words and not default_identity_state and not state_words.issubset(source_words):
        inferred.add("states")
    character["inferred_fields"] = sorted(inferred)
    return character


def _normalize_character_for_source(character: dict, source_text: str) -> dict:
    character = dict(character)
    kind_words = grounding_words(str(character.get("kind", "")))
    if not {"human", "person", "child"}.intersection(kind_words) and not character.get("clothes"):
        character["outfits"] = {"default": "not applicable"}
    lifecycle = {"seed", "root", "stem", "leaf", "leave", "flower"}.intersection(
        grounding_words(source_text)
    )
    if len(lifecycle) < 3:
        character["states"] = {"default": STABLE_DEFAULT_STATE}
    return character


def plan_short_story(pages: list[str], title: str) -> ShortPlan:
    sentences = source_sentences(pages)
    groups = story_scene_groups(sentences)

    def run_worker(gpu_id: int) -> dict:
        print(
            f"[planner] launching Qwen worker on GPU {gpu_id} for "
            f"{len(sentences)} source sentences",
            file=sys.stderr,
            flush=True,
        )
        result = subprocess.run(
            [sys.executable, "-m", "backend.scripts.qwen_short_story_worker", str(gpu_id)],
            input=json.dumps({"sentences": sentences, "scene_groups": groups}),
            text=True, stdout=subprocess.PIPE, timeout=600, check=False,
            env=_worker_env(),
        )
        if result.returncode:
            detail = f"worker exited with code {result.returncode}"
            try:
                error_payload = json.loads(result.stdout)
                detail = error_payload.get("planner_error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise StoryPlanningError(f"Short planner failed: {detail}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise StoryPlanningError(f"Short planner returned invalid JSON: {exc}") from exc

    raw = scheduler.run_on_gpu(run_worker)
    if draft_path := os.getenv("SHORT_PLANNER_DRAFT_PATH"):
        Path(draft_path).write_text(json.dumps(raw, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    try:
        draft = ShortPlanDraft.model_validate(raw)
    except ValidationError as exc:
        raise StoryPlanningError(str(exc)) from exc

    by_id = {item["id"]: item for item in sentences}
    name_words = grounding_words(draft.character.name)
    scenes = []
    for index, (ids, raw_scene) in enumerate(zip(groups, draft.scenes), 1):
        evidence = [by_id[item] for item in ids]
        text = " ".join(item["text"] for item in evidence)
        action_anchors = grounding_words(raw_scene.action) - name_words
        if not action_anchors.intersection(grounding_words(text)):
            raise StoryPlanningError(f"Scene {index} action is not grounded in its source group")
        scenes.append(ShortScene(
            **raw_scene.model_dump(),
            scene=index,
            source_sentence_ids=ids,
            text=text,
            pages=sorted({item["page"] for item in evidence}),
            keywords=[item["keyword"] for item in extract_video_keywords([text], limit=5)],
            narration_seconds=narration_seconds(raw_scene.narration),
        ))

    source_text = " ".join(item["text"] for item in sentences)
    plan = ShortPlan(
        title=title,
        summary=draft.summary,
        character=CharacterSpec.model_validate(
            _mark_inferred_character_fields(
                _normalize_character_for_source(draft.character.model_dump(), source_text),
                source_text,
            )
        ),
        style=StyleSpec(),
        scenes=scenes,
        source_sentences=sentences,
    )
    for scene in plan.scenes:
        scene.visual_prompt = scene_visual_prompt(plan, scene)
        scene.keyframe_prompt = scene_keyframe_prompt(plan, scene)
    return plan


def character_reference_prompt(plan: ShortPlan) -> str:
    character = plan.character
    style = plan.style
    appearance = ", ".join(f"{label} {value}" for label, value in (
        ("age", character.age), ("hair", character.hair), ("eyes", character.eyes))
        if value is not None) or character.visual_identity
    outfits = "; ".join(f"{name}: {clothes}" for name, clothes in character.outfits.items())
    states = "; ".join(f"{name}: {description}"
                       for name, description in character.states.items())
    return (
        f"GLOBAL_CHARACTER: {character.name}; {character.kind}; {appearance}. "
        f"OUTFITS: {outfits}. "
        f"CHARACTER_STATES: {states}. "
        f"GLOBAL_STYLE: {style.type}; {style.palette} palette; {style.shapes} shapes; "
        f"{style.lighting} lighting; {style.outline} outlines. "
        "Create one full-body character reference sheet with front and side views for each "
        "listed state, on a plain light background. Keep identity consistent across states. "
        "No text, labels, logos, or other characters. Portrait composition."
    )


def scene_keyframe_prompt(plan: ShortPlan, scene: ShortScene) -> str:
    return (
        scene_visual_prompt(plan, scene)
        + " Use the attached character reference as the exact same character."
    )


def scene_visual_prompt(plan: ShortPlan, scene: ShortScene) -> str:
    character = plan.character
    style = plan.style
    appearance = ", ".join(f"{label} {value}" for label, value in (
        ("age", character.age), ("hair", character.hair), ("eyes", character.eyes))
        if value is not None) or character.visual_identity
    outfit = character.outfits[scene.outfit_id]
    state = character.states[scene.state_id]
    return (
        f"GLOBAL_CHARACTER: {character.name}; {character.kind}; {appearance}; "
        f"outfit {scene.outfit_id}: {outfit}. "
        f"character state {scene.state_id}: {state}. "
        f"GLOBAL_STYLE: {style.type}; {style.palette} palette; {style.shapes} shapes; "
        f"{style.lighting} lighting; {style.outline} outlines. "
        f"SCENE_DESCRIPTION: Action: {scene.action}. Location: {scene.location}. "
        f"Weather: {scene.weather}. Camera: {scene.camera}. "
        "Single vertical 9:16 keyframe, full scene, no panels. "
        "No text, captions, logo, watermark, duplicate people, or extra limbs."
    )
