"""GPU worker for a validated four-scene short-video plan."""

import json
import os
import re
import sys
from collections.abc import Callable

from backend.app.models.shorts import ShortPlanDraft
from backend.app.services.shorts_planner import STABLE_DEFAULT_STATE, grounding_words
from backend.app.services.storyboard_schema import (
    StoryPlanningError,
    corrective_feedback,
    strip_code_fence,
)

DEFAULT_MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "You design concise educational 15-second vertical story videos. "
    "Use only the supplied source events. Return one valid JSON object and no Markdown. "
    "Keep one global character identity and style across all scenes. Scene fields describe "
    "only what happens in that shot; never repeat age, hair, eyes, or clothing there."
)


def build_source_block(sentences: list[dict], groups: list[list[int]]) -> str:
    by_id = {item["id"]: item for item in sentences}
    return "\n".join(
        f'Scene {index} (source IDs {", ".join(map(str, group))}): '
        + " ".join(by_id[item]["text"] for item in group)
        for index, group in enumerate(groups, 1)
    )


def build_messages(sentences: list[dict], groups: list[list[int]]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Create exactly four scenes, one for each source group and in the same order. "
            "Cover the most important visible action in every group, including the ending. "
            "Write one 8-30 word grammatical summary covering the beginning, important "
            "causes or changes, and outcome. The summary must use a meaningful source word "
            "from every group. Choose one recurring main character. Put stable appearance "
            "only in character.visual_identity and the character fields. visual_identity "
            "must describe physical identity only and must not contain clothes. For a human, fill "
            "age, hair, eyes, and clothes; use null for nonhuman fields that do not apply. "
            "Keep character.kind faithful to the source: a seed, plant, animal, or object "
            "must never be labeled human. Source text may omit appearance; choose a "
            "consistent design once and list each "
            "invented field in inferred_fields. Preserve explicitly stated colors and clothes. "
            "When the character changes clothes, define named variants in character.outfits "
            "and use the correct outfit_id before and after the change. Otherwise define one "
            "default outfit; nonhuman characters without clothes use 'not applicable'. "
            "When the subject visibly transforms, define every stable lifecycle form once in "
            "character.states and select the correct state_id in each scene. Ordinary stories "
            "use one default state. Do not invent story events, props, weather, or appearance for "
            "secondary characters. Each action, narration, and the summary must use at least "
            "two meaningful words from every corresponding source group. Motion must visibly "
            "animate that group's action from start to end. "
            "Narration must be one natural sentence of 6-9 words. No written dialogue. If rain "
            "stops, later scenes must not show falling rain. "
            'Schema: {"summary":"...","character":{"name":"...","kind":"...",'
            '"visual_identity":"...","age":null,"hair":null,"eyes":null,"clothes":null,'
            '"outfits":{"default":"..."},"states":{"default":"..."},'
            '"inferred_fields":[]},"scenes":['
            '{"action":"...","location":"...","weather":"...","camera":"...",'
            '"motion":"...","narration":"...","outfit_id":"default",'
            '"state_id":"default"}]}\n\n'
            "Source scene groups:\n" + build_source_block(sentences, groups)
        )},
    ]


def pad_short_narrations(payload: dict) -> None:
    """Keep otherwise valid model narration inside the spoken-word budget."""
    for scene in payload.get("scenes", []):
        narration = str(scene.get("narration", "")).strip()
        if 3 <= len(narration.split()) < 6:
            scene["narration"] = narration.rstrip(".!?") + " at that moment."


def apply_clothing_transition(draft: ShortPlanDraft, sentences: list[dict],
                              groups: list[list[int]]) -> None:
    """Derive before/after outfit IDs when the source explicitly changes clothes."""
    by_id = {item["id"]: item["text"] for item in sentences}
    group_texts = [" ".join(by_id[item] for item in group).lower() for group in groups]
    change_index = next((index for index, text in enumerate(group_texts)
                         if re.search(r"\b(put on|puts on|change clothes|changes clothes|"
                                      r"removed|removes)\b", text)), None)
    if change_index is None:
        return
    clothing_match = re.search(
        r"\bput(?:s)? on\s+(?:(?:his|her|their)\s+)?(.+?)(?=[.!?]|$)",
        group_texts[change_index],
    )
    source_clothing = clothing_match.group(1).strip(" ,") if clothing_match else None
    declared_clothing = draft.character.clothes
    if declared_clothing and declared_clothing.lower() in {
        "not applicable", "none", "unchanged from the reference",
    }:
        declared_clothing = None
    after = (source_clothing or declared_clothing or
             next(iter(draft.character.outfits.values()), "source-described outerwear"))
    if source_clothing:
        draft.character.clothes = source_clothing
    draft.character.outfits = {
        "before_change": f"simple base clothes without {after}",
        "after_change": after,
    }
    for index, scene in enumerate(draft.scenes):
        scene.outfit_id = "before_change" if index < change_index else "after_change"


def parse_short_plan(text: str, sentences: list[dict],
                     groups: list[list[int]]) -> ShortPlanDraft:
    try:
        payload = json.loads(strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"response was not valid JSON: {exc}") from exc
    pad_short_narrations(payload)
    source_words = grounding_words(" ".join(item["text"] for item in sentences))
    lifecycle_words = {"seed", "root", "stem", "leaf", "leave", "flower"}.intersection(
        source_words
    )
    if len(lifecycle_words) < 3:
        payload.setdefault("character", {})["states"] = {
            "default": STABLE_DEFAULT_STATE,
        }
        for scene in payload.get("scenes", []):
            scene["state_id"] = "default"

    draft = ShortPlanDraft.model_validate(payload)
    if len(draft.scenes) != len(groups):
        raise ValueError(f"must return exactly {len(groups)} scenes")

    by_id = {item["id"]: item["text"] for item in sentences}
    all_source_text = " ".join(by_id.values()).lower()
    all_source_words = grounding_words(all_source_text)
    kind_words = grounding_words(draft.character.kind)
    botanical = {"seed", "plant", "flower", "tree"}.intersection(all_source_words)
    human_cues = {"girl", "boy", "child", "woman", "man", "person", "dad", "father",
                  "mom", "mother"}.intersection(all_source_words)
    human_kind = {"human", "person", "child"}.intersection(kind_words)
    named_in_source = draft.character.name.lower() in all_source_text
    if (botanical and not human_cues and
            human_kind):
        raise ValueError(
            "character.kind cannot be human for a botanical story with no human character"
        )
    if not (human_kind and (human_cues or named_in_source)):
        if kind_words and not kind_words.intersection(all_source_words):
            raise ValueError(
                f"character.kind {sorted(kind_words)} is unsupported by source words"
            )
    clothes_words = grounding_words(str(draft.character.clothes or ""))
    identity_words = grounding_words(draft.character.visual_identity)
    if clothes_words and len(clothes_words.intersection(identity_words)) >= min(
            2, len(clothes_words)):
        raise ValueError("character.visual_identity must not repeat clothing details")
    if len(lifecycle_words) >= 3:
        if len(draft.character.states) < 2 or len({scene.state_id for scene in draft.scenes}) < 2:
            raise ValueError(
                "a lifecycle story must define and use multiple character.states"
            )
    apply_clothing_transition(draft, sentences, groups)
    name_words = grounding_words(draft.character.name)
    summary_words = grounding_words(draft.summary)
    for index, (group, scene) in enumerate(zip(groups, draft.scenes), 1):
        source_words = grounding_words(" ".join(by_id[item] for item in group))
        action_words = grounding_words(scene.action) - name_words
        narration_words = grounding_words(scene.narration) - name_words
        required = min(2, len(source_words - name_words))
        action_overlap = action_words.intersection(source_words)
        narration_overlap = narration_words.intersection(source_words)
        summary_overlap = summary_words.intersection(source_words)
        if len(action_overlap) < required:
            raise ValueError(
                f"scene {index} action is not grounded: action words {sorted(action_words)} "
                f"must include {required} source words from {sorted(source_words)}"
            )
        if len(narration_overlap) < required:
            raise ValueError(
                f"scene {index} narration is not grounded: narration words "
                f"{sorted(narration_words)} must include {required} source words from "
                f"{sorted(source_words)}"
            )
        if len(summary_overlap) < required:
            raise ValueError(
                f"summary does not cover source group {index}: it must include "
                f"{required} words from {sorted(source_words)}"
            )
    return draft


def plan_with_retries(generate: Callable[[list[dict]], str], messages: list[dict],
                      sentences: list[dict], groups: list[list[int]],
                      max_attempts: int) -> ShortPlanDraft:
    last_error: Exception | None = None
    for _ in range(max_attempts):
        answer = generate(messages)
        try:
            return parse_short_plan(answer, sentences, groups)
        except ValueError as error:
            last_error = error
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": corrective_feedback(error)})
    raise StoryPlanningError(
        f"Short storyboard invalid after {max_attempts} attempts: {last_error}"
    )


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_id = int(sys.argv[1])
    if not torch.cuda.is_available():
        raise RuntimeError("Short planner needs a visible CUDA GPU")
    request = json.load(sys.stdin)
    sentences = request["sentences"]
    groups = request["scene_groups"]
    model_id = os.getenv("PLANNER_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    max_attempts = int(os.getenv("PLANNER_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.to(f"cuda:{gpu_id}").eval()
    template_options = {"enable_thinking": False} if model_id == "Qwen/Qwen3-1.7B" else {}

    def generate(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, **template_options)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
        return tokenizer.decode(output[0][inputs.input_ids.shape[-1]:],
                                skip_special_tokens=True).strip()

    try:
        draft = plan_with_retries(
            generate, build_messages(sentences, groups), sentences, groups, max_attempts
        )
    except StoryPlanningError as error:
        raise SystemExit(str(error)) from error
    print(draft.model_dump_json())


if __name__ == "__main__":
    main()
