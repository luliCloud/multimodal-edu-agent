"""One-shot GPU planner process; exiting releases VRAM before video inference."""

import os
import sys
from typing import Callable

from backend.app.services.storyboard_schema import (
    TARGET_NARRATION_WORDS,
    StoryboardDraft,
    StoryPlanningError,
    corrective_feedback,
    parse_storyboard,
)

# A 4B instruct model drops a required field often enough that one retry is
# not enough; retries happen here, where the model is still loaded.
DEFAULT_MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "You are a precise storyboard editor. Use only the provided source text. "
    "Return only a valid JSON object, no Markdown. Do not invent story events, "
    "characters, colors or objects unsupported by the source. Prefer one visible "
    "action per 4-second scene. Keep chronology. Dialogue is not an image. "
    "Make temporal motion explicit and keep the main character visually consistent. "
    "The supplied scene groups already fix story coverage and order."
)


def build_source_block(sentences: list[dict], groups: list[list[int]]) -> str:
    by_id = {item["id"]: item for item in sentences}
    return "\n".join(
        f'Scene {index} (source IDs {", ".join(map(str, group))}): '
        + " ".join(by_id[item]["text"] for item in group)
        for index, group in enumerate(groups, 1)
    )


def build_messages(sentences: list[dict], groups: list[list[int]]) -> list[dict]:
    low, high = TARGET_NARRATION_WORDS
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Summarize this story in one grammatical English sentence with its main "
            "character, important actions, and outcome. Return exactly "
            f"{len(groups)} scenes in the order listed below: one scene for "
            "each supplied Scene group, with no skipped or repeated groups. "
            "For each group, make a visual_prompt using its actual subjects, "
            "setting, clothing colors, objects and weather; use nearby groups "
            "only to maintain visual continuity. Make its motion describe a "
            "visible start-to-end change grounded in that group's action. "
            "If a group includes dialogue or sound effects, show the supported "
            "visible action instead of writing or speaking the words. "
            "A resting subject can use a slow camera push-in. "
            "Rain should appear only in groups where it is still falling; "
            "after the rain stops, show clearing clouds, sun or rainbow as supported. "
            "Do not force a familiar story pattern onto a different story. "
            f"Also write narration of {low} to {high} words for each group: one "
            "spoken sentence a narrator reads over that shot, retelling only that "
            "group's action in natural words. Narration is prose, not a shot "
            "description, and must not name scenes, cameras or the storyboard. "
            "JSON schema: "
            '{"summary":"...","scenes":[{"visual_prompt":"...",'
            '"motion":"...","narration":"..."}]}\n\nSource scene groups:\n'
            + build_source_block(sentences, groups)
        )},
    ]


def plan_with_retries(generate: Callable[[list[dict]], str], messages: list[dict],
                      scene_count: int, max_attempts: int) -> StoryboardDraft:
    """Re-prompt the loaded model with its own validation errors until it complies."""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        answer = generate(messages)
        try:
            return parse_storyboard(answer, scene_count)
        except ValueError as error:
            last_error = error
            # Greedy decoding repeats itself, so the correction has to enter the
            # prompt for the next attempt to differ from this one.
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": corrective_feedback(error)})
    raise StoryPlanningError(f"Storyboard invalid after {max_attempts} attempts: {last_error}")


def main() -> None:
    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_id = int(sys.argv[1])
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen planner needs a visible CUDA GPU")
    request = json.load(sys.stdin)
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

    messages = build_messages(request["sentences"], groups)
    try:
        storyboard = plan_with_retries(generate, messages, len(groups), max_attempts)
    except StoryPlanningError as error:
        raise SystemExit(str(error)) from error
    print(storyboard.model_dump_json())


if __name__ == "__main__":
    main()
