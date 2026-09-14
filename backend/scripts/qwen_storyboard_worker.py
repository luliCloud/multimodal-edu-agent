"""One-shot GPU planner process; exiting releases VRAM before video inference."""

import json
import os
import sys


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_id = int(sys.argv[1])
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen planner needs a visible CUDA GPU")
    request = json.load(sys.stdin)
    groups = request["scene_groups"]
    target_scenes = len(groups)
    model_id = os.getenv("PLANNER_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.to(f"cuda:{gpu_id}").eval()
    by_id = {item["id"]: item for item in request["sentences"]}
    source = "\n".join(
        f'Scene {index} (source IDs {", ".join(map(str, group))}): '
        + " ".join(by_id[item]["text"] for item in group)
        for index, group in enumerate(groups, 1)
    )
    messages = [
        {"role": "system", "content": (
            "You are a precise storyboard editor. Use only the provided source text. "
            "Return only a valid JSON object, no Markdown. Do not invent story events, "
            "characters, colors or objects unsupported by the source. Prefer one visible "
            "action per 4-second scene. Keep chronology. Dialogue is not an image. "
            "Make temporal motion explicit and keep the main character visually consistent. "
            "The supplied scene groups already fix story coverage and order."
        )},
        {"role": "user", "content": (
            "Summarize this story in one grammatical English sentence with its main "
            "character, important actions, and outcome. Return exactly "
            f"{target_scenes} scenes in the order listed below: one scene for "
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
            "JSON schema: "
            '{"summary":"...","scenes":[{"visual_prompt":"...",'
            '"motion":"..."}]}\n\nSource scene groups:\n' + source
        )},
    ]
    template_options = {"enable_thinking": False} if model_id == "Qwen/Qwen3-1.7B" else {}
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, **template_options)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1100, do_sample=False)
    answer = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:],
                              skip_special_tokens=True).strip()
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    print(json.dumps(json.loads(answer), ensure_ascii=False))


if __name__ == "__main__":
    main()
