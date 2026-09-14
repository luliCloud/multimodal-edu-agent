"""One-shot GPU planner process; exiting releases VRAM before video inference."""

import json
import os
import sys
from math import ceil


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_id = int(sys.argv[1])
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen planner needs a visible CUDA GPU")
    request = json.load(sys.stdin)
    target_scenes = min(8, max(3, ceil(len(request["sentences"]) / 2.5)))
    model_id = os.getenv("PLANNER_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.to(f"cuda:{gpu_id}").eval()
    source = "\n".join(f'{item["id"]} (page {item["page"]}): {item["text"]}'
                       for item in request["sentences"])
    messages = [
        {"role": "system", "content": (
            "You are a precise storyboard editor. Use only the numbered source sentences. "
            "Return only a valid JSON object, no Markdown. Do not invent story events, "
            "characters, colors or objects unsupported by the source. Prefer one visible "
            "action per 4-second scene. Keep chronology. Dialogue is not an image. "
            "Make temporal motion explicit and keep the main character visually consistent. "
            "Source sentence IDs must increase across scenes and must never repeat."
        )},
        {"role": "user", "content": (
            "Summarize this story in one grammatical English sentence with a clear "
            "cause-to-growth-to-result sequence. If the source mentions them, "
            "the summary MUST explicitly include water, sun or sunlight, root, leaves and flower. "
            f"Then plan exactly {target_scenes} short animated scenes covering seed, "
            "rain/water, root, stem/leaves, flower, bee and new seeds if present. "
            "If the source ends with new seeds, the final scene MUST use that ending sentence. "
            "Combine related rain-falling and seed-drinking sentences into one water scene "
            "to leave room for the ending; do not omit the ending. "
            "Use separate scenes for root growth, stem/leaf growth, flower opening, "
            "and bee arrival whenever the source supports those actions. "
            "Every scene must visibly change over time. For a resting subject, use "
            "a slow camera push-in instead of saying it remains still. "
            "For each scene return source_sentence_ids (1 to 4 exact numbered sentences), "
            "in increasing order, with no repeated IDs across scenes. "
            "visual_prompt (concrete visible subjects and setting), and motion (one clear "
            "start-to-end action with direction where relevant). Show rain as distinct "
            "falling droplets if rain occurs. If a bee visits, show the already-open "
            "flower and a side-profile bee moving left to right toward it; the bee's head "
            "must face the flower and its abdomen trail behind. Avoid extra objects. JSON schema: "
            '{"summary":"...","scenes":[{"source_sentence_ids":[1],'
            '"visual_prompt":"...","motion":"..."}]}\n\nSource:\n' + source
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
