import json

import pytest
from pydantic import ValidationError

from backend.app.models.shorts import ShortPlanDraft
from backend.app.services.storyboard_schema import StoryPlanningError
from backend.scripts.qwen_short_story_worker import (
    apply_clothing_transition,
    apply_lifecycle_scene_contract,
    build_messages,
    pad_short_narrations,
    parse_short_plan,
    plan_with_retries,
)

SENTENCES = [
    {"id": 1, "page": 1, "text": "A seed sleeps in dark soil."},
    {"id": 2, "page": 1, "text": "Rain waters the seed."},
    {"id": 3, "page": 1, "text": "A root grows downward."},
    {"id": 4, "page": 1, "text": "A flower blooms."},
]
GROUPS = [[1], [2], [3], [4]]


def _payload() -> dict:
    return {
        "summary": "A seed sleeps, receives rain, grows a root, and a flower blooms.",
        "character": {
            "name": "Seed", "kind": "seed", "visual_identity": "one brown oval seed",
            "age": None, "hair": None, "eyes": None, "clothes": None,
            "outfits": {"default": "unchanged from the reference"},
            "states": {
                "seed": "one brown seed",
                "root": "the same seed with one root",
                "flower": "the same plant as a flower",
            },
            "inferred_fields": ["visual_identity"],
        },
        "scenes": [
            {"action": "The seed sleeps in soil", "location": "underground",
             "weather": "not visible", "camera": "soil cutaway",
             "motion": "Loose soil slowly settles around the seed.",
             "narration": "A seed sleeps quietly beneath the dark soil.",
             "outfit_id": "default", "state_id": "seed"},
            {"action": "Rain waters the seed", "location": "garden soil",
             "weather": "rain", "camera": "close cutaway",
             "motion": "Water sinks through soil toward the seed.",
             "narration": "Gentle rain gives the waiting seed water.",
             "outfit_id": "default", "state_id": "seed"},
            {"action": "A root grows downward", "location": "underground",
             "weather": "not visible", "camera": "macro cutaway",
             "motion": "The root visibly lengthens into the soil.",
             "narration": "A tiny root grows downward through soil.",
             "outfit_id": "default", "state_id": "root"},
            {"action": "A flower blooms", "location": "garden",
             "weather": "sunny", "camera": "medium shot",
             "motion": "The flower opens its petals from bud to bloom.",
             "narration": "The plant finally blooms into a beautiful flower.",
             "outfit_id": "default", "state_id": "flower"},
        ],
    }


def test_accepts_valid_four_scene_plan() -> None:
    draft = plan_with_retries(lambda messages: json.dumps(_payload()),
                              build_messages(SENTENCES, GROUPS), SENTENCES, GROUPS, 3)
    assert isinstance(draft, ShortPlanDraft)
    assert len(draft.scenes) == 4


def test_retries_with_specific_grounding_error() -> None:
    payload = _payload()
    payload["scenes"][2]["action"] = "A bicycle crosses the street"
    answers = iter([json.dumps(payload), json.dumps(_payload())])
    seen = []
    progress = []

    def generate(messages):
        seen.append(messages[-1]["content"])
        return next(answers)

    plan_with_retries(
        generate, build_messages(SENTENCES, GROUPS), SENTENCES, GROUPS, 3,
        progress.append,
    )
    assert "scene 3 action is not grounded" in seen[1]
    assert progress[0] == "generation attempt 1/3 started"
    assert any(message.startswith("validation failed:") for message in progress)
    assert progress[-1] == "script validation passed"


def test_rejects_narration_over_scene_budget() -> None:
    payload = _payload()
    payload["scenes"][0]["narration"] = "one two three four five six seven eight nine ten"
    with pytest.raises(ValidationError, match="expected 6-9"):
        ShortPlanDraft.model_validate(payload)


def test_pads_short_model_narration_before_validation() -> None:
    payload = _payload()
    payload["scenes"][0]["narration"] = "A seed sleeps in soil."
    pad_short_narrations(payload)
    assert payload["scenes"][0]["narration"] == (
        "A seed sleeps in soil now."
    )
    ShortPlanDraft.model_validate(payload)


def test_maps_free_text_lifecycle_state_to_declared_state() -> None:
    payload = _payload()
    payload["scenes"][0]["state_id"] = "dormant"
    draft = parse_short_plan(json.dumps(payload), SENTENCES, GROUPS)
    assert draft.scenes[0].state_id == "seed"


def test_lifecycle_contract_separates_first_and_final_seed_states() -> None:
    payload = {"character": {}, "scenes": [{"camera": "close"} for _ in range(8)]}
    apply_lifecycle_scene_contract(payload)
    assert payload["scenes"][0]["state_id"] == "dormant_seed"
    assert payload["scenes"][6]["state_id"] == "flower_with_bee"
    assert payload["scenes"][7]["state_id"] == "new_seeds"
    assert "entire bee and flower visible" in payload["scenes"][6]["camera"]


def test_gives_up_after_retry_budget() -> None:
    with pytest.raises(StoryPlanningError, match="invalid after 2 attempts"):
        plan_with_retries(lambda messages: "not json", build_messages(SENTENCES, GROUPS),
                          SENTENCES, GROUPS, 2)


def test_prompt_separates_character_from_scene_and_sets_duration_budget() -> None:
    prompt = build_messages(SENTENCES, GROUPS)[1]["content"]
    assert "exactly 4 scenes" in prompt
    assert "6-9 words" in prompt
    assert "character.outfits" in prompt
    assert "never repeat age, hair, eyes, or clothing" in build_messages(
        SENTENCES, GROUPS
    )[0]["content"]


def test_rejects_clothes_repeated_in_stable_visual_identity() -> None:
    payload = _payload()
    payload["character"]["clothes"] = "yellow raincoat and red boots"
    payload["character"]["visual_identity"] = (
        "one brown seed wearing a yellow raincoat and red boots"
    )
    with pytest.raises(ValueError, match="must not repeat clothing"):
        parse_short_plan(json.dumps(payload), SENTENCES, GROUPS)


def test_normalizes_freeform_state_ids_for_ordinary_story() -> None:
    payload = _payload()
    payload["summary"] = "A seed sleeps, receives rain, grows a root, and a blossom blooms."
    payload["scenes"][3]["action"] = "A blossom blooms"
    payload["scenes"][3]["narration"] = (
        "The plant finally blooms into a beautiful blossom."
    )
    payload["character"]["states"] = {"walking": "walking through rain"}
    for scene in payload["scenes"]:
        scene["state_id"] = "walking in rain"
    sentences = [
        {"id": 1, "text": "A seed sleeps in dark soil."},
        {"id": 2, "text": "Rain waters the seed."},
        {"id": 3, "text": "A root grows downward."},
        {"id": 4, "text": "A blossom blooms."},
    ]
    draft = parse_short_plan(json.dumps(payload), sentences, GROUPS)
    assert draft.character.states == {
        "default": "same physical identity throughout",
    }
    assert {scene.state_id for scene in draft.scenes} == {"default"}


def test_source_clothing_change_derives_before_and_after_outfits() -> None:
    payload = _payload()
    payload["character"].update({
        "name": "Mia", "kind": "human child", "clothes": "yellow raincoat and red boots",
        "outfits": {"default": "yellow raincoat and red boots"},
    })
    for scene in payload["scenes"]:
        scene["outfit_id"] = "default"
    draft = ShortPlanDraft.model_validate(payload)
    sentences = [
        {"id": 1, "text": "Mia watches dark clouds."},
        {"id": 2, "text": "Mia put on her yellow raincoat and red boots."},
        {"id": 3, "text": "Mia jumps into a puddle."},
        {"id": 4, "text": "Mia sees a rainbow."},
    ]
    apply_clothing_transition(draft, sentences, [[1], [2], [3], [4]])
    assert list(draft.character.outfits) == ["before_change", "after_change"]
    assert [scene.outfit_id for scene in draft.scenes] == [
        "before_change", "after_change", "after_change", "after_change"
    ]
    assert draft.character.clothes == "yellow raincoat and red boots"
    assert draft.character.outfits["after_change"] == "yellow raincoat and red boots"
