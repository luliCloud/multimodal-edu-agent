from backend.app.models.jobs import SegmentRequest
from backend.app.services.wan_video import WanVideoGenerator


def test_prompt_uses_scene_and_visual_keywords() -> None:
    prompt = WanVideoGenerator.build_prompt(
        SegmentRequest(text="A tiny root grows from the seed.", keywords=["seed", "root", "soil"])
    )
    assert "single thin pale root" in prompt
    assert "seed, root, soil" in prompt
    assert "watercolor children's storybook animation" in prompt
    assert "on-screen text" not in prompt


def test_prompt_omits_spoken_dialogue() -> None:
    prompt = WanVideoGenerator.build_prompt(
        SegmentRequest(text='A bee visits the flower. “Buzz, buzz!” said the bee. The flower sways.')
    )
    assert "bee visits" in prompt
    assert "flower sways" in prompt
    assert "Buzz" not in prompt


def test_prompt_makes_rain_and_bee_motion_explicit() -> None:
    rain = WanVideoGenerator.build_prompt(SegmentRequest(
        text="The rain fell softly on the ground. The little seed drank water.",
        visual_prompt="A small brown seed lies in dark soil under rain.",
        motion="Raindrops fall and the soil darkens.",
        keywords=["warm sun", "rain", "seed"],
    ))
    assert "Raindrops fall and the soil darkens" in rain
    assert "translucent pale blue raindrops" in rain
    assert "warm sun" not in rain
    assert "No sprout or plant" in rain

    bee = WanVideoGenerator.build_prompt(SegmentRequest(
        text="A bee flew over to visit the yellow flower.",
        visual_prompt="A bee approaches one yellow flower.",
        motion="The bee flies toward the flower.",
    ))
    assert "head facing the flower" in bee
    assert "abdomen trailing behind" in bee


def test_seed_before_growth_excludes_early_plant() -> None:
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="A little seed slept in the dark soil.", keywords=["seed", "dark soil"],
    ))
    assert "before germination" in prompt
    assert "no sprout, root, stem, or plant" in prompt


def test_root_scene_uses_underground_cutaway() -> None:
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="Soon, a tiny root grew down into the soil.", keywords=["root", "soil"],
    ))
    assert "cutaway cross-section" in prompt
    assert "single thin pale root" in prompt
    assert "no trees, leaves, stems" in prompt


def test_leaf_scene_requires_attachment() -> None:
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="It had two small leaves. Then it had four leaves.",
    ))
    assert "four leaves are visibly attached" in prompt
    assert "No flowers or floating leaves" in prompt


def test_new_seed_ending_shows_distinct_seeds() -> None:
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="And inside the flower were new seeds, ready to begin again.",
    ))
    assert "separate small brown oval new seeds" in prompt
    assert "empty dark disk" in prompt
