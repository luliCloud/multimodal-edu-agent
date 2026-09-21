from backend.app.models.jobs import SegmentRequest
from backend.app.services.wan_video import WanVideoGenerator
from backend.app.services.keyframe_motion import CudaKeyframeMotionGenerator
from backend.app.services.simple_character_rig import SimpleCharacterRig


def test_wan_defaults_to_portrait_output(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WAN_WIDTH", raising=False)
    monkeypatch.delenv("WAN_HEIGHT", raising=False)
    generator = WanVideoGenerator(tmp_path)
    assert (generator.width, generator.height) == (320, 576)
    assert generator.fps == 9
    generator.fit_scene_count(8)
    assert generator.num_frames == 17
    assert generator._controlled.num_frames == 17


def test_lifecycle_reference_routes_every_stage_to_controlled_motion(monkeypatch) -> None:
    monkeypatch.delenv("WAN_CONTROLLED_MOTION", raising=False)
    prefix = "GLOBAL_CHARACTER: Seedling; plant; same physical identity. "
    for text in (
        "A little seed slept in soil.",
        "Rain fell on the seed.",
        "A yellow flower appeared.",
        "A bee flew over to visit.",
        "New seeds formed inside the flower.",
    ):
        assert WanVideoGenerator.uses_controlled_motion(
            SegmentRequest(text=text, visual_prompt=prefix)
        )
    assert not WanVideoGenerator.uses_controlled_motion(
        SegmentRequest(text="A rainbow appeared over Mia.")
    )


def test_prompt_uses_scene_and_visual_keywords(monkeypatch) -> None:
    monkeypatch.setenv("WAN_CONTROLLED_MOTION", "1")
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


def test_prompt_makes_rain_and_bee_motion_explicit(monkeypatch) -> None:
    monkeypatch.setenv("WAN_CONTROLLED_MOTION", "1")
    rain = WanVideoGenerator.build_prompt(SegmentRequest(
        text="The rain fell softly on the ground. The little seed drank water.",
        visual_prompt="A small brown seed lies in dark soil under rain.",
        motion="Raindrops fall and the soil darkens.",
        keywords=["warm sun", "rain", "seed"],
    ))
    assert "Raindrops fall and the soil darkens" in rain
    assert "warm sun" not in rain
    assert "No sprout or plant" in rain

    bee = WanVideoGenerator.build_prompt(SegmentRequest(
        text="A bee flew over to visit the yellow flower.",
        visual_prompt="A bee approaches one yellow flower.",
        motion="The bee flies toward the flower.",
    ))
    assert "head facing the flower" in bee
    assert "abdomen trailing behind" in bee


def test_seed_before_growth_excludes_early_plant(monkeypatch) -> None:
    monkeypatch.setenv("WAN_CONTROLLED_MOTION", "1")
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="A little seed slept in the dark soil.", keywords=["seed", "dark soil"],
    ))
    assert "before germination" in prompt
    assert "no sprout, root, stem, or plant" in prompt


def test_root_scene_uses_underground_cutaway(monkeypatch) -> None:
    monkeypatch.setenv("WAN_CONTROLLED_MOTION", "1")
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="Soon, a tiny root grew down into the soil.", keywords=["root", "soil"],
    ))
    assert "cutaway cross-section" in prompt
    assert "single thin pale root" in prompt
    assert "no trees, leaves, stems" in prompt


def test_leaf_scene_requires_attachment(monkeypatch) -> None:
    monkeypatch.setenv("WAN_CONTROLLED_MOTION", "1")
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="It had two small leaves. Then it had four leaves.",
    ))
    assert "four leaves are visibly attached" in prompt
    assert "No flowers or floating leaves" in prompt


def test_new_seed_ending_shows_distinct_seeds(monkeypatch) -> None:
    monkeypatch.setenv("WAN_CONTROLLED_MOTION", "1")
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="And inside the flower were new seeds, ready to begin again.",
    ))
    assert "separate small brown oval new seeds" in prompt
    assert "empty dark disk" in prompt


def test_generic_rainbow_prompt_does_not_add_rain_or_soil(monkeypatch) -> None:
    monkeypatch.delenv("WAN_CONTROLLED_MOTION", raising=False)
    prompt = WanVideoGenerator.build_prompt(SegmentRequest(
        text="A beautiful rainbow appeared.",
        visual_prompt="A colorful rainbow arcs across a clearing sky.",
        motion="The rainbow appears from left to right.",
        keywords=["rainbow", "sky"],
    ))
    assert "rainbow appears from left to right" in prompt
    assert "raindrops" not in prompt.lower()
    assert "soil" not in prompt.lower()
    assert "Key visual elements" not in prompt


def test_reference_prompt_uses_image_for_identity_instead_of_rewriting_character() -> None:
    prompt = WanVideoGenerator.build_reference_prompt(SegmentRequest(
        text="Mia jumps into a puddle.",
        visual_prompt=(
            "GLOBAL_CHARACTER: Mia; human; age 10, brown hair, brown eyes. "
            "GLOBAL_STYLE: 2D children's book illustration. "
            "SCENE_DESCRIPTION: Mia jumps into a puddle."
        ),
        motion="Mia jumps and water splashes outward.",
        reference_id="character-123",
        reference_image="character.png",
    ))
    assert "person in the attached reference image" in prompt
    assert "same face" in prompt
    assert "GLOBAL_CHARACTER" not in prompt
    assert "brown hair" not in prompt
    assert "Mia jumps into a puddle" in prompt


def test_keyframe_weather_overlay_does_not_treat_rainbow_or_clearing_as_rain() -> None:
    assert CudaKeyframeMotionGenerator.uses_rain_overlay(SegmentRequest(
        text="Rain begins.", visual_prompt="Weather: light rain. Action: Mia watches."
    ))
    assert not CudaKeyframeMotionGenerator.uses_rain_overlay(SegmentRequest(
        text="The rain stopped.", visual_prompt="Weather: clearing. Action: Mia looks up."
    ))
    assert not CudaKeyframeMotionGenerator.uses_rain_overlay(SegmentRequest(
        text="A rainbow appears.", visual_prompt="Weather: sunny. Action: Mia points."
    ))


def test_layered_keyframe_assets_are_discovered_as_one_scene_bundle(tmp_path) -> None:
    keyframe = tmp_path / "scene_02_keyframe.png"
    background = tmp_path / "scene_02_background.png"
    character = tmp_path / "scene_02_character.png"
    for path in (keyframe, background, character):
        path.touch()
    segment = SegmentRequest(text="Mia jumps.", keyframe_image=str(keyframe))
    assert CudaKeyframeMotionGenerator.layered_assets(keyframe) == (
        background, character,
    )
    assert CudaKeyframeMotionGenerator.has_layered_assets(segment)


def test_simple_character_rig_has_one_stable_identity_and_distinct_poses(tmp_path) -> None:
    config = tmp_path / "simple_character.json"
    config.write_text('{"shirt":"#F3C62F","boots":"#D8443E"}', encoding="utf-8")
    rig = SimpleCharacterRig(config, 160, 288)
    standing = rig.render(scene=3, progress=0.0)
    jumping = rig.render(scene=2, progress=0.5)
    pointing_start = rig.render(scene=4, progress=0.0)
    pointing_end = rig.render(scene=4, progress=1.0)

    assert standing.mode == "RGBA"
    assert standing.size == (160, 288)
    assert standing.getbbox() != jumping.getbbox()
    assert pointing_start.tobytes() != pointing_end.tobytes()
    reference = tmp_path / "reference.png"
    rig.save_reference(reference)
    assert reference.is_file()
