from backend.app.models.jobs import SegmentRequest
from backend.app.services.wan_video import WanVideoGenerator


def test_prompt_uses_scene_and_visual_keywords() -> None:
    prompt = WanVideoGenerator.build_prompt(
        SegmentRequest(text="A tiny root grows from the seed.", keywords=["seed", "root", "soil"])
    )
    assert "tiny root grows" in prompt
    assert "seed, root, soil" in prompt
    assert "watercolor nature animation" in prompt
    assert "on-screen text" not in prompt


def test_prompt_omits_spoken_dialogue() -> None:
    prompt = WanVideoGenerator.build_prompt(
        SegmentRequest(text='A bee visits the flower. “Buzz, buzz!” said the bee. The flower sways.')
    )
    assert "bee visits" in prompt
    assert "flower sways" in prompt
    assert "Buzz" not in prompt
