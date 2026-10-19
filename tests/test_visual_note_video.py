import json

from PIL import Image

from backend.app.models.shorts import ShortPlan
from backend.app.services.visual_note_video import (
    cell_membrane_plan, extract_cell_membrane_bible,
)


def test_visual_note_plan_matches_short_contract() -> None:
    plan = ShortPlan.model_validate(cell_membrane_plan())
    assert len(plan.scenes) == 4
    assert sum(len(scene.narration.split()) for scene in plan.scenes) <= 36


def test_visual_bible_preserves_source_crops(tmp_path) -> None:
    source = Image.new("RGB", (1000, 1500), "white")
    manifest = extract_cell_membrane_bible(source, tmp_path, "notes.pdf")
    assert [asset["id"] for asset in manifest["assets"]] == [
        "phospholipid_bilayer", "ion_channel",
    ]
    saved = json.loads((tmp_path / "visual_bible.json").read_text())
    assert saved["focus"]["region"] == "first quarter"
    assert (tmp_path / "assets" / "phospholipid_bilayer.png").is_file()
    assert (tmp_path / "assets" / "ion_channel.png").is_file()
