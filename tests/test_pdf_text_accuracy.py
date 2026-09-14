from pathlib import Path

from backend.app.services.pdf_keywords import extract_pdf_pages
from backend.app.services.story_planner import source_sentences


def test_little_seed_pdf_preserves_key_story_events() -> None:
    pdf = (Path(__file__).parents[1] / "test_pdfs/The_Little_Seed.pdf").read_bytes()
    sentences = source_sentences(extract_pdf_pages(pdf))
    text = " ".join(item["text"] for item in sentences)
    assert "The rain fell softly on the ground." in text
    assert "The little seed drank the water." in text
    assert "Soon, a tiny root grew down into the soil." in text
    assert "It had two small leaves. Then it had four leaves." in text
    assert "A bee flew over to visit." in text
    assert "inside the flower were new seeds" in text


def test_after_rain_pdf_preserves_short_visual_events() -> None:
    pdf = (Path(__file__).parents[1] / "test_pdfs/After the Rain.pdf").read_bytes()
    sentences = source_sentences(extract_pdf_pages(pdf))
    text = " ".join(item["text"] for item in sentences)
    assert "Mia put on her yellow raincoat and red boots." in text
    assert "Mia jumped into one." in text
    assert "Then the rain stopped." in text
    assert "A beautiful rainbow appeared." in text
