import unittest
from pathlib import Path

from backend.app.services.pdf_keywords import (
    _pages_from_pdfplumber,
    extract_pdf_pages,
    extract_video_keywords,
)


class PdfKeywordsTest(unittest.TestCase):
    def test_design_doc_topics_have_page_evidence(self) -> None:
        path = Path(__file__).parents[1] / "docs/design/Design_Spec_for_Multimodal_AI_Agent.pdf"
        pages = extract_pdf_pages(path.read_bytes())
        keywords = extract_video_keywords(pages)
        self.assertEqual(len(pages), 5)
        self.assertFalse(any("parallelism" in item["keyword"] for item in keywords))
        self.assertFalse(any("gpu" in item["keyword"] for item in keywords))
        self.assertTrue(all(item["pages"] for item in keywords))

    def test_teaching_topic_survives_filter(self) -> None:
        keywords = extract_video_keywords(["Photosynthesis converts sunlight into chemical energy. " * 3])
        self.assertIn("chemical energy", {item["keyword"] for item in keywords})

    def test_story_keywords_are_visual(self) -> None:
        path = Path(__file__).parents[1] / "test_pdfs/The_Little_Seed.pdf"
        words = {item["keyword"] for item in extract_video_keywords(extract_pdf_pages(path.read_bytes()))}
        self.assertIn("seed", " ".join(words))
        self.assertIn("flower", " ".join(words))
        self.assertNotIn("drip drip", words)

    def test_rejects_non_pdf(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a PDF"):
            extract_pdf_pages(b"hello")

    def test_pdfplumber_fallback_matches_expected_text(self) -> None:
        """Runs even where pdftotext exists, so the fallback cannot silently rot."""
        path = Path(__file__).parents[1] / "test_pdfs/The_Little_Seed.pdf"
        pages = [page.strip() for page in _pages_from_pdfplumber(path.read_bytes())]
        text = " ".join(pages)
        self.assertIn("The rain fell softly on the ground.", text)
        self.assertIn("A bee flew over to visit.", text)

    def test_pdfplumber_fallback_rejects_corrupt_pdf(self) -> None:
        with self.assertRaisesRegex(ValueError, "Could not extract text"):
            _pages_from_pdfplumber(b"%PDF-1.4 not really a pdf")


if __name__ == "__main__":
    unittest.main()
