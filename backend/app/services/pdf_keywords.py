"""CPU-only PDF text and candidate video-topic extraction."""

import re
import shutil
import subprocess
from collections import Counter
from math import ceil


STOPWORDS = set("""a an and are as at be by can for from has have in into is it of on or our
per so that the their this to under use using was were will with video videos system
project design document component components current stage implementation testing test
requirements requirement pipeline process processing based each one two three more than
not yet future local initial planned described""".split())
STOPWORDS.update({"available", "environment", "setup", "level", "version", "figure", "appendix"})
STOPWORDS.update({"little", "small", "tiny", "one", "two", "four", "day", "morning",
                  "soon", "then", "every", "again", "anymore", "said", "was", "had",
                  "gave", "came", "grew", "grown", "fell", "flew", "pushed", "drank",
                  "slept", "appeared", "ready", "into", "toward", "down", "over",
                  "inside", "out", "up", "softly", "quiet", "drip", "buzz", "hello"})
INFRASTRUCTURE_TERMS = {"parallelism", "gpu", "cuda", "worker", "workers", "scheduler",
                        "scheduling", "queue", "queues", "inference", "latency", "throughput",
                        "gpus", "redis", "celery", "orchestration", "script", "segments",
                        "interfaces", "tests", "implemented", "engine", "task"}


def extract_pdf_pages(pdf: bytes) -> list[str]:
    if len(pdf) > 10 * 1024 * 1024:
        raise ValueError("PDF exceeds 10 MB limit")
    if not pdf.startswith(b"%PDF-"):
        raise ValueError("Input is not a PDF")
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("pdftotext (poppler-utils) is required")
    result = subprocess.run(
        [executable, "-layout", "-", "-"], input=pdf, capture_output=True,
        timeout=30, check=False,
    )
    if result.returncode:
        raise ValueError("Could not extract text from PDF")
    pages = [page.strip() for page in result.stdout.decode("utf-8", errors="replace").split("\f")]
    pages = [page for page in pages if page]
    if not pages or sum(len(page) for page in pages) < 30:
        raise ValueError("PDF has no extractable text; scanned pages need OCR")
    return pages


def extract_video_keywords(pages: list[str], limit: int = 10) -> list[dict]:
    """Rank short visual terms and retain page evidence."""
    counts: Counter[str] = Counter()
    page_hits: dict[str, set[int]] = {}
    word_counts: Counter[str] = Counter()
    for page_number, page in enumerate(pages, 1):
        normalized = re.sub(r"\s+", " ", page.lower())
        for sentence in re.split(r"[.;:!?\n•]+", normalized):
            words = re.findall(r"[a-z][a-z0-9-]*", sentence)
            word_counts.update(words)
            for size in (1, 2, 3):
                for i in range(len(words) - size + 1):
                    phrase = words[i:i + size]
                    if any(word in STOPWORDS or len(word) < 3 for word in (phrase[0], phrase[-1])):
                        continue
                    if sum(word in STOPWORDS for word in phrase):
                        continue
                    if any(word in INFRASTRUCTURE_TERMS or "gpu" in word for word in phrase):
                        continue
                    key = " ".join(phrase)
                    counts[key] += 1
                    page_hits.setdefault(key, set()).add(page_number)
    ranked = sorted(counts, key=lambda key: (
        -len(page_hits[key]), -counts[key],
        -sum(word_counts[word] for word in key.split()), -len(key.split()), key))
    selected: list[str] = []
    for key in ranked:
        if len(selected) >= limit:
            break
        if any((f" {key} " in f" {other} " or f" {other} " in f" {key} ")
               for other in selected):
            continue
        selected.append(key)
    return [{"keyword": key, "occurrences": counts[key], "pages": sorted(page_hits[key])} for key in selected]


def extract_video_scenes(pages: list[str], limit: int = 8) -> list[dict]:
    """Keep narrative order and source pages when planning short clips."""
    if limit < 1:
        raise ValueError("Scene limit must be positive")
    sentences: list[tuple[int, str]] = []
    for page_number, page in enumerate(pages, 1):
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        lines = [line for line in lines if line.lower() != "the end" and not line.isdigit()]
        if page_number == 1 and lines and len(lines[0].split()) <= 8 and not re.search(r"[.!?]$", lines[0]):
            lines = lines[1:]  # title, not a scene
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(lines)):
            sentence = sentence.strip(" \"“”")
            if len(re.findall(r"[A-Za-z]+", sentence)) >= 5:
                sentences.append((page_number, sentence))
    if not sentences:
        return []
    group_size = ceil(len(sentences) / limit)
    scenes = []
    for offset in range(0, len(sentences), group_size):
        group = sentences[offset:offset + group_size]
        scenes.append({"index": len(scenes) + 1,
                       "text": " ".join(text for _, text in group),
                       "pages": sorted({page for page, _ in group})})
    return scenes
