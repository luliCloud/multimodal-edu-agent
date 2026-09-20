import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app.api.routes as routes
from backend.app.main import app
from backend.app.services.storyboard_schema import ScopeExceededError, StoryPlanningError


def _story_pdf() -> bytes:
    return (Path(__file__).parents[1] / "test_pdfs/The_Little_Seed.pdf").read_bytes()


def test_pdf_plan_and_job() -> None:
    pdf = (Path(__file__).parents[1] / "test_pdfs/The_Little_Seed.pdf").read_bytes()
    client = TestClient(app)
    plan_response = client.post("/pdf/plan", content=pdf, headers={"Content-Type": "application/pdf"})
    assert plan_response.status_code == 200
    plan = plan_response.json()
    assert plan["title"] == "The Little Seed"
    assert len(plan["scenes"]) == 4
    assert "dark soil" in plan["scenes"][0]["text"]
    assert any("seed" in keyword for keyword in plan["scenes"][0]["keywords"])
    assert "new seeds" in plan["scenes"][-1]["text"]

    job_response = client.post("/pdf/jobs", content=pdf, headers={"Content-Type": "application/pdf"})
    assert job_response.status_code == 200
    job = job_response.json()
    assert len(job["segments"]) == 4
    assert any("seed" in keyword for keyword in job["segments"][0]["keywords"])
    assert job["segments"][0]["keywords"] == plan["scenes"][0]["keywords"]
    for _ in range(100):
        current = client.get(f"/status/{job['job_id']}").json()
        if current["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert current["status"] == "completed", current.get("error")
    videos = client.get(f"/videos/{job['doc_id']}").json()
    assert len(videos) == 5
    assert videos[-1]["segment_id"].endswith("-combined")
    assert client.get("/media/missing.mp4").status_code == 404


def test_plan_carries_narration_to_job_segments() -> None:
    client = TestClient(app)
    pdf = _story_pdf()
    scene = client.post("/pdf/plan", content=pdf,
                        headers={"Content-Type": "application/pdf"}).json()["scenes"][0]
    assert scene["narration"]
    assert scene["narration_seconds"] >= 3.0
    job = client.post("/pdf/jobs", content=pdf,
                      headers={"Content-Type": "application/pdf"}).json()
    assert job["segments"][0]["narration"] == scene["narration"]


def _reviewed_script() -> dict:
    source = [
        {"id": 1, "page": 1, "text": "Mia watches the rain."},
        {"id": 2, "page": 1, "text": "Mia jumps into a puddle."},
        {"id": 3, "page": 1, "text": "The clouds move away."},
        {"id": 4, "page": 1, "text": "A rainbow appears."},
    ]
    actions = ["Mia watches rain", "Mia jumps into a puddle",
               "Clouds move away", "A rainbow appears"]
    narrations = ["Mia watches gentle rain through the window.",
                  "Mia happily jumps into a shiny puddle.",
                  "The dark clouds slowly move far away.",
                  "A colorful rainbow appears across the sky."]
    return {
        "title": "Reviewed Rain Script",
        "summary": "Mia watches rain, jumps in a puddle, and sees a rainbow.",
        "character": {
            "name": "Mia", "kind": "human child", "visual_identity": "young child",
            "clothes": "yellow raincoat", "outfits": {"default": "yellow raincoat"},
            "states": {"default": "young child"},
        },
        "style": {},
        "source_sentences": source,
        "duration_seconds": 15,
        "scenes": [{
            "scene": index,
            "source_sentence_ids": [index],
            "text": source[index - 1]["text"],
            "pages": [1],
            "keywords": ["rain" if index < 4 else "rainbow"],
            "action": actions[index - 1],
            "location": "sidewalk",
            "weather": "rainy" if index < 3 else "clearing",
            "camera": "medium shot",
            "motion": actions[index - 1],
            "narration": narrations[index - 1],
            "narration_seconds": 3.0,
            "outfit_id": "default",
            "state_id": "default",
            "visual_prompt": actions[index - 1],
        } for index in range(1, 5)],
    }


def test_reviewed_script_can_start_video_without_replanning() -> None:
    client = TestClient(app)
    response = client.post("/scripts/jobs", json=_reviewed_script())
    assert response.status_code == 200
    job = response.json()
    assert len(job["segments"]) == 4
    for _ in range(100):
        current = client.get(f"/status/{job['job_id']}").json()
        if current["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert current["status"] == "completed", current.get("error")
    videos = client.get(f"/videos/{job['doc_id']}").json()
    assert len(videos) == 5
    assert videos[-1]["segment_id"].endswith("-combined")


@pytest.mark.parametrize("error, expected_status", [
    (ScopeExceededError("Storyboard supports up to 32 narrative sentences"), 422),
    (StoryPlanningError("Storyboard invalid after 3 attempts"), 502),
])
def test_planner_errors_are_reported_not_raised_as_500(monkeypatch, error, expected_status) -> None:
    def fail(pages, backend, title="Untitled PDF"):
        raise error

    monkeypatch.setattr(routes, "plan_story", fail)
    response = TestClient(app).post("/pdf/plan", content=_story_pdf(),
                                    headers={"Content-Type": "application/pdf"})
    assert response.status_code == expected_status
    assert str(error) in response.json()["detail"]
