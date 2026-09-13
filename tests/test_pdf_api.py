import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app


def test_pdf_plan_and_job() -> None:
    pdf = (Path(__file__).parents[1] / "test_pdfs/The_Little_Seed.pdf").read_bytes()
    client = TestClient(app)
    plan_response = client.post("/pdf/plan", content=pdf, headers={"Content-Type": "application/pdf"})
    assert plan_response.status_code == 200
    plan = plan_response.json()
    assert plan["title"] == "The Little Seed"
    assert len(plan["scenes"]) == 7
    assert "dark soil" in plan["scenes"][0]["text"]
    assert "new seeds" in plan["scenes"][-1]["text"]

    job_response = client.post("/pdf/jobs", content=pdf, headers={"Content-Type": "application/pdf"})
    assert job_response.status_code == 200
    job = job_response.json()
    assert len(job["segments"]) == 7
    for _ in range(100):
        current = client.get(f"/status/{job['job_id']}").json()
        if current["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert current["status"] == "completed", current.get("error")
    assert len(client.get(f"/videos/{job['doc_id']}").json()) == 7
