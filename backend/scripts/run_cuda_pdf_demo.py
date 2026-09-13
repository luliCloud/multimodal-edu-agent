"""Exercise PDF planning and the selected generator through the API app."""

import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.jobs import VideoArtifact
from backend.app.services.video_assembly import assemble_mp4


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_pdfs/The_Little_Seed.pdf")
    client = TestClient(app)
    response = client.post("/pdf/jobs", content=path.read_bytes(),
                           headers={"Content-Type": "application/pdf"})
    response.raise_for_status()
    job = response.json()
    print(f"job_id={job['job_id']} scenes={len(job['segments'])}")
    while True:
        current = client.get(f"/status/{job['job_id']}").json()
        if current["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    print(f"status={current['status']} error={current['error']}")
    for index, video in enumerate(current["videos"], 1):
        print(f"scene={index} gpu={video['gpu_id']} artifact={video['path']}")
    if current["status"] != "completed":
        raise SystemExit(1)
    combined = next((video for video in current["videos"]
                     if video["segment_id"] == f"{job['doc_id']}-combined"), None)
    if combined:
        output = Path(combined["path"])
    elif all(video["media_type"] == "video/mp4" for video in current["videos"]):
        output = assemble_mp4(job["doc_id"],
                              [VideoArtifact.model_validate(video) for video in current["videos"]],
                              Path("storage/videos"))
    else:
        output = None
    if output is not None:
        print(f"combined={output}")


if __name__ == "__main__":
    main()
