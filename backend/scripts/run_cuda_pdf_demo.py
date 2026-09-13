"""Exercise PDF planning and the selected generator through the API app."""

import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app


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
    for video in current["videos"]:
        print(f"gpu={video['gpu_id']} artifact={video['path']}")
    if current["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
