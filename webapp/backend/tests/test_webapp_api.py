import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.config import WebSettings
from webapp.backend.app.main import create_app

REPO = Path(__file__).resolve().parents[3]
PDF = REPO / "test_pdfs" / "The_Little_Seed.pdf"


def _settings(tmp_path: Path, **overrides) -> WebSettings:
    values = dict(data_dir=tmp_path / "projects", video_dir=tmp_path / "videos",
                  render_dir=tmp_path / "scratch", prerendered_dir=tmp_path / "demo",
                  render_backend="mock", planner_backend="extractive")
    values.update(overrides)
    settings = WebSettings(**values)
    for directory in (settings.data_dir, settings.video_dir, settings.render_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings


def _wait(client: TestClient, project_id: str, *done: str) -> dict:
    for _ in range(400):
        project = client.get(f"/api/projects/{project_id}").json()
        if project["status"] in done:
            return project
        time.sleep(0.05)
    raise AssertionError(f"timed out in status {project['status']}")


def _upload(client: TestClient) -> dict:
    response = client.post("/api/projects", files={
        "file": ("The_Little_Seed.pdf", PDF.read_bytes(), "application/pdf")})
    assert response.status_code == 202, response.text
    return response.json()


def test_upload_edit_render_and_serve(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.importorskip("imageio_ffmpeg")
    client = TestClient(create_app(_settings(tmp_path)))
    created = _upload(client)
    project = _wait(client, created["id"], "draft", "failed")
    assert project["status"] == "draft", project["error"]
    assert project["title"] == "The Little Seed"
    assert project["source"]["page_count"] >= 1
    script = project["script"]
    assert len(script["scenes"]) == 4

    script["scenes"][0]["action"] = "A seed sleeps in dark soil"
    script["scenes"][0]["narration"] = "A tiny seed sleeps underground."
    script["scenes"].pop()
    saved = client.put(f"/api/projects/{project['id']}/script",
                       json={"script": script, "base_revision": project["script_revision"]})
    assert saved.status_code == 200, saved.text
    saved = saved.json()
    assert len(saved["script"]["scenes"]) == 3
    assert "A seed sleeps in dark soil" in saved["script"]["scenes"][0]["visual_prompt"]
    stale = client.put(f"/api/projects/{project['id']}/script",
                       json={"script": script, "base_revision": project["script_revision"]})
    assert stale.status_code == 409

    started = client.post(f"/api/projects/{project['id']}/render", json={})
    assert started.status_code == 202, started.text
    done = _wait(client, project["id"], "rendered", "draft")
    assert done["status"] == "rendered", done["error"]
    kinds = [video["kind"] for video in done["videos"]]
    assert kinds == ["final", "scene", "scene", "scene"]
    final = done["videos"][0]
    head = client.get(final["url"], headers={"Range": "bytes=0-99"})
    assert head.status_code == 206 and len(head.content) == 100

    library = client.get("/api/library").json()
    assert {item["key"] for item in library} == {video["key"] for video in done["videos"]}
    assert all(item["project_title"] == "The Little Seed" for item in library)

    client.post(f"/api/projects/{project['id']}/render", json={"scenes": [2]})
    again = _wait(client, project["id"], "rendered", "draft")
    assert [video["scene"] for video in again["videos"]] == [None, 2]
    assert len(client.get("/api/library").json()) == 2  # previous render cleaned up

    assert client.delete(f"/api/projects/{project['id']}").status_code == 204
    assert client.get("/api/library").json() == []


def test_directory_backend_and_external_videos(tmp_path: Path) -> None:
    settings = _settings(tmp_path, render_backend="directory")
    demo = settings.prerendered_dir / "the_little_seed"
    demo.mkdir(parents=True)
    (demo / "final.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    (settings.video_dir / "dropped-in.mp4").write_bytes(b"fake")
    client = TestClient(create_app(settings))
    project = _wait(client, _upload(client)["id"], "draft", "failed")
    client.post(f"/api/projects/{project['id']}/render")
    done = _wait(client, project["id"], "rendered", "draft")
    assert done["status"] == "rendered", done["error"]
    assert client.get(done["videos"][0]["url"]).content.endswith(b"fake")
    kinds = sorted(item["kind"] for item in client.get("/api/library").json())
    assert kinds == ["external", "final"]


def test_rejects_non_pdf_and_bad_media_keys(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))
    bad = client.post("/api/projects", files={"file": ("x.pdf", b"hello", "application/pdf")})
    assert bad.status_code == 415
    assert client.get("/api/media/../secrets.mp4").status_code == 404
    assert client.get("/api/projects/not-an-id").status_code == 404


def test_planner_script_round_trips_through_editor() -> None:
    from backend.app.models.shorts import ShortPlan
    from backend.app.services.shorts_planner import scene_visual_prompt
    from tests.test_pdf_api import _reviewed_script
    from webapp.backend.app.models import EditableScript
    from webapp.backend.app.services.pipeline_bridge import (
        _upload_request, refresh_derived_fields, validate_script,
    )

    raw = _reviewed_script()
    plan = ShortPlan.model_validate(raw)
    script = refresh_derived_fields(EditableScript.model_validate({**raw, "planner_backend": "qwen"}))
    assert script.scenes[1].visual_prompt == scene_visual_prompt(plan, plan.scenes[1])
    assert validate_script(script) == []
    request = _upload_request(script, [1, 3], "mock", Path("/nonexistent"))
    assert [segment.title for segment in request.segments] == ["Scene 1", "Scene 3"]

    script.scenes[0].custom_prompt = True
    script.scenes[0].visual_prompt = "hand written"
    script.scenes[0].narration = "Too short."
    script = refresh_derived_fields(script)
    assert script.scenes[0].visual_prompt == "hand written"
    issues = validate_script(script)
    assert any(issue.scene == 1 and "words" in issue.message for issue in issues)
    # Contract broken by the edit: rendering still works via the direct segment path.
    assert _upload_request(script, [1], "mock", Path("/x")).segments[0].visual_prompt == "hand written"
