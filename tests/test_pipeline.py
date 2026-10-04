import unittest
import tempfile
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.services.pipeline import LocalPipeline
from backend.scripts.render_short_script import publish_final_video
from backend.app.models.shorts import ShortPlan
from backend.app.services.script_video import upload_request_from_script


class LocalPipelineTest(unittest.TestCase):
    def test_local_pipeline_generates_artifacts(self) -> None:
        request = UploadRequest(
            title="Unit Test",
            segments=[
                SegmentRequest(text="First segment"),
                SegmentRequest(text="Second segment"),
            ],
        )

        progress = []
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "videos"
            job = LocalPipeline(
                settings=Settings(generator_backend="mock", storage_dir=output_dir),
                progress_callback=progress.append,
            ).submit(request, run_inline=True)

            self.assertEqual(job.status, "completed")
            self.assertEqual(job.progress, 1.0)
            self.assertEqual(len(job.videos), 3)
            self.assertTrue(job.videos[-1].segment_id.endswith("-combined"))
            self.assertEqual(len(progress), 6)
            self.assertIn("scene 1/2: generating", progress[0])
            self.assertIn("combined video complete", progress[-1])
            for artifact in job.videos:
                self.assertTrue(Path(artifact.path).exists())

            final_output = Path(directory) / "final.mp4"
            published = publish_final_video(job, final_output)
            self.assertEqual(published, final_output)
            self.assertTrue(final_output.exists())
            self.assertEqual(
                final_output.read_bytes(), Path(job.videos[-1].path).read_bytes()
            )


def test_mock_generator_sizes_clip_to_synced_narration_duration(tmp_path) -> None:
    """Once tts/sync_storyboard_audio.py has set start/end_seconds on a
    segment, the mock backend should size the clip to that real duration
    instead of the fixed mock_clip_seconds default."""
    request = UploadRequest(
        title="Synced",
        segments=[
            SegmentRequest(text="Segment with real timing",
                           start_seconds=1.0, end_seconds=6.5),
            SegmentRequest(text="Segment with no timing yet"),
        ],
    )
    job = LocalPipeline(
        settings=Settings(generator_backend="mock", storage_dir=tmp_path),
    ).submit(request, run_inline=True)

    assert job.status == "completed"
    scene_videos = [video for video in job.videos if not video.segment_id.endswith("-combined")]
    assert scene_videos[0].duration_seconds == 6  # round(6.5 - 1.0)
    assert scene_videos[1].duration_seconds == 3  # falls back to mock_clip_seconds default


if __name__ == "__main__":
    unittest.main()


def test_reviewed_script_attaches_one_visual_bible_to_every_human_scene(tmp_path) -> None:
    source = [{"id": index, "page": 1, "text": f"Event {index}."}
              for index in range(1, 5)]
    plan = ShortPlan.model_validate({
        "title": "Story", "summary": "A child experiences four simple events in order.",
        "character": {
            "name": "Mia", "kind": "human child",
            "visual_identity": "one distinctive child",
            "outfits": {"default": "simple clothes"},
            "states": {"default": "same physical identity throughout"},
        },
        "scenes": [{
            "scene": index, "action": f"Event {index}", "location": "home",
            "weather": "clear", "camera": "medium", "motion": "She moves gently.",
            "narration": "Mia experiences this simple event with joy.",
            "outfit_id": "default", "state_id": "default",
            "source_sentence_ids": [index], "text": f"Event {index}.",
            "pages": [1], "narration_seconds": 3.0,
        } for index in range(1, 5)],
        "source_sentences": source,
    })
    reference = tmp_path / "character.png"
    reference.touch()
    (tmp_path / "scene_01_keyframe.png").touch()
    request = upload_request_from_script(
        plan, character_reference_path=reference, keyframe_dir=tmp_path,
    )

    assert request.segments
    assert {segment.reference_image for segment in request.segments} == {
        str(reference.resolve())
    }
    assert len({segment.reference_id for segment in request.segments}) == 1
    assert all(segment.reference_prompt for segment in request.segments)
    assert request.segments[0].keyframe_image == str(
        (tmp_path / "scene_01_keyframe.png").resolve()
    )
    assert all(segment.keyframe_image is None for segment in request.segments[1:])
