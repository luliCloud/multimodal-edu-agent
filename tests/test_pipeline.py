import unittest
import tempfile
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.services.pipeline import LocalPipeline
from backend.scripts.render_short_script import publish_final_video


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


if __name__ == "__main__":
    unittest.main()
