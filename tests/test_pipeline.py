import unittest
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.services.pipeline import LocalPipeline


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
        job = LocalPipeline(progress_callback=progress.append).submit(
            request, run_inline=True
        )

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.progress, 1.0)
        self.assertEqual(len(job.videos), 3)
        self.assertTrue(job.videos[-1].segment_id.endswith("-combined"))
        self.assertEqual(len(progress), 6)
        self.assertIn("scene 1/2: generating", progress[0])
        self.assertIn("combined video complete", progress[-1])
        for artifact in job.videos:
            self.assertTrue(Path(artifact.path).exists())


if __name__ == "__main__":
    unittest.main()
