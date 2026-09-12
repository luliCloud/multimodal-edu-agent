from backend.app.models.jobs import SegmentRequest, UploadRequest
from backend.app.services.pipeline import LocalPipeline


def main() -> None:
    request = UploadRequest(
        title="Mac Local Skeleton Demo",
        segments=[
            SegmentRequest(
                title="Concept",
                text="Segment-level parallelism lets each script chunk become a separate video job.",
            ),
            SegmentRequest(
                title="Worker",
                text="A GPU worker will later replace this mock generator on CUDA machines.",
            ),
        ],
    )
    job = LocalPipeline().submit(request, run_inline=True)
    print(f"job_id={job.job_id}")
    print(f"doc_id={job.doc_id}")
    print(f"status={job.status}")
    for video in job.videos:
        print(f"video={video.path}")


if __name__ == "__main__":
    main()

