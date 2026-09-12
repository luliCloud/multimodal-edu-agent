import hashlib
import shutil
import subprocess
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact


def _color_for_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


class MockVideoGenerator:
    def __init__(self, output_dir: Path, duration_seconds: int = 3) -> None:
        self.output_dir = output_dir
        self.duration_seconds = duration_seconds
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, segment_id: str, segment: SegmentRequest) -> VideoArtifact:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            path = self._generate_mp4(ffmpeg, segment_id, segment)
            return VideoArtifact(
                segment_id=segment_id,
                path=str(path),
                media_type="video/mp4",
                duration_seconds=self.duration_seconds,
            )

        path = self.output_dir / f"{segment_id}.txt"
        path.write_text(f"Mock video for: {segment.text}\n", encoding="utf-8")
        return VideoArtifact(
            segment_id=segment_id,
            path=str(path),
            media_type="text/plain",
            duration_seconds=self.duration_seconds,
        )

    def _generate_mp4(
        self, ffmpeg: str, segment_id: str, segment: SegmentRequest
    ) -> Path:
        path = self.output_dir / f"{segment_id}.mp4"
        title = (segment.title or "EduTok Segment").replace(":", " -")
        color = _color_for_text(segment.text)
        text = segment.text[:90].replace(":", " -").replace("'", "")
        vf = (
            f"color=c={color}:s=640x360:d={self.duration_seconds},"
            "format=yuv420p,"
            f"drawtext=text='{title}':x=32:y=36:fontsize=28:fontcolor=white,"
            f"drawtext=text='{text}':x=32:y=170:fontsize=22:fontcolor=white"
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                vf,
                "-t",
                str(self.duration_seconds),
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return path

