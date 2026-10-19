import hashlib
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact


def _color_for_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


@lru_cache(maxsize=4)
def _supports_drawtext(ffmpeg: str) -> bool:
    """Homebrew's default ffmpeg ships without libfreetype, so drawtext is absent."""
    result = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                            capture_output=True, text=True, check=False)
    return " drawtext " in result.stdout


MIN_CLIP_SECONDS = 0.5


def _segment_duration(segment: SegmentRequest, default: int) -> float:
    """Actual audio-synced duration if tts/sync_storyboard_audio.py has run,
    otherwise the fixed default (word-count estimates are not reliable
    enough to size a clip from)."""
    if segment.start_seconds is not None and segment.end_seconds is not None:
        return max(segment.end_seconds - segment.start_seconds, MIN_CLIP_SECONDS)
    return default


class MockVideoGenerator:
    def __init__(self, output_dir: Path, duration_seconds: int = 3) -> None:
        self.output_dir = output_dir
        self.duration_seconds = duration_seconds
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, segment_id: str, segment: SegmentRequest) -> VideoArtifact:
        duration = _segment_duration(segment, self.duration_seconds)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass
        if ffmpeg:
            path = self._generate_mp4(ffmpeg, segment_id, segment, duration)
            return VideoArtifact(
                segment_id=segment_id,
                path=str(path),
                media_type="video/mp4",
                duration_seconds=round(duration),
            )

        path = self.output_dir / f"{segment_id}.txt"
        path.write_text(f"Mock video for: {segment.text}\n", encoding="utf-8")
        return VideoArtifact(
            segment_id=segment_id,
            path=str(path),
            media_type="text/plain",
            duration_seconds=round(duration),
        )

    def _generate_mp4(
        self, ffmpeg: str, segment_id: str, segment: SegmentRequest, duration: float
    ) -> Path:
        path = self.output_dir / f"{segment_id}.mp4"
        title = (segment.title or "EduTok Segment").replace(":", " -")
        color = _color_for_text(segment.text)
        text = segment.text[:90].replace(":", " -").replace("'", "")
        vf = f"color=c={color}:s=640x360:d={duration},format=yuv420p"
        if _supports_drawtext(ffmpeg):
            vf += (
                f",drawtext=text='{title}':x=32:y=36:fontsize=28:fontcolor=white"
                f",drawtext=text='{text}':x=32:y=170:fontsize=22:fontcolor=white"
            )
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                vf,
                "-t",
                str(duration),
                "-movflags",
                "+faststart",
                str(path),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            lines = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
            raise RuntimeError(f"ffmpeg could not render {segment_id}: "
                               f"{lines[-1] if lines else 'unknown error'}")
        return path
