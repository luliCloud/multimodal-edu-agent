"""Concatenate completed scene clips in their story order."""

import subprocess
import tempfile
import shutil
from pathlib import Path

from backend.app.models.jobs import VideoArtifact


def assemble_mp4(doc_id: str, videos: list[VideoArtifact], output_dir: Path) -> Path:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install ffmpeg or imageio-ffmpeg to assemble scene clips")

    if not videos or any(video.media_type != "video/mp4" for video in videos):
        raise ValueError("All scenes must have MP4 artifacts before assembly")
    output = output_dir / f"{doc_id}-combined.mp4"
    with tempfile.TemporaryDirectory() as directory:
        playlist = Path(directory) / "scenes.txt"
        paths = [Path(video.path).resolve() for video in videos]
        if any("'" in str(path) or not path.is_file() for path in paths):
            raise ValueError("A scene file is missing or has an unsupported path")
        playlist.write_text("".join(f"file '{path}'\n" for path in paths), encoding="utf-8")
        result = subprocess.run(
            [ffmpeg, "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(playlist),
             "-c", "copy", "-movflags", "+faststart", str(output)],
            capture_output=True, timeout=120, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"MP4 assembly failed: {result.stderr.decode(errors='replace')[-500:]}")
    return output
