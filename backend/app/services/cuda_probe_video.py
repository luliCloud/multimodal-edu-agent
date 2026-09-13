"""Synthetic CUDA clip for testing scheduling and encoding, not text-to-video inference."""

import hashlib
import subprocess
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact


class CudaProbeVideoGenerator:
    def __init__(self, output_dir: Path, duration_seconds: int = 3) -> None:
        self.output_dir = output_dir
        self.duration_seconds = duration_seconds
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, segment_id: str, segment: SegmentRequest, gpu_id: int) -> VideoArtifact:
        import imageio_ffmpeg
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend selected but no GPU is visible")
        device = torch.device(f"cuda:{gpu_id}")
        width, height, fps = 256, 144, 8
        x = torch.linspace(0, 1, width, device=device).view(1, width)
        y = torch.linspace(0, 1, height, device=device).view(height, 1)
        color = bytes.fromhex(hashlib.sha256(segment.text.encode()).hexdigest()[:6])
        frames = bytearray()
        for index in range(self.duration_seconds * fps):
            phase = index / (self.duration_seconds * fps)
            red = (x.expand(height, -1) + color[0] / 255 + phase) % 1
            green = (y.expand(-1, width) + color[1] / 255 + phase) % 1
            blue = ((x + y) / 2 + color[2] / 255 + phase) % 1
            frame = torch.stack((red, green, blue), dim=-1)
            frames.extend((frame * 255).to(torch.uint8).cpu().numpy().tobytes())
        path = self.output_dir / f"{segment_id}.mp4"
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "pipe:0", "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
             str(path)],
            input=frames, capture_output=True, timeout=60, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Video encoding failed: {result.stderr.decode(errors='replace')[-500:]}")
        return VideoArtifact(segment_id=segment_id, path=str(path), media_type="video/mp4",
                             duration_seconds=self.duration_seconds, gpu_id=gpu_id)
