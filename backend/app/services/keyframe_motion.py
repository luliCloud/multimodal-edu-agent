"""CUDA camera/weather animation that never redraws a reviewed keyframe."""

import subprocess
import hashlib
import re
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact


class CudaKeyframeMotionGenerator:
    """Animate pixels from a scene keyframe while preserving character identity exactly."""

    def __init__(self, output_dir: Path, width: int, height: int,
                 num_frames: int, fps: int) -> None:
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps

    @staticmethod
    def uses_rain_overlay(segment: SegmentRequest) -> bool:
        visual = segment.visual_prompt or ""
        weather = re.search(r"Weather:\s*([^.]*)", visual, flags=re.IGNORECASE)
        if weather:
            description = weather.group(1).lower()
            return "rain" in description and not any(
                term in description for term in ("clearing", "stopped", "sunny")
            )
        return bool(re.search(r"\b(?:rain|raindrops?)\b", segment.text, re.IGNORECASE))

    def generate(self, segment_id: str, segment: SegmentRequest, gpu_id: int) -> VideoArtifact:
        import imageio_ffmpeg
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from PIL import Image, ImageOps

        if not segment.keyframe_image:
            raise ValueError("Stable keyframe animation needs a scene keyframe")
        path = Path(segment.keyframe_image)
        if not path.is_file():
            raise FileNotFoundError(f"Scene keyframe does not exist: {path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Stable keyframe animation needs a visible CUDA GPU")

        with Image.open(path) as source:
            fitted = ImageOps.fit(
                source.convert("RGB"), (self.width, self.height),
                method=Image.Resampling.LANCZOS,
            )
            pixels = np.asarray(fitted).copy()
        device = torch.device(f"cuda:{gpu_id}")
        image = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=device, dtype=torch.float32),
            torch.arange(self.width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        rainy = self.uses_rain_overlay(segment)
        direction_seed = int(hashlib.sha256(segment.text.encode()).hexdigest()[:2], 16)
        direction = -1.0 if direction_seed % 2 else 1.0
        raw = bytearray()
        with torch.inference_mode():
            for index in range(self.num_frames):
                progress = index / max(self.num_frames - 1, 1)
                # Gentle push-in plus a tiny horizontal drift. This transforms the complete
                # reviewed frame uniformly, so facial and body proportions cannot mutate.
                zoom = 1.0 + 0.035 * progress
                theta = torch.tensor(
                    [[[1.0 / zoom, 0.0, direction * 0.018 * progress],
                      [0.0, 1.0 / zoom, -0.012 * progress]]],
                    device=device, dtype=torch.float32,
                )
                grid = functional.affine_grid(theta, image.shape, align_corners=False)
                frame = functional.grid_sample(
                    image, grid, mode="bilinear", padding_mode="border",
                    align_corners=False,
                )[0]
                if rainy:
                    phase = torch.remainder(
                        yy + xx * 0.38 - progress * self.height * 1.8, 67.0
                    )
                    lanes = torch.remainder(xx + index * 13.0, 43.0)
                    drops = ((phase < 8.0) & (lanes < 1.5)).float() * 0.32
                    rain_color = torch.tensor(
                        (0.82, 0.93, 1.0), device=device
                    ).view(3, 1, 1)
                    frame = frame * (1 - drops.unsqueeze(0)) + rain_color * drops.unsqueeze(0)
                raw.extend((frame.permute(1, 2, 0).clamp(0, 1) * 255)
                           .to(torch.uint8).cpu().numpy().tobytes())

        output = self.output_dir / f"{segment_id}.mp4"
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
             "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(output)],
            input=raw, capture_output=True, timeout=120, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Keyframe MP4 encoding failed: "
                + result.stderr.decode(errors="replace")[-500:]
            )
        return VideoArtifact(
            segment_id=segment_id, path=str(output), media_type="video/mp4",
            duration_seconds=round(self.num_frames / self.fps), gpu_id=gpu_id,
        )
