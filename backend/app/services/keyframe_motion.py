"""CUDA camera/weather animation that never redraws a reviewed keyframe."""

import hashlib
import math
import re
import subprocess
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

    @staticmethod
    def layered_assets(keyframe: Path) -> tuple[Path, Path]:
        prefix = keyframe.name.removesuffix("_keyframe.png")
        return (
            keyframe.with_name(f"{prefix}_background.png"),
            keyframe.with_name(f"{prefix}_character.png"),
        )

    @classmethod
    def has_layered_assets(cls, segment: SegmentRequest) -> bool:
        if not segment.keyframe_image:
            return False
        background, character = cls.layered_assets(Path(segment.keyframe_image))
        return background.is_file() and character.is_file()

    def _load_layers(self, keyframe: Path, device):
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        background_path, character_path = self.layered_assets(keyframe)
        with Image.open(background_path) as source:
            background_image = ImageOps.fit(
                source.convert("RGB"), (self.width, self.height),
                method=Image.Resampling.LANCZOS,
            )
        with Image.open(character_path) as source:
            character = source.convert("RGBA")
            scene_match = re.search(r"scene_(\d+)", keyframe.name)
            scene = int(scene_match.group(1)) if scene_match else 1
            target_height = int(self.height * (0.72 if scene == 2 else 0.86))
            target_width = max(1, round(character.width * target_height / character.height))
            character = character.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
            layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            x = (self.width - target_width) // 2
            y = int(self.height * (0.16 if scene == 2 else 0.12))
            layer.alpha_composite(character, (x, y))

        background = torch.from_numpy(np.asarray(background_image).copy()).to(
            device=device, dtype=torch.float32
        ).permute(2, 0, 1) / 255.0
        foreground = torch.from_numpy(np.asarray(layer).copy()).to(
            device=device, dtype=torch.float32
        ).permute(2, 0, 1) / 255.0
        return background, foreground, scene

    def _layered_frame(self, torch, functional, background, foreground,
                       scene: int, progress: float, xx, yy):
        cycle = math.sin(progress * math.tau)
        if scene == 2:
            # One visible jump arc: rise, hang briefly, and land back over the puddle.
            dy = -34.0 * math.sin(progress * math.pi)
            dx = 5.0 * math.sin(progress * math.tau)
            angle = 0.035 * cycle
        else:
            dy = 3.0 * cycle
            dx = 2.5 * math.sin(progress * math.tau + scene)
            angle = 0.012 * cycle * (-1 if scene == 4 else 1)
        cosine, sine = math.cos(angle), math.sin(angle)
        theta = torch.tensor(
            [[[cosine, -sine, -2.0 * dx / self.width],
              [sine, cosine, -2.0 * dy / self.height]]],
            device=background.device, dtype=torch.float32,
        )
        source = foreground.unsqueeze(0)
        grid = functional.affine_grid(theta, source.shape, align_corners=False)
        moved = functional.grid_sample(
            source, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0]
        alpha = moved[3:4].clamp(0, 1)
        frame = background * (1 - alpha) + moved[:3] * alpha

        if scene == 2:
            # Animated concentric ripples make the landing read as an action.
            landing = max(0.0, (progress - 0.55) / 0.45)
            if landing > 0:
                radius_x = 18.0 + landing * 72.0
                radius_y = 5.0 + landing * 17.0
                distance = torch.abs(
                    ((xx - self.width * 0.5) / radius_x) ** 2
                    + ((yy - self.height * 0.79) / radius_y) ** 2 - 1.0
                )
                ring = (1.0 - distance * 20.0).clamp(0, 1) * (1.0 - landing) * 0.75
                color = torch.tensor((0.86, 0.96, 1.0), device=frame.device).view(3, 1, 1)
                frame = frame * (1 - ring.unsqueeze(0)) + color * ring.unsqueeze(0)
        elif scene == 3:
            warmth = 0.05 + 0.06 * progress
            glow = torch.tensor((1.0, 0.86, 0.48), device=frame.device).view(3, 1, 1)
            frame = frame * (1 - warmth) + glow * warmth
        elif scene == 4:
            # Small moving highlights direct the eye from Mia's finger to the rainbow.
            for offset in (0.0, 0.33, 0.66):
                phase = (progress + offset) % 1.0
                cx = self.width * (0.54 + 0.34 * phase)
                cy = self.height * (0.36 - 0.22 * phase)
                sparkle = (
                    ((xx - cx) ** 2 + (yy - cy) ** 2)
                    < (2.0 + 3.0 * (1 - phase)) ** 2
                ).float()
                frame = frame * (1 - sparkle.unsqueeze(0) * 0.7) + sparkle.unsqueeze(0) * 0.7
        return frame.clamp(0, 1)

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

        device = torch.device(f"cuda:{gpu_id}")
        layered = self.has_layered_assets(segment)
        if layered:
            background, foreground, scene = self._load_layers(path, device)
            image = None
        else:
            with Image.open(path) as source:
                fitted = ImageOps.fit(
                    source.convert("RGB"), (self.width, self.height),
                    method=Image.Resampling.LANCZOS,
                )
                pixels = np.asarray(fitted).copy()
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
                if layered:
                    frame = self._layered_frame(
                        torch, functional, background, foreground, scene, progress, xx, yy
                    )
                else:
                    # Gentle push-in plus a tiny horizontal drift. This transforms the
                    # complete frame uniformly, so proportions cannot mutate.
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
