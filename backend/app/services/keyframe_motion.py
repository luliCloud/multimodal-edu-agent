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
        def ease(value: float) -> float:
            value = max(0.0, min(1.0, value))
            return value * value * (3.0 - 2.0 * value)

        def rotate_part(source, region, pivot_x, pivot_y, angle):
            """Rotate one body region around a joint while keeping the torso fixed."""
            cosine, sine = math.cos(angle), math.sin(angle)
            dx, dy = xx - pivot_x, yy - pivot_y
            input_x = cosine * dx + sine * dy + pivot_x
            input_y = -sine * dx + cosine * dy + pivot_y
            grid = torch.stack((
                input_x / max(self.width - 1, 1) * 2 - 1,
                input_y / max(self.height - 1, 1) * 2 - 1,
            ), dim=-1).unsqueeze(0)
            part = source * region
            moved = functional.grid_sample(
                part.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                align_corners=True,
            )[0]
            moved_region = functional.grid_sample(
                region.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                align_corners=True,
            )[0]
            return (source * (1 - region) + moved * moved_region).clamp(0, 1)

        animated = foreground
        if scene == 2:
            # Anticipation -> takeoff -> airborne arc -> landing squash.
            if progress < 0.16:
                amount = math.sin(progress / 0.16 * math.pi / 2)
                scale_x, scale_y, dy = 1 + 0.08 * amount, 1 - 0.09 * amount, 12 * amount
            elif progress < 0.78:
                flight = (progress - 0.16) / 0.62
                jump = math.sin(flight * math.pi)
                scale_x, scale_y, dy = 1 - 0.025 * jump, 1 + 0.035 * jump, -72 * jump
            else:
                impact = math.sin((progress - 0.78) / 0.22 * math.pi)
                scale_x, scale_y, dy = 1 + 0.10 * impact, 1 - 0.12 * impact, 14 * impact
            dx = 0.0
            angle = 0.0
            cosine, sine = 1.0, 0.0
            theta = torch.tensor(
                [[[cosine / scale_x, -sine, -2.0 * dx / self.width],
                  [sine, cosine / scale_y, -2.0 * dy / self.height]]],
                device=background.device, dtype=torch.float32,
            )
            source = animated.unsqueeze(0)
            grid = functional.affine_grid(theta, source.shape, align_corners=False)
            animated = functional.grid_sample(
                source, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )[0]
        elif scene == 3:
            # Mia deliberately raises her gaze as the sunlight returns.
            head_lift = -0.065 * ease(progress / 0.55)
            region = (
                (yy < self.height * 0.38)
                & (xx > self.width * 0.20)
                & (xx < self.width * 0.80)
            ).float().unsqueeze(0)
            animated = rotate_part(
                animated, region, self.width * 0.50, self.height * 0.38, head_lift
            )
        scene_background = background
        if scene == 4:
            # Reveal the colorful arc while the reviewed pointing pose stays intact.
            saturation = background.max(dim=0).values - background.min(dim=0).values
            upper_sky = (yy < self.height * 0.39).float()
            rainbow_mask = (saturation > 0.24).float() * upper_sky
            visibility = ease(progress / 0.68)
            sky = torch.tensor(
                (0.62, 0.82, 0.96), device=background.device
            ).view(3, 1, 1)
            hidden = rainbow_mask.unsqueeze(0) * (1.0 - visibility) * 0.88
            scene_background = background * (1 - hidden) + sky * hidden

        alpha = animated[3:4].clamp(0, 1)
        frame = scene_background * (1 - alpha) + animated[:3] * alpha

        if scene == 1:
            # Keep Mia's painted character layer intact. The storm develops outside the
            # window without cutting or warping her raised arm.
            storm = 0.10 * ease(progress / 0.72)
            cool_gray = torch.tensor(
                (0.42, 0.52, 0.60), device=frame.device
            ).view(3, 1, 1)
            window = (
                (xx > self.width * 0.48)
                & (yy < self.height * 0.68)
            ).float().unsqueeze(0)
            frame = frame * (1 - window * storm) + cool_gray * window * storm
        elif scene == 2:
            # A grounded shadow shrinks during flight and expands again on landing.
            airborne = max(0.0, -dy / 72.0)
            shadow_rx = 37.0 - 18.0 * airborne
            shadow_ry = 8.0 - 4.0 * airborne
            shadow_distance = (
                ((xx - self.width * 0.5) / shadow_rx) ** 2
                + ((yy - self.height * 0.865) / shadow_ry) ** 2
            )
            shadow = (1.0 - shadow_distance).clamp(0, 1) * (0.24 - 0.10 * airborne)
            frame = frame * (1 - shadow.unsqueeze(0))
            # Animated concentric ripples make the landing read as an action.
            landing = max(0.0, (progress - 0.76) / 0.24)
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
