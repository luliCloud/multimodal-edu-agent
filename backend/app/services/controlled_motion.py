"""CUDA-rendered shots for motions that require exact direction and anatomy."""

import math
import subprocess
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact


class CudaControlledMotionGenerator:
    def __init__(self, output_dir: Path, width: int, height: int,
                 num_frames: int, fps: int) -> None:
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps

    def generate(self, segment_id: str, segment: SegmentRequest, gpu_id: int) -> VideoArtifact:
        import imageio_ffmpeg
        import torch

        source = segment.text.lower()
        if not torch.cuda.is_available():
            raise RuntimeError("Controlled animation needs a visible CUDA GPU")
        if not any(term in source for term in
                   ("seed", "rain", "root", "stem", "leaves", "flower", "bee")):
            raise ValueError("Controlled animation needs a plant lifecycle scene")
        device = torch.device(f"cuda:{gpu_id}")
        yy, xx = torch.meshgrid(torch.arange(self.height, device=device, dtype=torch.float32),
                                torch.arange(self.width, device=device, dtype=torch.float32),
                                indexing="ij")
        raw = bytearray()
        with torch.inference_mode():
            for index in range(self.num_frames):
                progress = index / max(self.num_frames - 1, 1)
                if "new seeds" in source:
                    frame = self._new_seeds_frame(torch, xx, yy, progress)
                elif "bee" in source:
                    frame = self._bee_frame(torch, xx, yy, progress, index)
                elif "flower" in source:
                    frame = self._flower_frame(torch, xx, yy, progress)
                elif "root" in source and "stem" not in source:
                    frame = self._root_frame(torch, xx, yy, progress)
                elif "leaves" in source and "flower" not in source:
                    frame = self._leaves_frame(torch, xx, yy, progress)
                elif "stem" in source:
                    frame = self._stem_frame(torch, xx, yy, progress)
                elif "rain" in source:
                    frame = self._rain_seed_frame(torch, xx, yy, progress)
                else:
                    frame = self._seed_frame(torch, xx, yy, progress)
                raw.extend((frame.permute(1, 2, 0).clamp(0, 1) * 255)
                           .to(torch.uint8).cpu().numpy().tobytes())
        output = self.output_dir / f"{segment_id}.mp4"
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
             "-r", str(self.fps), "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(output)],
            input=raw, capture_output=True, timeout=120, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Controlled MP4 encoding failed: {result.stderr.decode(errors='replace')[-500:]}")
        return VideoArtifact(segment_id=segment_id, path=str(output), media_type="video/mp4",
                             duration_seconds=round(self.num_frames / self.fps), gpu_id=gpu_id)

    @staticmethod
    def _blend(torch, frame, alpha, color):
        a = alpha.clamp(0, 1).unsqueeze(0)
        return frame * (1 - a) + torch.tensor(color, device=frame.device).view(3, 1, 1) * a

    @staticmethod
    def _ellipse(torch, xx, yy, cx, cy, rx, ry, angle=0.0):
        cosine, sine = math.cos(angle), math.sin(angle)
        dx, dy = xx - cx, yy - cy
        px = dx * cosine + dy * sine
        py = -dx * sine + dy * cosine
        radius = torch.sqrt((px / rx) ** 2 + (py / ry) ** 2)
        return ((1.02 - radius) * 18).clamp(0, 1)

    @staticmethod
    def _line(torch, xx, yy, x1, y1, x2, y2, width):
        dx, dy = x2 - x1, y2 - y1
        projection = (((xx - x1) * dx + (yy - y1) * dy) /
                      max(dx * dx + dy * dy, 1)).clamp(0, 1)
        distance = torch.sqrt((xx - x1 - projection * dx) ** 2 +
                              (yy - y1 - projection * dy) ** 2)
        return (width + 0.5 - distance).clamp(0, 1)

    def _root_frame(self, torch, xx, yy, progress):
        frame = torch.empty((3, self.height, self.width), device=xx.device)
        frame[:] = torch.tensor((0.97, 0.95, 0.87), device=xx.device).view(3, 1, 1)
        soil_top = self.height * 0.25 + 5 * torch.sin(xx / 35)
        frame = self._blend(torch, frame, (yy >= soil_top).float(), (0.34, 0.22, 0.14))
        texture = (torch.sin(xx * 0.17 + yy * 0.13) * torch.cos(yy * 0.21 - xx * 0.08)) * 0.018
        frame = (frame + texture.unsqueeze(0) * (yy >= soil_top).float()).clamp(0, 1)
        cx, cy = self.width * 0.5, self.height * 0.43
        # A single root visibly extends downward from the seed; it never grows upward.
        points = [(cx + 13 * math.sin(2 * t), cy + 19 + self.height * 0.36 * t)
                  for t in [i / 30 for i in range(31)]]
        visible = max(1, int(progress * 30))
        segments = list(zip(points[:visible], points[1:visible + 1]))
        for start, end in segments:
            outline = self._line(torch, xx, yy, *start, *end, 4.5)
            frame = self._blend(torch, frame, outline, (0.17, 0.11, 0.08))
        for start, end in segments:
            inside = self._line(torch, xx, yy, *start, *end, 2.8)
            frame = self._blend(torch, frame, inside, (0.96, 0.88, 0.67))
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, cx, cy, 24, 18),
                            (0.17, 0.11, 0.08))
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, cx, cy, 21, 15),
                            (0.64, 0.37, 0.16))
        return frame

    def _seed_frame(self, torch, xx, yy, progress):
        frame = torch.empty((3, self.height, self.width), device=xx.device)
        frame[:] = torch.tensor((0.96, 0.98, 0.90), device=xx.device).view(3, 1, 1)
        soil_top = self.height * 0.43 + 4 * torch.sin(xx / 37)
        frame = self._blend(torch, frame, (yy >= soil_top).float(), (0.43, 0.28, 0.17))
        texture = torch.sin(xx * 0.13 + yy * 0.09) * torch.cos(yy * 0.17) * 0.018
        frame = (frame + texture.unsqueeze(0) * (yy >= soil_top).float()).clamp(0, 1)
        cx = self.width * 0.5
        cy = self.height * (0.54 + 0.012 * progress)
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, cx, cy, 31, 22),
                            (0.18, 0.11, 0.07))
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, cx, cy, 27, 18),
                            (0.66, 0.39, 0.17))
        return frame

    def _rain_seed_frame(self, torch, xx, yy, progress):
        frame = torch.empty((3, self.height, self.width), device=xx.device)
        frame[:] = torch.tensor((0.84, 0.90, 0.91), device=xx.device).view(3, 1, 1)
        soil_top = self.height * 0.62 + 4 * torch.sin(xx / 35)
        frame = self._blend(torch, frame, (yy >= soil_top).float(), (0.34, 0.23, 0.16))
        cx, cy = self.width * 0.5, self.height * 0.66
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, cx, cy, 27, 19),
                            (0.18, 0.11, 0.07))
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, cx, cy, 24, 16),
                            (0.62, 0.36, 0.16))
        for index in range(12):
            drop_x = self.width * (0.08 + 0.075 * index)
            drop_y = (self.height * (0.06 + 0.13 * (index % 5)) +
                      progress * self.height * 0.48) % (self.height * 0.60)
            drop = self._line(torch, xx, yy, drop_x, drop_y,
                              drop_x - 3, drop_y + 15, 1.8)
            frame = self._blend(torch, frame, drop, (0.25, 0.60, 0.80))
        return frame

    def _bee_frame(self, torch, xx, yy, progress, index):
        frame = torch.empty((3, self.height, self.width), device=xx.device)
        frame[:] = torch.tensor((0.95, 0.98, 0.88), device=xx.device).view(3, 1, 1)
        ground = self.height * 0.88 + 3 * torch.sin(xx / 41)
        frame = self._blend(torch, frame, (yy >= ground).float(), (0.70, 0.52, 0.31))
        fx, fy = self.width * 0.77, self.height * 0.50
        frame = self._blend(torch, frame, self._line(torch, xx, yy, fx, fy + 10,
                                                       fx, ground.mean().item(), 3),
                            (0.22, 0.47, 0.22))
        for petal in range(8):
            angle = petal * math.tau / 8
            px, py = fx + 27 * math.cos(angle), fy + 27 * math.sin(angle)
            frame = self._blend(torch, frame,
                                self._ellipse(torch, xx, yy, px, py, 23, 13, angle),
                                (0.22, 0.17, 0.08))
            frame = self._blend(torch, frame,
                                self._ellipse(torch, xx, yy, px, py, 21, 11, angle),
                                (0.98, 0.79, 0.14))
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, fx, fy, 13, 13),
                            (0.56, 0.31, 0.11))
        bx = self.width * (0.12 + 0.60 * progress)
        by = fy - 11 + 6 * math.sin(progress * math.tau * 2)
        wing_flap = math.sin(index * math.tau / 4)
        frame = self._blend(torch, frame,
                            self._ellipse(torch, xx, yy, bx - 4, by - 16 - wing_flap * 4,
                                          15, 8, -0.45), (0.80, 0.92, 0.96))
        frame = self._blend(torch, frame,
                            self._ellipse(torch, xx, yy, bx + 6, by - 17 + wing_flap * 4,
                                          15, 8, 0.45), (0.87, 0.95, 0.97))
        body = self._ellipse(torch, xx, yy, bx - 8, by, 23, 13)
        frame = self._blend(torch, frame, body, (0.13, 0.12, 0.09))
        frame = self._blend(torch, frame,
                            self._ellipse(torch, xx, yy, bx - 9, by, 20, 10),
                            (0.96, 0.72, 0.08))
        for stripe_x in (-20, -9, 2):
            stripe = body * (torch.abs(xx - (bx + stripe_x)) < 2.8).float()
            frame = self._blend(torch, frame, stripe, (0.12, 0.11, 0.08))
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, bx + 14, by, 11, 10),
                            (0.12, 0.11, 0.08))
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, bx + 18, by - 2, 2, 2),
                            (0.97, 0.96, 0.88))
        for offset in (-8, 6):
            frame = self._blend(torch, frame,
                                self._line(torch, xx, yy, bx + offset, by + 10,
                                           bx + offset + 4, by + 20, 1.5),
                                (0.13, 0.11, 0.08))
        return frame

    def _plant_background(self, torch, xx, yy):
        frame = torch.empty((3, self.height, self.width), device=xx.device)
        frame[:] = torch.tensor((0.96, 0.98, 0.90), device=xx.device).view(3, 1, 1)
        soil_top = self.height * 0.84 + 3 * torch.sin(xx / 43)
        frame = self._blend(torch, frame, (yy >= soil_top).float(), (0.56, 0.38, 0.23))
        return frame

    def _stem_frame(self, torch, xx, yy, progress):
        frame = self._plant_background(torch, xx, yy)
        sun_x, sun_y = self.width * 0.78, self.height * 0.18
        frame = self._blend(torch, frame,
                            self._ellipse(torch, xx, yy, sun_x, sun_y, 22, 22),
                            (0.97, 0.79, 0.21))
        base_x, base_y = self.width * 0.5, self.height * 0.84
        tip_y = base_y - self.height * (0.06 + 0.40 * progress)
        frame = self._blend(torch, frame,
                            self._line(torch, xx, yy, base_x, base_y, base_x + 7,
                                       tip_y, 5), (0.12, 0.32, 0.13))
        frame = self._blend(torch, frame,
                            self._line(torch, xx, yy, base_x, base_y, base_x + 7,
                                       tip_y, 3), (0.30, 0.68, 0.22))
        frame = self._blend(torch, frame,
                            self._ellipse(torch, xx, yy, base_x + 7, tip_y, 5, 8),
                            (0.22, 0.54, 0.17))
        return frame

    def _leaves_frame(self, torch, xx, yy, progress):
        frame = self._plant_background(torch, xx, yy)
        cx, base_y = self.width * 0.5, self.height * 0.84
        frame = self._blend(torch, frame,
                            self._line(torch, xx, yy, cx, base_y, cx, base_y - 136, 4),
                            (0.18, 0.48, 0.17))
        leaves = [
            (-28, -65, -0.45, 1.0), (28, -73, 0.45, 1.0),
            (-24, -112, -0.50, max(0.0, min(1.0, (progress - 0.28) / 0.42))),
            (24, -121, 0.50, max(0.0, min(1.0, (progress - 0.44) / 0.42))),
        ]
        for dx, dy, angle, scale in leaves:
            if scale <= 0:
                continue
            leaf_x, leaf_y = cx + dx * scale, base_y + dy
            frame = self._blend(torch, frame,
                                self._line(torch, xx, yy, cx, leaf_y + 6,
                                           leaf_x, leaf_y, 2), (0.17, 0.43, 0.16))
            outer = self._ellipse(torch, xx, yy, leaf_x, leaf_y,
                                  max(2, 25 * scale), max(2, 11 * scale), angle)
            frame = self._blend(torch, frame, outer, (0.12, 0.37, 0.14))
            inner = self._ellipse(torch, xx, yy, leaf_x, leaf_y,
                                  max(1, 22 * scale), max(1, 9 * scale), angle)
            frame = self._blend(torch, frame, inner, (0.32, 0.70, 0.25))
        return frame

    def _flower_frame(self, torch, xx, yy, progress):
        frame = self._leaves_frame(torch, xx, yy, 1.0)
        cx, base_y = self.width * 0.5, self.height * 0.84
        flower_y = base_y - 158
        scale = max(0.08, progress)
        for petal in range(8):
            angle = petal * math.tau / 8
            px = cx + 25 * scale * math.cos(angle)
            py = flower_y + 25 * scale * math.sin(angle)
            outer = self._ellipse(torch, xx, yy, px, py,
                                  max(2, 21 * scale), max(2, 11 * scale), angle)
            frame = self._blend(torch, frame, outer, (0.93, 0.65, 0.08))
            inner = self._ellipse(torch, xx, yy, px, py,
                                  max(1, 18 * scale), max(1, 9 * scale), angle)
            frame = self._blend(torch, frame, inner, (0.99, 0.84, 0.20))
        center = self._ellipse(torch, xx, yy, cx, flower_y, max(2, 11 * scale),
                               max(2, 11 * scale))
        return self._blend(torch, frame, center, (0.55, 0.30, 0.10))

    def _new_seeds_frame(self, torch, xx, yy, progress):
        frame = torch.empty((3, self.height, self.width), device=xx.device)
        frame[:] = torch.tensor((0.96, 0.98, 0.90), device=xx.device).view(3, 1, 1)
        cx, cy = self.width * 0.5, self.height * 0.48
        for petal in range(10):
            angle = petal * math.tau / 10
            px, py = cx + 78 * math.cos(angle), cy + 78 * math.sin(angle)
            frame = self._blend(
                torch, frame, self._ellipse(torch, xx, yy, px, py, 66, 28, angle),
                (0.98, 0.80, 0.16),
            )
        frame = self._blend(torch, frame, self._ellipse(torch, xx, yy, cx, cy, 55, 55),
                            (0.48, 0.27, 0.10))
        scale = max(0.05, progress)
        for dx, dy, angle in ((-23, -18, -0.4), (20, -15, 0.5),
                              (-18, 20, 0.4), (23, 22, -0.5), (2, 4, 0.1)):
            frame = self._blend(
                torch, frame,
                self._ellipse(torch, xx, yy, cx + dx, cy + dy,
                              max(2, 11 * scale), max(2, 16 * scale), angle),
                (0.73, 0.39, 0.15),
            )
        return frame
