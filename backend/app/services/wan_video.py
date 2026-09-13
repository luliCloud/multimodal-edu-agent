"""Wan2.1 text-to-video backend; the model is loaded on first GPU task."""

import hashlib
import logging
import os
import re
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact

logger = logging.getLogger(__name__)


class WanVideoGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.height = int(os.getenv("WAN_HEIGHT", "320"))
        self.width = int(os.getenv("WAN_WIDTH", "576"))
        self.num_frames = int(os.getenv("WAN_NUM_FRAMES", "33"))
        self.steps = int(os.getenv("WAN_STEPS", "20"))
        self.fps = int(os.getenv("WAN_FPS", "8"))
        if self.height % 16 or self.width % 16 or (self.num_frames - 1) % 4:
            raise ValueError("Wan needs dimensions divisible by 16 and frames = 4*k+1")
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from diffusers import AutoencoderKLWan, WanPipeline
            from diffusers.schedulers import UniPCMultistepScheduler

            logger.info("Loading Wan model %s", self.model_id)
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=torch.float32
            )
            pipeline = WanPipeline.from_pretrained(
                self.model_id, vae=vae, torch_dtype=torch.bfloat16
            )
            pipeline.scheduler = UniPCMultistepScheduler.from_config(
                pipeline.scheduler.config, flow_shift=5.0
            )
            pipeline.enable_model_cpu_offload()
            self._pipeline = pipeline
            logger.info("Wan model loaded")
        return self._pipeline

    @staticmethod
    def build_prompt(segment: SegmentRequest) -> str:
        concepts = ", ".join(segment.keywords)
        visual_text = re.sub(r'[“"][^”"]+[”"]\s*said[^.]*\.', '', segment.text)
        visual_text = re.sub(r"\s+", " ", visual_text).strip()
        prompt = (
            "A short 2D watercolor nature animation, soft natural lighting, balanced earthy colors, "
            "gentle motion, one continuous shot, clear visual storytelling. "
            f"Scene: {visual_text} "
        )
        if concepts:
            prompt += f"Key visual elements: {concepts}. "
        return prompt.strip()

    def generate(self, segment_id: str, segment: SegmentRequest, gpu_id: int) -> VideoArtifact:
        import torch
        from diffusers.utils import export_to_video

        if not torch.cuda.is_available():
            raise RuntimeError("Wan backend selected but no GPU is visible")
        pipeline = self._load_pipeline()
        seed = int(hashlib.sha256(segment.text.encode()).hexdigest()[:8], 16)
        logger.info("Generating %s on GPU %d with %d steps", segment_id, gpu_id, self.steps)
        with torch.inference_mode():
            frames = pipeline(
                prompt=self.build_prompt(segment),
                negative_prompt=(
                    "on-screen text, letters, words, captions, title card, subtitles, "
                    "logo, watermark, overexposed, washed out, blurry, distorted, low quality"
                ),
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.steps,
                guidance_scale=5.0,
                generator=torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed),
            ).frames[0]
        path = self.output_dir / f"{segment_id}.mp4"
        export_to_video(frames, str(path), fps=self.fps)
        return VideoArtifact(
            segment_id=segment_id, path=str(path), media_type="video/mp4",
            duration_seconds=round(self.num_frames / self.fps), gpu_id=gpu_id,
        )
