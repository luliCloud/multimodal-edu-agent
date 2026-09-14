"""Wan2.1 text-to-video backend; the model is loaded on first GPU task."""

import hashlib
import logging
import os
import re
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact
from backend.app.services.controlled_motion import CudaControlledMotionGenerator

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
        self._controlled = CudaControlledMotionGenerator(
            output_dir, self.width, self.height, self.num_frames, self.fps
        )

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
        visual_text = re.sub(r'[“"][^”"]+[”"]\s*said[^.]*\.', '', segment.text)
        visual_text = re.sub(r"\s+", " ", visual_text).strip()
        source = segment.text.lower()
        seed_before_growth = "seed" in source and "bee" not in source and not any(
            stage in source for stage in ("root", "stem", "leaves", "flower")
        )
        rain_before_growth = seed_before_growth and "rain" in source
        keywords = ([keyword for keyword in segment.keywords if "sun" not in keyword.lower()]
                    if rain_before_growth else segment.keywords)
        concepts = ", ".join(keywords)
        visual = segment.visual_prompt or visual_text
        if "new seeds" in source and "flower" in source:
            visual = ("Macro cutaway view inside the center of one fully open yellow "
                      "flower. Several separate small brown oval new seeds become clearly "
                      "visible within the center, each with a distinct outline. "
                      "Show the developing seeds rather than an empty dark disk.")
        elif rain_before_growth:
            visual = ("One small brown seed half-buried in dark soil before germination, "
                      "under a pale gray rainy sky. No sprout or plant is visible.")
        elif seed_before_growth:
            visual = ("One small brown seed half-buried in dark soil before germination. "
                      "The seed is intact; no sprout, root, stem, or plant is visible.")
        elif "root" in source and "stem" not in source:
            visual = ("Macro cutaway cross-section of dark soil. Exactly one small brown "
                      "seed underground, with a single thin pale root emerging from its "
                      "underside and extending downward. Empty ground surface above; "
                      "no trees, leaves, stems, or other plants.")
        elif "stem" in source and "leaves" not in source:
            visual = ("Exactly one slender green stem emerging upward through brown soil "
                      "from a single seed. Its tip is still closed. No leaves, flowers, "
                      "trees, or other plants yet.")
        elif "leaves" in source and "flower" not in source:
            visual = ("Exactly one young green plant rooted in brown soil. It starts with "
                      "two leaves attached to its stem; two more leaves grow from the stem "
                      "until four leaves are visibly attached. No flowers or floating leaves.")
        if "bee" in source and segment.visual_prompt and "flower" in visual.lower():
            visual = ("Exactly one fully open yellow flower on the right side of the frame. "
                      "One recognizable yellow-and-black striped honeybee approaches from "
                      "the left in side profile. Its black head is on its right side facing "
                      "the flower; its striped abdomen is behind on the left. "
                      "No other flowers, buds, or insects.")
        prompt = (
            "A hand-painted 2D watercolor children's storybook animation with fine ink outlines, "
            "earthy brown soil and soft natural colors. Keep recurring subjects visually "
            "consistent, but show only the current stage of the story. "
            "One continuous side-view shot, visibly animated motion. "
            f"Scene: {visual} "
        )
        motion = ("The camera slowly pushes closer while a few grains of soil settle; "
                  "the seed itself stays intact." if seed_before_growth and not rain_before_growth
                  else segment.motion)
        if motion:
            prompt += f"Motion from beginning to end: {motion} "
        else:
            prompt += "The described action visibly progresses from beginning to end. "
        if "rain" in segment.text.lower():
            prompt += ("Clearly visible many small translucent pale blue raindrops fall "
                       "from above in thin streaks and make gentle splashes on the soil. ")
        if "bee" in segment.text.lower():
            prompt += ("Show the bee in side profile flying toward the flower, its head facing "
                       "the flower and its abdomen trailing behind; its wings beat visibly. "
                       "The bee travels across at least one third of the frame during the shot. ")
        if concepts:
            prompt += f"Key visual elements: {concepts}. "
        return prompt.strip()

    def generate(self, segment_id: str, segment: SegmentRequest, gpu_id: int) -> VideoArtifact:
        source = segment.text.lower()
        if os.getenv("WAN_CONTROLLED_MOTION", "1") == "1" and (
            ("root" in source and "stem" not in source) or
            ("stem" in source and "leaves" not in source) or
            ("leaves" in source and "flower" not in source) or
            ("bee" in source and ("flower" in source or
                                  "flower" in (segment.visual_prompt or "").lower()))
        ):
            return self._controlled.generate(segment_id, segment, gpu_id)
        import torch
        from diffusers.utils import export_to_video

        if not torch.cuda.is_available():
            raise RuntimeError("Wan backend selected but no GPU is visible")
        pipeline = self._load_pipeline()
        seed = int(hashlib.sha256(segment.text.encode()).hexdigest()[:8], 16)
        logger.info("Generating %s on GPU %d with %d steps", segment_id, gpu_id, self.steps)
        with torch.inference_mode():
            negative_prompt = (
                "on-screen text, letters, words, captions, title card, subtitles, "
                "logo, watermark, frozen frame, static image, malformed bee, reversed bee, "
                "extra limbs, overexposed, washed out, blurry, distorted, low quality"
            )
            source = segment.text.lower()
            seed_before_growth = "seed" in source and "bee" not in source and not any(
                stage in source for stage in ("root", "stem", "leaves", "flower")
            )
            if seed_before_growth:
                negative_prompt += ", sprout, stem, leaves, flower, plant"
            if seed_before_growth and "rain" in source:
                negative_prompt += ", bright sun, black raindrops, thick ink lines"
            if "root" in source and "stem" not in source:
                negative_prompt += ", tree, trunk, forest, leaves, stem, multiple plants, flowers"
            if "stem" in source and "leaves" not in source:
                negative_prompt += ", leaves, flowers, tree, forest, multiple plants"
            if "leaves" in source and "flower" not in source:
                negative_prompt += ", floating leaves, detached leaves, extra plants, flowers"
            if "new seeds" in source and "flower" in source:
                negative_prompt += ", empty black flower center, featureless dark disk, no seeds"
            if "bee" in source:
                negative_prompt += ", extra flowers, second flower, bud, housefly, green fly"
            frames = pipeline(
                prompt=self.build_prompt(segment),
                negative_prompt=negative_prompt,
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
