"""Wan2.1 text-to-video backend; the model is loaded on first GPU task."""

import hashlib
import gc
import logging
import os
import re
from pathlib import Path

from backend.app.models.jobs import SegmentRequest, VideoArtifact
from backend.app.services.controlled_motion import CudaControlledMotionGenerator
from backend.app.services.keyframe_motion import CudaKeyframeMotionGenerator

logger = logging.getLogger(__name__)


class WanVideoGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.height = int(os.getenv("WAN_HEIGHT", "576"))
        self.width = int(os.getenv("WAN_WIDTH", "320"))
        self._frames_explicit = "WAN_NUM_FRAMES" in os.environ
        self.num_frames = int(os.getenv("WAN_NUM_FRAMES", "33"))
        self.steps = int(os.getenv("WAN_STEPS", "20"))
        self.fps = int(os.getenv("WAN_FPS", "9"))
        if self.height % 16 or self.width % 16 or (self.num_frames - 1) % 4:
            raise ValueError("Wan needs dimensions divisible by 16 and frames = 4*k+1")
        self._pipeline = None
        self._reference_pipeline = None
        self.reference_model_id = os.getenv(
            "WAN_REFERENCE_MODEL_ID", "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        )
        self.reference_scale = float(os.getenv("WAN_REFERENCE_SCALE", "1.0"))
        self._controlled = CudaControlledMotionGenerator(
            output_dir, self.width, self.height, self.num_frames, self.fps
        )
        self._keyframe_motion = CudaKeyframeMotionGenerator(
            output_dir, self.width, self.height, self.num_frames, self.fps
        )

    @staticmethod
    def frames_for_scene_count(scene_count: int, fps: int, seconds: int = 15) -> int:
        target = seconds * fps / max(scene_count, 1)
        return max(5, 4 * round((target - 1) / 4) + 1)

    def fit_scene_count(self, scene_count: int) -> None:
        if self._frames_explicit:
            return
        self.num_frames = self.frames_for_scene_count(scene_count, self.fps)
        self._controlled.num_frames = self.num_frames
        self._keyframe_motion.num_frames = self.num_frames

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

    def _load_reference_pipeline(self):
        if self._reference_pipeline is None:
            import torch
            from diffusers import AutoencoderKLWan, WanVACEPipeline
            from diffusers.schedulers import UniPCMultistepScheduler

            logger.info("Loading Wan reference model %s", self.reference_model_id)
            vae = AutoencoderKLWan.from_pretrained(
                self.reference_model_id, subfolder="vae", torch_dtype=torch.float32
            )
            pipeline = WanVACEPipeline.from_pretrained(
                self.reference_model_id, vae=vae, torch_dtype=torch.bfloat16
            )
            pipeline.scheduler = UniPCMultistepScheduler.from_config(
                pipeline.scheduler.config, flow_shift=5.0
            )
            pipeline.enable_model_cpu_offload()
            self._reference_pipeline = pipeline
            logger.info("Wan visual-reference model loaded")
        return self._reference_pipeline

    def _release_text_pipeline(self) -> None:
        """Do not retain both model pipelines after making a one-time reference."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

    def _ensure_reference_image(self, segment: SegmentRequest, gpu_id: int) -> Path:
        if not segment.reference_image:
            raise ValueError("A visual reference path is required")
        path = Path(segment.reference_image)
        if path.is_file():
            return path
        if not segment.reference_prompt:
            raise FileNotFoundError(
                f"Character reference does not exist and has no generation prompt: {path}"
            )

        import numpy as np
        import torch
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = self._load_pipeline()
        seed_source = segment.reference_id or segment.reference_prompt
        seed = int(hashlib.sha256(seed_source.encode()).hexdigest()[:8], 16)
        logger.info("Drawing canonical character reference once -> %s", path)
        prompt = (
            segment.reference_prompt
            + " This is a still identity asset, not a scene. Keep the pose neutral and static."
        )
        with torch.inference_mode():
            frame = pipeline(
                prompt=prompt,
                negative_prompt=(
                    "multiple people, multiple views, character sheet, panels, text, labels, "
                    "logo, watermark, cropped body, cropped feet, props, scenery, action pose, "
                    "distorted face, extra limbs, low quality"
                ),
                height=self.height,
                width=self.width,
                num_frames=5,
                num_inference_steps=self.steps,
                guidance_scale=5.0,
                generator=torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed),
            ).frames[0][0]
        pixels = np.asarray(frame)
        if pixels.dtype != np.uint8:
            pixels = (pixels.clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(path)
        self._release_text_pipeline()
        return path

    @staticmethod
    def build_reference_prompt(segment: SegmentRequest) -> str:
        """Describe only the scene; visual identity comes from the attached image."""
        visual = segment.visual_prompt or segment.text
        visual = re.sub(
            r"GLOBAL_CHARACTER:.*?(?=GLOBAL_STYLE:)", "", visual,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        keyframe_instruction = (
            "Begin from the attached scene keyframe exactly, preserving its composition and "
            "character design; animate it naturally without replacing the person. "
            if segment.keyframe_image else ""
        )
        prompt = (
            "Use the person in the attached reference image as the exact main character. "
            "Preserve the same face, facial proportions, skin tone, hairstyle, body build, "
            "height proportions, and illustration style throughout the entire shot. Do not "
            "redesign or reinterpret the person. "
            f"{keyframe_instruction}Scene: {visual} "
        )
        if segment.motion:
            prompt += f"Motion from beginning to end: {segment.motion} "
        return prompt + (
            "One continuous portrait 9:16 shot. Keep the complete character inside the frame."
        )

    @staticmethod
    def build_prompt(segment: SegmentRequest) -> str:
        visual_text = re.sub(r'[“"][^”"]+[”"]\s*said[^.]*\.', '', segment.text)
        visual_text = re.sub(r"\s+", " ", visual_text).strip()
        source = segment.text.lower()
        seed_preset = os.getenv("WAN_CONTROLLED_MOTION", "1") == "1"
        seed_before_growth = seed_preset and "seed" in source and "bee" not in source and not any(
            stage in source for stage in ("root", "stem", "leaves", "flower")
        )
        rain_before_growth = seed_before_growth and "rain" in source
        keywords = ([keyword for keyword in segment.keywords if "sun" not in keyword.lower()]
                    if rain_before_growth else segment.keywords)
        concepts = ", ".join(keywords)
        visual = segment.visual_prompt or visual_text
        if seed_preset and "new seeds" in source and "flower" in source:
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
        elif seed_preset and "root" in source and "stem" not in source:
            visual = ("Macro cutaway cross-section of dark soil. Exactly one small brown "
                      "seed underground, with a single thin pale root emerging from its "
                      "underside and extending downward. Empty ground surface above; "
                      "no trees, leaves, stems, or other plants.")
        elif seed_preset and "stem" in source and "leaves" not in source:
            visual = ("Exactly one slender green stem emerging upward through brown soil "
                      "from a single seed. Its tip is still closed. No leaves, flowers, "
                      "trees, or other plants yet.")
        elif seed_preset and "leaves" in source and "flower" not in source:
            visual = ("Exactly one young green plant rooted in brown soil. It starts with "
                      "two leaves attached to its stem; two more leaves grow from the stem "
                      "until four leaves are visibly attached. No flowers or floating leaves.")
        if seed_preset and "bee" in source and segment.visual_prompt and "flower" in visual.lower():
            visual = ("Exactly one fully open yellow flower on the right side of the frame. "
                      "One recognizable yellow-and-black striped honeybee approaches from "
                      "the left in side profile. Its black head is on its right side facing "
                      "the flower; its striped abdomen is behind on the left. "
                      "No other flowers, buds, or insects.")
        prompt = (
            "A hand-painted 2D watercolor children's storybook animation with fine ink outlines "
            "and colors faithful to the story. Keep recurring subjects visually "
            "consistent, but show only the current stage of the story. "
            "One continuous portrait 9:16 shot with visibly animated motion. Keep the entire "
            "main subject inside the frame with generous space on every side; no cropped body. "
            f"Scene: {visual} "
        )
        motion = ("The camera slowly pushes closer while a few grains of soil settle; "
                  "the seed itself stays intact." if seed_before_growth and not rain_before_growth
                  else segment.motion)
        if motion:
            prompt += f"Motion from beginning to end: {motion} "
        else:
            prompt += "The described action visibly progresses from beginning to end. "
        if seed_preset and "bee" in source:
            prompt += ("Show the bee in side profile flying toward the flower, its head facing "
                       "the flower and its abdomen trailing behind; its wings beat visibly. "
                       "The bee travels across at least one third of the frame during the shot. ")
        if rain_before_growth:
            prompt += ("Many small translucent pale-blue raindrops fall visibly from above "
                       "and make gentle splashes on the dark soil. ")
        if concepts and not segment.visual_prompt:
            prompt += f"Key visual elements: {concepts}. "
        return prompt.strip()

    @staticmethod
    def uses_controlled_motion(segment: SegmentRequest) -> bool:
        if os.getenv("WAN_CONTROLLED_MOTION", "1") != "1":
            return False
        source = segment.text.lower()
        visual = (segment.visual_prompt or "").lower()
        explicit_lifecycle = any(
            term in source for term in ("seed", "root", "stem", "leaves", "new seeds")
        )
        referenced_plant = "global_character:" in visual and "; plant;" in visual
        return explicit_lifecycle or (referenced_plant and any(
            term in source for term in ("flower", "bee", "rain")
        ))

    def generate(self, segment_id: str, segment: SegmentRequest, gpu_id: int) -> VideoArtifact:
        source = segment.text.lower()
        if self.uses_controlled_motion(segment):
            return self._controlled.generate(segment_id, segment, gpu_id)
        if (segment.keyframe_image and
                os.getenv("WAN_KEYFRAME_MOTION", "stable").lower() == "stable"):
            logger.info(
                "Animating reviewed keyframe without redrawing identity: %s",
                segment.keyframe_image,
            )
            return self._keyframe_motion.generate(segment_id, segment, gpu_id)
        import torch
        from diffusers.utils import export_to_video

        if not torch.cuda.is_available():
            raise RuntimeError("Wan backend selected but no GPU is visible")
        uses_reference = bool(segment.reference_image)
        reference_path = (
            self._ensure_reference_image(segment, gpu_id) if uses_reference else None
        )
        pipeline = (
            self._load_reference_pipeline() if uses_reference else self._load_pipeline()
        )
        seed = int(hashlib.sha256(segment.text.encode()).hexdigest()[:8], 16)
        logger.info("Generating %s on GPU %d with %d steps", segment_id, gpu_id, self.steps)
        with torch.inference_mode():
            negative_prompt = (
                "on-screen text, letters, words, captions, title card, subtitles, "
                "logo, watermark, frozen frame, static image, "
                "cropped subject, cut-off body, anthropomorphic seed, anthropomorphic plant, "
                "flower with a face, flower with eyes, arms, legs, humanoid plant, extra limbs, "
                "overexposed, washed out, blurry, distorted, low quality"
            )
            source = segment.text.lower()
            seed_preset = os.getenv("WAN_CONTROLLED_MOTION", "1") == "1"
            seed_before_growth = seed_preset and "seed" in source and "bee" not in source and not any(
                stage in source for stage in ("root", "stem", "leaves", "flower")
            )
            if seed_before_growth:
                negative_prompt += ", sprout, stem, leaves, flower, plant"
            if seed_before_growth and "rain" in source:
                negative_prompt += ", bright sun, black raindrops, thick ink lines"
            if seed_preset and "root" in source and "stem" not in source:
                negative_prompt += ", tree, trunk, forest, leaves, stem, multiple plants, flowers"
            if seed_preset and "stem" in source and "leaves" not in source:
                negative_prompt += ", leaves, flowers, tree, forest, multiple plants"
            if seed_preset and "leaves" in source and "flower" not in source:
                negative_prompt += ", floating leaves, detached leaves, extra plants, flowers"
            if seed_preset and "new seeds" in source and "flower" in source:
                negative_prompt += ", empty black flower center, featureless dark disk, no seeds"
            if seed_preset and "bee" in source:
                negative_prompt += ", extra flowers, second flower, bud, housefly, green fly"
            call = dict(
                prompt=(self.build_reference_prompt(segment)
                        if uses_reference else self.build_prompt(segment)),
                negative_prompt=negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.steps,
                guidance_scale=5.0,
                generator=torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed),
            )
            if uses_reference:
                from PIL import Image, ImageOps
                logger.info(
                    "Generating %s with visual reference %s (%s)",
                    segment_id, segment.reference_id, reference_path,
                )
                with Image.open(reference_path) as image:
                    call["reference_images"] = [image.convert("RGB")]
                    call["conditioning_scale"] = self.reference_scale
                    if segment.keyframe_image:
                        keyframe_path = Path(segment.keyframe_image)
                        if not keyframe_path.is_file():
                            raise FileNotFoundError(f"Scene keyframe does not exist: {keyframe_path}")
                        with Image.open(keyframe_path) as source_keyframe:
                            keyframe = ImageOps.fit(
                                source_keyframe.convert("RGB"),
                                (self.width, self.height),
                                method=Image.Resampling.LANCZOS,
                            )
                        gray = Image.new("RGB", (self.width, self.height), (128, 128, 128))
                        call["video"] = [keyframe] + [gray] * (self.num_frames - 1)
                        black = Image.new("L", (self.width, self.height), 0)
                        white = Image.new("L", (self.width, self.height), 255)
                        call["mask"] = [black] + [white] * (self.num_frames - 1)
                        logger.info("Fixing scene start to keyframe %s", keyframe_path)
                    frames = pipeline(**call).frames[0]
            else:
                frames = pipeline(**call).frames[0]
        path = self.output_dir / f"{segment_id}.mp4"
        export_to_video(frames, str(path), fps=self.fps)
        return VideoArtifact(
            segment_id=segment_id, path=str(path), media_type="video/mp4",
            duration_seconds=round(self.num_frames / self.fps), gpu_id=gpu_id,
        )
