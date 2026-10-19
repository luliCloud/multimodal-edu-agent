"""Extract source-faithful visual assets from an image PDF and animate a short lesson."""

from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


WIDTH = 540
HEIGHT = 960
FPS = 12
INK = (31, 67, 123)
PAPER = (248, 246, 237)
GREEN = (105, 157, 88)
PURPLE = (143, 73, 137)
YELLOW = (238, 197, 76)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / family
    return ImageFont.truetype(str(path), size)


def _normalized_crop(image: Image.Image, box: list[float]) -> Image.Image:
    width, height = image.size
    pixels = (
        round(box[0] * width), round(box[1] * height),
        round(box[2] * width), round(box[3] * height),
    )
    return image.crop(pixels)


def render_pdf_page(pdf_path: Path, page_number: int = 1) -> Image.Image:
    """Rasterize a PDF page without depending on a system Poppler install."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    if not 1 <= page_number <= len(document):
        raise ValueError(f"Page {page_number} is outside this {len(document)}-page PDF")
    page = document[page_number - 1]
    bitmap = page.render(scale=1.5)
    return bitmap.to_pil().convert("RGB")


def extract_cell_membrane_bible(page: Image.Image, output_dir: Path,
                                 source_name: str) -> dict:
    """Extract the two teaching elements from the first quarter of this visual note.

    Coordinates are normalized and recorded in the manifest. This keeps the extraction
    reviewable now and gives a future vision planner the exact contract it must produce.
    """
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    boxes = {
        "focus_top_quarter": [0.055, 0.035, 0.975, 0.270],
        "phospholipid_bilayer": [0.485, 0.060, 0.985, 0.255],
        "ion_channel": [0.550, 0.065, 0.900, 0.245],
    }
    page.save(assets / "page_01.png")
    for name, box in boxes.items():
        _normalized_crop(page, box).save(assets / f"{name}.png")
    manifest = {
        "version": 1,
        "source": {"file": source_name, "page": 1, "image_size": list(page.size)},
        "focus": {
            "region": "first quarter",
            "topic": "cell membrane structure and selective transport",
            "bbox_normalized": boxes["focus_top_quarter"],
        },
        "style": {
            "type": "handwritten biology class notes",
            "paper": "warm ruled notebook paper",
            "ink": "blue pen with green highlights",
            "preserve_source_art": True,
        },
        "assets": [
            {
                "id": "phospholipid_bilayer",
                "path": "assets/phospholipid_bilayer.png",
                "bbox_normalized": boxes["phospholipid_bilayer"],
                "role": "show hydrophilic heads and hydrophobic tails",
            },
            {
                "id": "ion_channel",
                "path": "assets/ion_channel.png",
                "bbox_normalized": boxes["ion_channel"],
                "role": "show a selective path for ions through the membrane",
            },
        ],
    }
    (output_dir / "visual_bible.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def cell_membrane_plan() -> dict:
    narrations = [
        "Cell membranes form flexible, selective boundaries.",
        "Water-loving heads protect water-fearing tails inside.",
        "Ion channels guide selected particles across membranes.",
        "Selective transport helps cells maintain internal balance.",
    ]
    actions = [
        "Zoom from the note into the cell membrane diagram",
        "Highlight the water-facing heads and protected fatty tails",
        "Move ions through the extracted integral protein channel",
        "Contrast selected ions crossing with blocked particles",
    ]
    motions = [
        "camera pushes into the first-quarter diagram",
        "blue head rows and yellow hydrophobic core pulse in sequence",
        "purple ions travel downward through the channel",
        "selected ions pass while another particle stops at the membrane",
    ]
    sentences = [
        "The cell membrane is a selectively permeable phospholipid bilayer.",
        "Hydrophilic heads face water while hydrophobic fatty-acid tails form the core.",
        "Integral proteins can form channels that let specific ions cross.",
        "Selective transport helps the cell control its internal environment.",
    ]
    scenes = []
    for index in range(4):
        scenes.append({
            "action": actions[index],
            "location": "cell membrane class-note diagram",
            "weather": "not applicable",
            "camera": "educational diagram close-up",
            "motion": motions[index],
            "narration": narrations[index],
            "outfit_id": "source",
            "state_id": "source",
            "scene": index + 1,
            "source_sentence_ids": [index + 1],
            "text": sentences[index],
            "pages": [1],
            "keywords": ["cell membrane", "phospholipid bilayer", "ion channel"],
            "visual_prompt": f"Use extracted source-note assets. {actions[index]}.",
            "keyframe_prompt": f"Preserve source drawing exactly. {actions[index]}.",
            "narration_seconds": 3.75,
        })
    return {
        "title": "Cell Membrane: A Selective Boundary",
        "summary": (
            "A source-faithful visual lesson introduces the phospholipid bilayer and shows "
            "how ion channels support selective membrane transport."
        ),
        "character": {
            "name": "Cell membrane diagram",
            "kind": "scientific diagram",
            "visual_identity": "the exact blue, yellow, and purple diagram extracted from the note",
            "outfits": {"source": "original class-note colors and labels"},
            "states": {"source": "source drawing remains unchanged"},
            "inferred_fields": [],
        },
        "style": {
            "type": "animated handwritten study note",
            "palette": "source blue, green, yellow, and purple",
            "shapes": "source-faithful biological diagram",
            "lighting": "flat paper",
            "outline": "blue pen",
        },
        "scenes": scenes,
        "source_sentences": [
            {"id": index + 1, "page": 1, "text": text}
            for index, text in enumerate(sentences)
        ],
        "duration_seconds": 15,
    }


def _paper() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    for y in range(42, HEIGHT, 42):
        draw.line((0, y, WIDTH, y), fill=(209, 220, 224), width=1)
    draw.line((45, 0, 45, HEIGHT), fill=(226, 184, 184), width=2)
    return image


def _contain(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    left, top, right, bottom = box
    fitted = ImageOps.contain(image, (right - left, bottom - top), Image.Resampling.LANCZOS)
    return fitted, (left + (right - left - fitted.width) // 2,
                    top + (bottom - top - fitted.height) // 2)


def _rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=1, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _base_frame(title: str, narration: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    frame = _paper()
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text((67, 54), title, font=_font(29, bold=True), fill=INK)
    _rounded(draw, (58, 790, 520, 918), (255, 255, 255, 232), (31, 67, 123, 70), 2)
    words = narration.split()
    lines, line = [], []
    for word in words:
        trial = " ".join(line + [word])
        if draw.textlength(trial, font=_font(23)) > 410 and line:
            lines.append(" ".join(line)); line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    for index, text in enumerate(lines):
        draw.text((82, 818 + 31 * index), text, font=_font(23), fill=(35, 46, 62))
    return frame, draw


def _fade(frame: Image.Image, local_frame: int, scene_frames: int) -> Image.Image:
    edge = max(1, round(FPS * 0.22))
    opacity = min(1.0, local_frame / edge, (scene_frames - 1 - local_frame) / edge)
    if opacity >= 1:
        return frame
    white = Image.new("RGB", frame.size, PAPER)
    return Image.blend(white, frame, max(0.0, opacity))


def _scene_frame(scene: int, progress: float, page: Image.Image,
                 focus: Image.Image, membrane: Image.Image, channel: Image.Image,
                 narration: str) -> Image.Image:
    titles = [
        "1  Find the membrane", "2  Two-layer structure",
        "3  A selective channel", "4  Controlled crossing",
    ]
    frame, draw = _base_frame(titles[scene - 1], narration)
    if scene == 1:
        # Start with the page, then zoom into the extracted first-quarter region.
        ease = progress * progress * (3 - 2 * progress)
        source_ratio = page.width / page.height
        focus_box = (0.055, 0.035, 0.975, 0.270)
        start = (0.0, 0.0, 1.0, 1.0)
        box = [start[i] + (focus_box[i] - start[i]) * ease for i in range(4)]
        crop = _normalized_crop(page, box)
        visual, pos = _contain(crop, (58, 125, 520, 730))
        frame.paste(visual, pos)
        pulse = int(80 + 90 * (0.5 + 0.5 * math.sin(progress * math.tau * 2)))
        draw.rounded_rectangle((292, 320, 505, 525), radius=16,
                               outline=(*GREEN, pulse), width=6)
    elif scene == 2:
        visual, pos = _contain(ImageEnhance.Contrast(membrane).enhance(1.05), (58, 150, 520, 650))
        frame.paste(visual, pos)
        x, y = pos
        band_alpha = int(50 + 80 * (0.5 + 0.5 * math.sin(progress * math.tau * 2)))
        # Positions refer to the two head rows and hydrophobic center in the extracted asset.
        draw.rounded_rectangle((x + 24, y + visual.height * .28, x + visual.width * .70,
                                y + visual.height * .43), radius=12,
                               fill=(80, 156, 218, band_alpha), outline=(43, 100, 170, 190), width=3)
        draw.rounded_rectangle((x + 24, y + visual.height * .43, x + visual.width * .70,
                                y + visual.height * .68), radius=12,
                               fill=(*YELLOW, band_alpha), outline=(181, 137, 35, 190), width=3)
        draw.ellipse((91, 674, 107, 690), fill=(80, 156, 218), outline=(43, 100, 170))
        draw.text((118, 670), "water-facing hydrophilic heads", font=_font(19, bold=True),
                  fill=(43, 100, 170))
        draw.ellipse((91, 711, 107, 727), fill=YELLOW, outline=(181, 137, 35))
        draw.text((118, 707), "protected hydrophobic tails", font=_font(19, bold=True),
                  fill=(151, 103, 28))
    elif scene == 3:
        visual, pos = _contain(channel, (90, 130, 490, 650))
        frame.paste(visual, pos)
        x, y = pos
        center_x = x + visual.width * 0.52
        for offset in (0.0, 0.34, 0.68):
            phase = (progress + offset) % 1.0
            cy = y + visual.height * (0.20 + 0.68 * phase)
            radius = 10
            draw.ellipse((center_x-radius, cy-radius, center_x+radius, cy+radius),
                         fill=(*PURPLE, 235), outline=(90, 43, 92, 255), width=2)
        draw.line((center_x, y + visual.height * .18, center_x, y + visual.height * .90),
                  fill=(*PURPLE, 140), width=5)
        draw.polygon(((center_x - 10, y + visual.height * .86),
                      (center_x + 10, y + visual.height * .86),
                      (center_x, y + visual.height * .92)), fill=(*PURPLE, 210))
        draw.text((137, 682), "specific ions follow the channel", font=_font(20, bold=True), fill=INK)
    else:
        visual, pos = _contain(membrane, (58, 155, 520, 610))
        frame.paste(visual, pos)
        x, y = pos
        gate_x = x + visual.width * .44
        for offset in (0.0, .45):
            phase = (progress + offset) % 1.0
            cy = y + visual.height * (.20 + .67 * phase)
            draw.ellipse((gate_x-9, cy-9, gate_x+9, cy+9), fill=(*PURPLE, 235))
        blocked_x = x + visual.width * .80
        blocked_y = y + visual.height * .27 + math.sin(progress * math.tau) * 8
        draw.ellipse((blocked_x-14, blocked_y-14, blocked_x+14, blocked_y+14),
                     fill=(94, 164, 103, 230), outline=(45, 104, 57, 255), width=3)
        draw.line((blocked_x-18, blocked_y+24, blocked_x+18, blocked_y+24),
                  fill=(190, 66, 67, 230), width=5)
        _rounded(draw, (82, 645, 495, 742), (224, 241, 218, 235), (*GREEN, 220), 3)
        draw.text((108, 663), "Selective permeability", font=_font(25, bold=True), fill=(43, 102, 57))
        draw.text((108, 700), "controls the cell's balance", font=_font(20), fill=(43, 102, 57))
    return frame


def render_animation(output_dir: Path, plan: dict, *, audio_path: Path | None = None,
                     asset_dir: Path | None = None) -> Path:
    import imageio_ffmpeg

    output_dir.mkdir(parents=True, exist_ok=True)
    assets = asset_dir or output_dir / "assets"
    page = Image.open(assets / "page_01.png").convert("RGB")
    focus = Image.open(assets / "focus_top_quarter.png").convert("RGB")
    membrane = Image.open(assets / "phospholipid_bilayer.png").convert("RGB")
    channel = Image.open(assets / "ion_channel.png").convert("RGB")
    silent = output_dir / "video_silent.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    scene_frames = round(3.75 * FPS)
    keyframes = output_dir / "keyframes"
    keyframes.mkdir(exist_ok=True)
    try:
        for scene_data in plan["scenes"]:
            scene = int(scene_data["scene"])
            for index in range(scene_frames):
                progress = index / max(scene_frames - 1, 1)
                frame = _scene_frame(
                    scene, progress, page, focus, membrane, channel,
                    scene_data["narration"],
                )
                frame = _fade(frame, index, scene_frames)
                if index == scene_frames // 2:
                    frame.save(keyframes / f"scene_{scene:02d}_keyframe.png")
                process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the visual-note animation")
    if audio_path is None:
        final = output_dir / "final.mp4"
        final.write_bytes(silent.read_bytes())
        return final
    final = output_dir / "final_with_audio.mp4"
    video_seconds = len(plan["scenes"]) * scene_frames / FPS
    with wave.open(str(audio_path), "rb") as source:
        audio_seconds = source.getnframes() / source.getframerate()
    # Chatterbox speech varies by voice. Gently tighten long narration instead of
    # cutting the final sentence at the 15-second video boundary.
    speed = max(1.0, audio_seconds / max(video_seconds - 0.2, 0.5))
    result = subprocess.run([
        ffmpeg, "-loglevel", "error", "-y", "-i", str(silent), "-i", str(audio_path),
        "-c:v", "copy", "-filter:a", f"atempo={speed:.6f}",
        "-c:a", "aac", "-b:a", "128k", "-t", f"{video_seconds:.3f}",
        "-movflags", "+faststart", str(final),
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-800:])
    return final
