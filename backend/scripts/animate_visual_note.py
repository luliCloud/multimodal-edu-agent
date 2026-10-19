"""Create a source-faithful short from the first quarter of a visual class-note PDF."""

import argparse
import json
import subprocess
from pathlib import Path

from backend.app.models.shorts import ShortPlan
from backend.app.services.visual_note_video import (
    cell_membrane_plan, extract_cell_membrane_bible, render_animation, render_pdf_page,
)


REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("storage/demo/biomedical_class_note"))
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("[visual-note] rasterizing PDF page 1", flush=True)
    page = render_pdf_page(args.pdf.resolve())
    manifest = extract_cell_membrane_bible(page, output, args.pdf.name)
    print(
        "[visual-note] visual bible: "
        + ", ".join(asset["id"] for asset in manifest["assets"]),
        flush=True,
    )

    plan = cell_membrane_plan()
    ShortPlan.model_validate(plan)
    script = output / "script.json"
    script.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    audio = None
    if not args.no_audio:
        tts_python = REPO / "tts" / "venv" / "bin" / "python"
        sync = REPO / "tts" / "sync_storyboard_audio.py"
        audio = output / "narration.wav"
        print("[visual-note] synthesizing four narration lines", flush=True)
        subprocess.run([
            str(tts_python), str(sync), str(script), "--gender", "female", "--seed", "42",
            "--speaker", "professor",
            "--output-audio", str(audio),
            "--output-script", str(output / "script.synced.json"),
        ], check=True, cwd=REPO)

    print("[visual-note] animating extracted note elements", flush=True)
    final = render_animation(output, plan, audio_path=audio)
    print(f"[visual-note] visual bible -> {output / 'visual_bible.json'}", flush=True)
    print(f"[visual-note] final video -> {final}", flush=True)


if __name__ == "__main__":
    main()
