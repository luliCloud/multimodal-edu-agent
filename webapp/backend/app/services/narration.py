"""Local narration synthesis and MP4 audio muxing for web renders."""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from webapp.backend.app.models import EditableScript

REPO_ROOT = Path(__file__).resolve().parents[4]
TTS_DIR = REPO_ROOT / "tts"


class NarrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarrationTrack:
    path: Path
    # Scene number -> the interval allocated to that scene in the combined WAV.
    timings: dict[int, tuple[float, float]]


def _voice_clip(gender: str) -> Path:
    filename = "p297_female.wav" if gender == "female" else "p311_male.wav"
    return TTS_DIR / "reference_voices" / "candidates" / filename


def synthesize_narration(*, script: EditableScript, selected: list[int],
                         tts_python: Path, gender: str, work_dir: Path,
                         seed: int = 42) -> NarrationTrack:
    """Run Chatterbox in its isolated venv and return real scene timing."""
    if gender not in {"female", "male"}:
        raise NarrationError("TTS_GENDER must be female or male")
    # Keep the venv's python symlink intact. Path.resolve() would turn it into the
    # system interpreter and silently discard the venv's installed packages.
    tts_python = tts_python.absolute()
    work_dir = work_dir.resolve()
    if not tts_python.is_file():
        raise NarrationError(
            f"TTS is enabled but its Python was not found at {tts_python}. "
            "Run ./setup.sh wan once, or set TTS_PYTHON."
        )
    voice = _voice_clip(gender)
    if not voice.is_file():
        raise NarrationError(f"Narrator reference voice is missing: {voice}")

    by_number = {scene.scene: scene for scene in script.scenes}
    turns = []
    for number in selected:
        scene = by_number[number]
        line = (scene.narration or scene.text or scene.action).strip()
        if not line:
            raise NarrationError(f"Scene {number} has no narration text")
        turns.append({"speaker": "narrator", "gender": gender, "line": line})

    work_dir.mkdir(parents=True, exist_ok=True)
    turns_path = work_dir / "narration-turns.json"
    voice_map_path = work_dir / "voice-map.json"
    audio_path = work_dir / "narration.wav"
    manifest_path = work_dir / "narration-manifest.json"
    turns_path.write_text(json.dumps(turns, indent=2), encoding="utf-8")
    voice_map_path.write_text(json.dumps({
        "narrator": [{"ref_clip": str(voice.resolve()), "gender": gender}],
    }, indent=2), encoding="utf-8")

    result = subprocess.run(
        [str(tts_python), str(TTS_DIR / "tts_cli.py"), "generate", str(turns_path),
         "--voice-map", str(voice_map_path), "--output", str(audio_path),
         "--manifest", str(manifest_path), "--seed", str(seed)],
        cwd=TTS_DIR, capture_output=True, text=True, timeout=1800, check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1500:]
        raise NarrationError(f"Narration synthesis failed: {detail}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    produced = manifest.get("turns", [])
    if len(produced) != len(selected) or any(turn.get("status") != "ok" for turn in produced):
        raise NarrationError("Narration synthesis did not produce every selected scene")

    # Allocate the silence between lines to the preceding scene. This makes the
    # concatenated video duration exactly match the WAV timeline.
    timings: dict[int, tuple[float, float]] = {}
    for index, (number, turn) in enumerate(zip(selected, produced)):
        start = float(turn["start_s"])
        end = (float(produced[index + 1]["start_s"])
               if index + 1 < len(produced) else float(turn["end_s"]))
        timings[number] = (start, end)
    return NarrationTrack(audio_path, timings)


def ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise NarrationError("ffmpeg is required to add narration to the final MP4") from exc


def media_duration(path: Path) -> float:
    result = subprocess.run(
        [ffmpeg_executable(), "-i", str(path)], capture_output=True, text=True,
        timeout=30, check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise NarrationError(f"Could not read media duration: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def mux_narration(video: Path, narration: Path, output: Path) -> Path:
    """Copy video frames and encode the narration as an AAC audio stream."""
    output.parent.mkdir(parents=True, exist_ok=True)
    video_seconds = media_duration(video)
    result = subprocess.run(
        [ffmpeg_executable(), "-loglevel", "error", "-y", "-i", str(video),
         "-i", str(narration), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-af", "apad", "-t", f"{video_seconds:.3f}",
         "-movflags", "+faststart", str(output)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if result.returncode:
        raise NarrationError(f"Could not add narration to video: {result.stderr[-1200:]}")
    return output
