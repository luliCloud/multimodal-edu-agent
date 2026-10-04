#!/usr/bin/env python3
"""Synthesize narration audio for a video storyboard and write the ACTUAL
per-scene timing back into the script.

Bridges this tts/ module with the backend/ video pipeline's storyboard
schema (backend/app/models/shorts.py: ShortPlan/ShortScene). A ShortScene's
`narration_seconds` is only a pre-TTS word-count estimate (see
narration_seconds() in backend/app/services/storyboard_schema.py) - real
speech duration can't be known until TTS actually runs. This script:

  1. Reads a ShortPlan script.json (one `narration` string per scene, one
     global `character` for the whole plan - this project currently treats
     each plan as having a single narrator, no per-scene speaker switch).
  2. Converts the scenes into this project's `{speaker, gender, line}` turn
     format and runs tts_cli.py's normal `generate` command on them.
  3. Reads the resulting manifest's real per-turn start_s/end_s and writes
     them back into the scenes as `start_seconds`/`end_seconds`, plus a
     top-level `narration_audio_path` pointing at the combined WAV.

The output is a script.json with the same shape as the input, just with
real timing filled in - video generation (mock_video.py / cuda_probe_video.py
/ wan_video.py, via SegmentRequest.start_seconds/end_seconds) then sizes
each scene's clip to match its actual narration length instead of a
hardcoded or estimated duration. Muxing the audio track into the rendered
video is a separate, not-yet-built step.

Usage:
    python sync_storyboard_audio.py storage/demo/after_the_rain/script.json \\
        --gender female
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def build_turns(scenes: list[dict], speaker: str, gender: str) -> list[dict]:
    return [{"speaker": speaker, "gender": gender, "line": scene["narration"]} for scene in scenes]


def run_tts(turns_path: Path, voice_map: Path, output_audio: Path,
           manifest_path: Path, seed: int | None) -> None:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "tts_cli.py"), "generate", str(turns_path),
        "--voice-map", str(voice_map), "--output", str(output_audio),
        "--manifest", str(manifest_path),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("script", help="Path to a ShortPlan script.json from the video pipeline")
    parser.add_argument("--gender", required=True, choices=["male", "female"],
                        help="Voice gender for this plan's single narrator")
    parser.add_argument("--speaker", default=None,
                        help="Speaker/role name for voice-pool lookup "
                             "(default: the plan's character name)")
    parser.add_argument("--voice-map", default=str(SCRIPT_DIR / "voice_map.json"))
    parser.add_argument("--output-script", default=None,
                        help="Where to write the updated script.json "
                             "(default: <script>.synced.json)")
    parser.add_argument("--output-audio", default=None,
                        help="Where to write the combined narration WAV "
                             "(default: <script>.narration.wav)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the per-role voice pick (see tts_cli.py)")
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        sys.exit(1)
    plan = json.loads(script_path.read_text())

    speaker = args.speaker or plan["character"]["name"]
    output_script = (
        Path(args.output_script) if args.output_script
        else script_path.with_name(script_path.stem + ".synced.json")
    )
    output_audio = (
        Path(args.output_audio) if args.output_audio
        else script_path.with_name(script_path.stem + ".narration.wav")
    )

    scenes = sorted(plan["scenes"], key=lambda scene: scene["scene"])
    turns = build_turns(scenes, speaker, args.gender)

    with tempfile.TemporaryDirectory(prefix="sync_storyboard_") as tmp_dir:
        turns_path = Path(tmp_dir) / "turns.json"
        manifest_path = Path(tmp_dir) / "manifest.json"
        turns_path.write_text(json.dumps(turns, indent=2))

        run_tts(turns_path, Path(args.voice_map), output_audio, manifest_path, args.seed)

        manifest = json.loads(manifest_path.read_text())

    manifest_turns = manifest["turns"]
    if len(manifest_turns) != len(scenes):
        print(
            f"ERROR: tts produced {len(manifest_turns)} turn(s) for {len(scenes)} scene(s) "
            "- counts must match 1:1", file=sys.stderr,
        )
        sys.exit(1)

    for scene, turn in zip(scenes, manifest_turns):
        if turn["status"] != "ok":
            print(
                f"ERROR: scene {scene['scene']} failed to synthesize: {turn.get('error')}",
                file=sys.stderr,
            )
            sys.exit(1)
        scene["start_seconds"] = turn["start_s"]
        scene["end_seconds"] = turn["end_s"]

    plan["scenes"] = scenes
    plan["narration_audio_path"] = str(output_audio.resolve())
    output_script.write_text(json.dumps(plan, indent=2) + "\n")

    print(f"Synced {len(scenes)} scene(s) with actual narration timing:")
    for scene in scenes:
        duration = scene["end_seconds"] - scene["start_seconds"]
        print(
            f"  scene {scene['scene']}: {scene['start_seconds']:.2f}s - "
            f"{scene['end_seconds']:.2f}s ({duration:.2f}s)"
        )
    print(f"\nNarration audio -> {output_audio}")
    print(f"Synced script   -> {output_script}")


if __name__ == "__main__":
    main()
