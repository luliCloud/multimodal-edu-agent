"""End-to-end demo: narration audio -> timed video -> one playable file,
from a single script.json.

Chains three stages, the first two of which live in separate venvs:

  1. tts/sync_storyboard_audio.py        - Chatterbox TTS (tts/'s own venv)
  2. backend.scripts.render_short_script - video generation (this venv)
  3. tts/mux_narration_video.py          - combine them into one MP4

so you go from a reviewed ShortPlan script.json straight to a single file
with both the video and its narration embedded, ready to just double-click
and play. See tts/README.md ("Syncing with the video pipeline") for what
sync_storyboard_audio.py actually does to the script.

Usage (try it with the bundled example, no GPU or Qwen needed):

    python -m backend.scripts.run_synced_short_demo \\
        examples/carbon_bonding/script.json --gender female

Any extra arguments (e.g. --scene 1, --character-reference PATH) are passed
through to render_short_script.py. Pass --no-mux to skip step 3 and leave
the video silent and the narration as a separate WAV.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# venv layout differs by OS: Scripts/python.exe on Windows, bin/python elsewhere.
DEFAULT_TTS_PYTHON = (
    REPO_ROOT / "tts" / "venv" / "Scripts" / "python.exe" if sys.platform == "win32"
    else REPO_ROOT / "tts" / "venv" / "bin" / "python"
)
SYNC_SCRIPT = REPO_ROOT / "tts" / "sync_storyboard_audio.py"
MUX_SCRIPT = REPO_ROOT / "tts" / "mux_narration_video.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("script", type=Path, help="Path to a reviewed ShortPlan script.json")
    parser.add_argument("--gender", required=True, choices=["male", "female"],
                        help="Voice gender for the plan's single narrator")
    parser.add_argument("--backend", default="mock", choices=["mock", "cuda_probe", "wan"])
    parser.add_argument("--tts-python", default=str(DEFAULT_TTS_PYTHON),
                        help="Python executable inside tts/'s own venv "
                             "(default: tts/venv/Scripts/python.exe on Windows, "
                             "tts/venv/bin/python elsewhere)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the per-role voice pick (see tts_cli.py)")
    parser.add_argument("--no-mux", action="store_true",
                        help="Skip muxing narration into the video; leave them as separate files")
    args, passthrough = parser.parse_known_args()

    script_path = args.script.resolve()
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    tts_python = Path(args.tts_python)
    if not tts_python.exists():
        setup = (
            "  cd tts && python -m venv venv && venv\\Scripts\\pip install -r requirements.txt"
            if sys.platform == "win32" else
            "  cd tts && python3 -m venv venv && venv/bin/pip install -r requirements.txt"
        )
        print(
            f"ERROR: tts python not found at {tts_python}\n"
            f"Set up tts/'s venv first:\n{setup}\n"
            "or pass --tts-python /path/to/python",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=== [1/2] Synthesizing narration audio ({args.gender} voice) ===", flush=True)
    sync_cmd = [str(tts_python), str(SYNC_SCRIPT), str(script_path), "--gender", args.gender]
    if args.seed is not None:
        sync_cmd += ["--seed", str(args.seed)]
    subprocess.run(sync_cmd, check=True)

    synced_script = script_path.with_name(script_path.stem + ".synced.json")
    narration_audio = json.loads(synced_script.read_text())["narration_audio_path"]

    final_output = synced_script.with_name("final.mp4")
    print(f"\n=== [2/{'3' if not args.no_mux else '2'}] "
         f"Rendering video (backend={args.backend}) ===", flush=True)
    render_cmd = [
        sys.executable, "-m", "backend.scripts.render_short_script",
        str(synced_script), "--backend", args.backend,
        "--final-output", str(final_output),
        *passthrough,
    ]
    subprocess.run(render_cmd, check=True, cwd=REPO_ROOT)

    if args.no_mux:
        return

    muxed_output = final_output.with_name(final_output.stem + "_with_audio.mp4")
    print("\n=== [3/3] Muxing narration into the video ===", flush=True)
    mux_cmd = [
        sys.executable, str(MUX_SCRIPT), str(final_output), str(narration_audio),
        "--output", str(muxed_output),
    ]
    subprocess.run(mux_cmd, check=True)
    print(f"\nPlay this: {muxed_output}")


if __name__ == "__main__":
    main()
