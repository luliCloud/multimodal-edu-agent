"""End-to-end demo: narration audio -> timed video, from one script.json.

Chains the two stages that currently live in separate venvs:

  1. tts/sync_storyboard_audio.py        - Chatterbox TTS (tts/'s own venv)
  2. backend.scripts.render_short_script - video generation (this venv)

so you can go from a reviewed ShortPlan script.json straight to a narration
WAV and a video whose per-scene clip lengths match that narration, in one
command. See tts/README.md ("Syncing with the video pipeline") for what
sync_storyboard_audio.py actually does to the script.

Usage (try it with the bundled example, no GPU or Qwen needed):

    python -m backend.scripts.run_synced_short_demo \\
        examples/carbon_bonding/script.json --gender female

Any extra arguments (e.g. --scene 1, --character-reference PATH) are passed
through to render_short_script.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TTS_PYTHON = REPO_ROOT / "tts" / "venv" / "bin" / "python"
SYNC_SCRIPT = REPO_ROOT / "tts" / "sync_storyboard_audio.py"


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
                             "(default: tts/venv/bin/python)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the per-role voice pick (see tts_cli.py)")
    args, passthrough = parser.parse_known_args()

    script_path = args.script.resolve()
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    tts_python = Path(args.tts_python)
    if not tts_python.exists():
        print(
            f"ERROR: tts python not found at {tts_python}\n"
            "Set up tts/'s venv first:\n"
            "  cd tts && python3 -m venv venv && venv/bin/pip install -r requirements.txt\n"
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

    print(f"\n=== [2/2] Rendering video (backend={args.backend}) ===", flush=True)
    render_cmd = [
        sys.executable, "-m", "backend.scripts.render_short_script",
        str(synced_script), "--backend", args.backend,
        *passthrough,
    ]
    subprocess.run(render_cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
