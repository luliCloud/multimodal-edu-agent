#!/usr/bin/env python3
"""Multi-speaker TTS generator for structured educational dialogue scripts.

Uses Chatterbox TTS (resemble-ai/chatterbox) for fully local/offline voice
cloning: each speaker role maps to a POOL of candidate reference clips in
voice_map.json, and one clip is picked per role per run (consistently for
every line spoken by that role within that run, so a single output file
never switches a character's voice mid-way). Different runs of the same
script can land on different pool members, which is intentional: it gives
variety across the many scripts this project will eventually produce
without having to hand-edit the config for every video. No paid APIs, no
cloud inference.
"""
import argparse
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("tts_cli")

VALID_GENDERS = {"male", "female"}
PROJECT_ROOT = Path(__file__).parent
CANDIDATES_DIR = PROJECT_ROOT / "reference_voices" / "candidates"


def load_voice_map(voice_map_path: Path) -> dict:
    if voice_map_path.exists():
        with open(voice_map_path) as f:
            return json.load(f)
    return {}


def save_voice_map(voice_map: dict, voice_map_path: Path):
    with open(voice_map_path, "w") as f:
        json.dump(voice_map, f, indent=2)
        f.write("\n")


def list_candidate_clips(gender: str) -> list:
    """All reference clips in reference_voices/candidates/ for a gender.

    Candidate files are named <speaker_id>_<gender>.wav.
    """
    return sorted(CANDIDATES_DIR.glob(f"*_{gender}.wav"))


def all_mapped_clips(voice_map: dict) -> set:
    used = set()
    for pool in voice_map.values():
        for entry in pool:
            if isinstance(entry, dict) and "ref_clip" in entry:
                used.add(entry["ref_clip"])
    return used


def resolve_voice_map(voice_map: dict, turns: list, voice_map_path: Path) -> dict:
    """Fill in a starter pool for any speaker missing from the voice map.

    Assigns the next unused candidate reference clip of the right gender
    (round-robin, reused once exhausted) and persists the result back to
    the config file. You can freely hand-edit the file afterward to add
    more alternates to that role's pool.
    """
    voice_map = {role: list(pool) for role, pool in voice_map.items()}
    used = all_mapped_clips(voice_map)

    seen_speakers = set()
    changed = False
    for turn in turns:
        speaker = turn.get("speaker")
        gender = turn.get("gender")
        if speaker in seen_speakers or speaker in voice_map:
            continue
        seen_speakers.add(speaker)

        if gender not in VALID_GENDERS:
            continue  # will be reported as a per-line error during synthesis

        pool = list_candidate_clips(gender)
        if not pool:
            continue  # will be reported as a per-line error during synthesis
        pool_strs = [str(p.relative_to(PROJECT_ROOT)) for p in pool]
        unused = [c for c in pool_strs if c not in used]
        chosen = unused[0] if unused else pool_strs[len(used) % len(pool_strs)]
        used.add(chosen)
        voice_map[speaker] = [{"ref_clip": chosen, "gender": gender}]
        changed = True
        logger.info("Assigned new speaker '%s' (%s) -> ref_clip '%s'", speaker, gender, chosen)

    if changed:
        save_voice_map(voice_map, voice_map_path)
        logger.info("Updated voice map saved to %s", voice_map_path)

    return voice_map


def pick_voice_for_run(voice_map: dict, turns: list, rng: random.Random) -> dict:
    """Pick one reference clip per role for this run, from that role's pool.

    Picked once per role (not per line) so a role's voice stays fixed for
    the whole output, then logged so the manifest/console record exactly
    which pool member this particular run used.
    """
    chosen = {}
    for turn in turns:
        speaker = turn.get("speaker")
        if speaker in chosen or speaker not in voice_map:
            continue
        pool = voice_map[speaker]
        if not pool:
            continue
        entry = rng.choice(pool)
        chosen[speaker] = entry
        logger.info("Role '%s' -> %s (%s) [%d option(s) in pool]", speaker, entry.get("ref_clip"), entry.get("gender"), len(pool))
    return chosen


def run_generate(args):
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script file not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    with open(script_path) as f:
        turns = json.load(f)
    if not isinstance(turns, list) or not turns:
        print("ERROR: script must be a non-empty JSON array of turns", file=sys.stderr)
        sys.exit(1)

    voice_map_path = Path(args.voice_map)
    voice_map = load_voice_map(voice_map_path)
    voice_map = resolve_voice_map(voice_map, turns, voice_map_path)

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    run_voices = pick_voice_for_run(voice_map, turns, rng)

    import soundfile as sf
    from pydub import AudioSegment

    from chatterbox_device import load_chatterbox_model

    logger.info("Loading Chatterbox model (this downloads weights on first run)...")
    model, device = load_chatterbox_model()
    logger.info("Chatterbox loaded on device: %s", device)

    sample_rate = model.sr
    silence = AudioSegment.silent(duration=args.silence_ms, frame_rate=sample_rate)

    # Chatterbox's prepare_conditionals(ref_clip) - the reference-audio
    # encoding step - is identical every time for the same clip. generate()
    # only redoes it when you pass audio_prompt_path, so we call it once
    # per unique clip up front, cache the resulting Conditionals object,
    # and swap it into model.conds before each line instead of re-encoding.
    conditioning_cache = {}
    synth_time_total = 0.0

    with tempfile.TemporaryDirectory(prefix="edutok_tts_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        final_audio = AudioSegment.empty()
        manifest = []
        cursor_ms = 0
        first_clip = True

        for idx, turn in enumerate(turns):
            speaker = turn.get("speaker")
            gender = turn.get("gender")
            text = turn.get("line")
            entry = {
                "index": idx,
                "speaker": speaker,
                "gender": gender,
                "text": text,
            }

            try:
                if not speaker or not text:
                    raise ValueError("missing required 'speaker' or 'line' field")
                if gender not in VALID_GENDERS:
                    raise ValueError(f"invalid gender '{gender}' (expected 'male' or 'female')")
                voice_entry = run_voices.get(speaker)
                if not voice_entry:
                    raise ValueError(
                        f"no reference clip mapped for speaker '{speaker}' - check {voice_map_path}"
                    )
                ref_clip_path = PROJECT_ROOT / voice_entry["ref_clip"]
                if not ref_clip_path.exists():
                    raise ValueError(f"reference clip not found: {ref_clip_path}")

                ref_clip_key = str(ref_clip_path)
                if ref_clip_key not in conditioning_cache:
                    t0 = time.perf_counter()
                    model.prepare_conditionals(ref_clip_key)
                    prep_s = time.perf_counter() - t0
                    conditioning_cache[ref_clip_key] = model.conds
                    logger.info("Encoded reference clip for '%s' (%s) in %.2fs - cached for rest of run", speaker, ref_clip_path.name, prep_s)
                else:
                    model.conds = conditioning_cache[ref_clip_key]

                t0 = time.perf_counter()
                wav = model.generate(text)
                synth_time_total += time.perf_counter() - t0
                audio_np = wav.squeeze(0).cpu().numpy()
                clip_path = tmp_dir / f"line_{idx:04d}.wav"
                sf.write(clip_path, audio_np, sample_rate)

                clip = AudioSegment.from_wav(clip_path)

                if not first_clip:
                    final_audio += silence
                    cursor_ms += args.silence_ms
                first_clip = False

                start_ms = cursor_ms
                final_audio += clip
                cursor_ms += len(clip)
                end_ms = cursor_ms

                entry.update(
                    {
                        "status": "ok",
                        "ref_clip": voice_entry["ref_clip"],
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "start_s": round(start_ms / 1000, 3),
                        "end_s": round(end_ms / 1000, 3),
                    }
                )
                manifest.append(entry)
                logger.info("[%d] %s (%s): synthesized (%dms)", idx, speaker, voice_entry["ref_clip"], len(clip))

            except Exception as exc:
                entry.update({"status": "skipped", "error": str(exc)})
                manifest.append(entry)
                logger.error("[%d] SKIPPED speaker=%s text=%r reason=%s", idx, speaker, text, exc)

        if len(final_audio) == 0:
            print("ERROR: no lines were successfully synthesized; nothing to write", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output)
        final_audio.set_frame_rate(sample_rate).export(output_path, format="wav")

        manifest_path = Path(args.manifest)
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "output_audio": str(output_path),
                    "sample_rate": sample_rate,
                    "silence_ms_between_turns": args.silence_ms,
                    "voices_used_this_run": {k: v["ref_clip"] for k, v in run_voices.items()},
                    "turns": manifest,
                },
                f,
                indent=2,
            )

    ok_count = sum(1 for m in manifest if m["status"] == "ok")
    skip_count = len(manifest) - ok_count
    logger.info(
        "Total generate() time: %.2fs across %d lines (%d unique voices encoded, cached)",
        synth_time_total, ok_count, len(conditioning_cache),
    )
    logger.info(
        "Done: %d/%d lines synthesized, %d skipped. Audio -> %s, manifest -> %s",
        ok_count, len(manifest), skip_count, output_path, manifest_path,
    )


def run_list_speakers(args):
    voice_map_path = Path(args.voice_map)
    voice_map = load_voice_map(voice_map_path)
    if not voice_map:
        print(f"No speakers mapped yet in {voice_map_path}")
        return
    print(f"Speakers mapped in {voice_map_path}:\n")
    for role, pool in voice_map.items():
        print(f"  {role}:")
        for i, entry in enumerate(pool, start=1):
            ref_clip = entry.get("ref_clip")
            gender = entry.get("gender", "?")
            exists = "" if (PROJECT_ROOT / ref_clip).exists() else "  [MISSING FILE]"
            print(f"    {i}. {gender:<7} {ref_clip}{exists}")


def run_add_speaker(args):
    voice_map_path = Path(args.voice_map)
    voice_map = load_voice_map(voice_map_path)

    ref_clip_path = Path(args.ref_clip)
    if not ref_clip_path.exists():
        print(f"ERROR: reference clip not found: {ref_clip_path}", file=sys.stderr)
        sys.exit(1)

    try:
        stored_path = str(ref_clip_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        stored_path = str(ref_clip_path)

    pool = voice_map.setdefault(args.role, [])
    if any(e.get("ref_clip") == stored_path for e in pool):
        print(f"'{stored_path}' is already in '{args.role}''s pool, nothing to do.")
        return
    pool.append({"ref_clip": stored_path, "gender": args.gender})
    save_voice_map(voice_map, voice_map_path)
    print(f"Added {stored_path} ({args.gender}) to '{args.role}''s pool ({len(pool)} option(s) now) in {voice_map_path}")


def run_remove_speaker(args):
    voice_map_path = Path(args.voice_map)
    voice_map = load_voice_map(voice_map_path)
    if args.role not in voice_map:
        print(f"ERROR: role '{args.role}' not found in {voice_map_path}", file=sys.stderr)
        sys.exit(1)

    if args.ref_clip:
        pool = voice_map[args.role]
        new_pool = [e for e in pool if e.get("ref_clip") != args.ref_clip]
        if len(new_pool) == len(pool):
            print(f"ERROR: '{args.ref_clip}' not found in '{args.role}''s pool", file=sys.stderr)
            sys.exit(1)
        if new_pool:
            voice_map[args.role] = new_pool
            print(f"Removed {args.ref_clip} from '{args.role}''s pool ({len(new_pool)} option(s) left)")
        else:
            del voice_map[args.role]
            print(f"Removed {args.ref_clip} (last option) - '{args.role}' has no more voices, role removed")
    else:
        del voice_map[args.role]
        print(f"Removed role '{args.role}' entirely from {voice_map_path}")

    save_voice_map(voice_map, voice_map_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Synthesize audio from a dialogue script")
    gen.add_argument("script", help="Path to the dialogue script JSON file")
    gen.add_argument("--voice-map", default="voice_map.json", help="Path to voice mapping config (JSON)")
    gen.add_argument("--output", default="output.wav", help="Final concatenated WAV output path")
    gen.add_argument("--manifest", default="manifest.json", help="Manifest JSON output path")
    gen.add_argument(
        "--silence-ms", type=int, default=350,
        help="Silence in milliseconds inserted between speaker turns (default: 350)",
    )
    gen.add_argument(
        "--seed", type=int, default=None,
        help="Seed the per-role voice pick for reproducible runs (default: random each run)",
    )
    gen.set_defaults(func=run_generate)

    list_sp = subparsers.add_parser("list-speakers", help="List roles and their candidate voice pools")
    list_sp.add_argument("--voice-map", default="voice_map.json")
    list_sp.set_defaults(func=run_list_speakers)

    add_sp = subparsers.add_parser("add-speaker", help="Add a reference clip to a role's voice pool")
    add_sp.add_argument("role", help="Speaker role name, e.g. 'professor'")
    add_sp.add_argument("--ref-clip", required=True, help="Path to a reference audio clip for this voice")
    add_sp.add_argument("--gender", required=True, choices=sorted(VALID_GENDERS))
    add_sp.add_argument("--voice-map", default="voice_map.json")
    add_sp.set_defaults(func=run_add_speaker)

    remove_sp = subparsers.add_parser("remove-speaker", help="Remove a role, or one clip from its pool")
    remove_sp.add_argument("role")
    remove_sp.add_argument(
        "--ref-clip", default=None,
        help="Remove only this clip from the role's pool (default: remove the whole role)",
    )
    remove_sp.add_argument("--voice-map", default="voice_map.json")
    remove_sp.set_defaults(func=run_remove_speaker)

    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
