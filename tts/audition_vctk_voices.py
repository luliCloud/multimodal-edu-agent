#!/usr/bin/env python3
"""Generate one audition sample per VCTK candidate reference clip, using
Chatterbox TTS, so they can be listened through and compared side by side.

Usage:
    python audition_vctk_voices.py
"""
from pathlib import Path

import soundfile as sf

from chatterbox_device import load_chatterbox_model

TEST_LINE = (
    "Carbon can bond to itself over and over, forming chains and rings, "
    "and that's basically why organic chemistry exists at all."
)

CANDIDATES_DIR = Path(__file__).parent / "reference_voices" / "candidates"
AUDITIONS_DIR = Path(__file__).parent / "reference_voices" / "auditions"


def main():
    candidates = sorted(CANDIDATES_DIR.glob("*.wav"))
    if not candidates:
        print(f"No candidate clips found in {CANDIDATES_DIR}")
        return

    AUDITIONS_DIR.mkdir(parents=True, exist_ok=True)

    model, device = load_chatterbox_model()
    print(f"\nChatterbox loaded on device: {device}\n")

    results = []
    for candidate in candidates:
        stem = candidate.stem  # e.g. "p294_female"
        out_path = AUDITIONS_DIR / f"{stem}_sample.wav"
        print(f"Synthesizing with {candidate.name}...")
        wav = model.generate(TEST_LINE, audio_prompt_path=str(candidate))
        sf.write(out_path, wav.squeeze(0).numpy(), model.sr)
        results.append((stem, out_path))

    print("\nAudition samples ready. Listen and pick your favorites:\n")
    for i, (stem, out_path) in enumerate(results, start=1):
        print(f"  {i}. {stem}  ->  {out_path}")


if __name__ == "__main__":
    main()
