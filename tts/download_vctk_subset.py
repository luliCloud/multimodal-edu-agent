#!/usr/bin/env python3
"""Download a small, curated subset of the CSTR VCTK Corpus for use as
Chatterbox TTS reference (voice-cloning) clips.

The full corpus is an ~11GB zip hosted on Edinburgh DataShare. Rather than
downloading it, this script uses `remotezip` to perform HTTP range requests
against the zip's central directory, so only the specific entries we want
(speaker-info.txt + a handful of per-speaker audio files) are ever
transferred.

License: CSTR VCTK Corpus is CC BY 4.0 (Yamagishi, Veaux, MacDonald,
University of Edinburgh). See README.md for the attribution note required
before publishing anything derived from it.
"""
import re
from pathlib import Path

import remotezip

VCTK_ZIP_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

# Chosen from speaker-info.txt: 4 female + 4 male, all labeled "American"
# accent (to match this project's American-accent English content), no
# "(mic2 files unavailable)" / "(Text ... unavailable)" caveats in their row.
CHOSEN_SPEAKERS = {
    "p294": "female",  # 33, San Francisco
    "p297": "female",  # 20, New York
    "p300": "female",  # 23, California
    "p305": "female",  # 19, Philadelphia
    "p311": "male",    # 21, Iowa
    "p334": "male",    # 18, Chicago
    "p345": "male",    # 22, Florida
    "p360": "male",    # 19, New Jersey
}

# Early-ish sentence indices: in VCTK's numbering these fall in the shared
# "rainbow passage" portion that most speakers read early on, giving
# full, clean, comparable sentences across speakers (rather than the
# often-shorter later newspaper sentences).
CANDIDATE_INDICES = [5, 15, 25]

RAW_DIR = Path(__file__).parent / "reference_voices" / "vctk_raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to {VCTK_ZIP_URL} (range requests only, not a full download)...")
    rz = remotezip.RemoteZip(VCTK_ZIP_URL)

    meta_path = RAW_DIR.parent / "speaker-info.txt"
    if not meta_path.exists():
        meta_path.write_bytes(rz.read("speaker-info.txt"))
        print(f"Saved speaker metadata -> {meta_path}")

    all_names = set(rz.namelist())

    for speaker_id, gender in CHOSEN_SPEAKERS.items():
        speaker_dir = RAW_DIR / speaker_id
        speaker_dir.mkdir(exist_ok=True)
        pattern = re.compile(rf"^wav48_silence_trimmed/{speaker_id}/{speaker_id}_(\d+)_mic1\.flac$")
        available = sorted(
            (int(m.group(1)), n) for n in all_names if (m := pattern.match(n))
        )
        by_idx = dict(available)

        picked = 0
        for target_idx in CANDIDATE_INDICES:
            # exact index if present, else nearest available
            idx = target_idx if target_idx in by_idx else min(by_idx, key=lambda i: abs(i - target_idx))
            name = by_idx.pop(idx)
            out_path = speaker_dir / Path(name).name
            if not out_path.exists():
                out_path.write_bytes(rz.read(name))
            print(f"  [{speaker_id}/{gender}] {out_path.name} -> {out_path}")
            picked += 1

        if picked == 0:
            print(f"WARNING: no audio files found for speaker {speaker_id}")

    print("\nDone. Raw candidate clips are under reference_voices/vctk_raw/<speaker_id>/")


if __name__ == "__main__":
    main()
