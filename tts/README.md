# edutok-tts

Multi-speaker TTS generator for structured educational dialogue scripts.
Fully local/offline — no paid APIs, no cloud inference.

Uses [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) for
voice cloning: each speaker role in `voice_map.json` maps to a pool of
short reference audio clips, and Chatterbox clones that voice identity
per line.

This directory is self-contained: its own `venv`, `requirements.txt`, and
scripts, independent of the `backend/` app elsewhere in this repo.

## Setup

```bash
cd tts
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The reference clips needed to run generation out of the box are already
included under `reference_voices/candidates/` (see the licensing note
below), so no download step is required before the first run.

## Usage

```bash
# Synthesize the bundled example script (chemistry dialogue, 3 speakers)
python tts_cli.py generate ch1_structure_bonding_script.json --output out.wav --manifest out_manifest.json

# List roles and their candidate voice pools
python tts_cli.py list-speakers

# Add a reference clip to a role's pool
python tts_cli.py add-speaker professor --ref-clip reference_voices/candidates/p345_male.wav --gender male

# Remove one clip from a pool (or the whole role, if --ref-clip is omitted)
python tts_cli.py remove-speaker professor --ref-clip reference_voices/candidates/p345_male.wav
```

`out.wav` and `out_manifest.json` are generated output and are gitignored
(see `.gitignore`) — re-run the command above to reproduce them.

A role's voice is picked once per `generate` run (randomly from its pool,
or deterministically with `--seed N`), so a single output file never
switches a character's voice mid-way, but re-running the same script can
land on a different pool member for variety across videos.

### Other scripts

- `download_vctk_subset.py` — re-downloads the curated VCTK reference
  clips into `reference_voices/vctk_raw/` (range requests only, not the
  full 11GB corpus). Not needed unless you want to rebuild the candidate
  set from scratch.
- `audition_vctk_voices.py` — synthesizes a sample line with every
  candidate clip so you can listen through and compare voices, writing
  results to `reference_voices/auditions/`.
- `benchmark_conditioning.py` — measures how much of Chatterbox's
  per-line time is model load vs. reference-clip conditioning vs.
  generation, used to validate the conditioning-cache optimization in
  `tts_cli.py`.

## Reference voices: CSTR VCTK Corpus

The reference clips under `reference_voices/` are sourced from the
**CSTR VCTK Corpus** (version 0.92), a small curated subset downloaded
via `download_vctk_subset.py`.

The VCTK Corpus is licensed **CC BY 4.0** and requires attribution if
this content — or anything synthesized from these reference voices — is
ever published:

> Christophe Veaux, Junichi Yamagishi, Kirsten MacDonald, "CSTR VCTK
> Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit",
> University of Edinburgh, The Centre for Speech Technology Research
> (CSTR).

License text: https://creativecommons.org/licenses/by/4.0/
Corpus homepage: https://datashare.ed.ac.uk/handle/10283/3443
