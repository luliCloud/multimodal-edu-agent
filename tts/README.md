# edutok-tts

Multi-speaker TTS generator for structured educational dialogue scripts, and
the bridge that syncs its narration audio with `backend/`'s video generation
pipeline so one `script.json` produces one playable video-with-audio file
from a single command. Fully local/offline — no paid APIs, no cloud
inference.

Uses [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) for voice
cloning: each speaker role in `voice_map.json` maps to a pool of short
reference audio clips, and Chatterbox clones that voice identity per line.

This directory is self-contained: its own `venv`, `requirements.txt`, and
scripts, independent of the `backend/` app elsewhere in this repo (different
dependencies — Chatterbox/torch here, FastAPI/Pydantic there).

## Architecture: how one script.json becomes one video-with-audio file

```
script.json
(one narration string + visual fields per scene, one character)
        |
        v
STEP 1  tts/sync_storyboard_audio.py        [tts/'s venv - Chatterbox TTS]
        synthesizes scene[i].narration in order, measures the REAL
        duration of each line (can't be known before TTS actually runs)
        |
        v
script.synced.json + script.narration.wav
(same scenes, now with start_seconds/end_seconds + narration_audio_path)
        |
        v
STEP 2  backend.scripts.render_short_script  [backend/'s venv - mock/cuda_probe/wan]
        sizes each scene's clip to end_seconds - start_seconds (its real
        measured duration), concatenates into one silent final.mp4
        |
        v
final.mp4  (silent)
        |
        v
STEP 3  tts/mux_narration_video.py           [ffmpeg - no Python deps needed]
        combines the silent video + narration WAV into one file with
        both streams embedded
        |
        v
final_with_audio.mp4
(the one file you actually play - open it, picture and sound play together)
```

`backend/scripts/run_synced_short_demo.py` runs all three steps for you from
one `script.json` - that's the single command (see Quick Start below).

### Why this is sequential, not parallel

Steps 1 and 2 can't run at the same time if you want exact sync, and this is
a real dependency, not an arbitrary choice:

- **Speech duration can only be known after TTS actually runs.** There's no
  reliable way to predict "this line takes 3.2 seconds" before synthesizing
  it - it depends on the words, the voice, and Chatterbox's own pacing.
- **Video needs that duration *before* it generates a clip, not after.** The
  mock backend passes it straight to ffmpeg as `-t <duration>`; the Wan
  backend needs the frame count fixed before it starts the (slow) diffusion
  process, since frame count isn't something you can change afterward.

So audio runs first to get the real number, then video is sized to exactly
match it. The alternative - generating both at once against an *estimated*
duration, then time-stretching the video clip to fit afterward - would allow
parallelism, but introduces drift for the mock backend (harmless, just
colored rectangles) and visibly choppier output for Wan (stretching already-
generated frames either skips or duplicates them, instead of generating the
right frame count from the start). Given that TTS synthesis (seconds to tens
of seconds) is small next to how long real video generation takes anyway
(mock: near-instant; Wan: minutes), sequential execution costs very little
in practice while keeping sync exact - so that's what this pipeline does.

## Quick Start (single command, generates audio + video + merges them)

### 1. Set up both venvs

This repo has two independent Python environments - `tts/`'s (Chatterbox) and
`backend/`'s (FastAPI/Pydantic) - because their dependencies don't overlap
and Chatterbox's are heavy (torch, etc.). Both are required for the full
synced flow; `tts/`'s alone is enough if you only want audio (see Usage).

**macOS / Linux:**
```bash
# from the repo root
cd tts
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

cd ..
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell):**
```powershell
# from the repo root
cd tts
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
deactivate

cd ..
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

You'll also need **ffmpeg** on your `PATH` (used for video assembly and
muxing): macOS `brew install ffmpeg`, Windows `winget install ffmpeg` (or
download a build and add it to `PATH` manually), Linux `apt install ffmpeg`
or your distro's equivalent.

The reference voice clips needed to run TTS out of the box are already
included under `reference_voices/candidates/`, so no download step is
required before the first run.

### 2. Run the one command

From the repo root, with `backend/`'s venv active:

```bash
python -m backend.scripts.run_synced_short_demo examples/carbon_bonding/script.json --gender female
```

This needs no GPU and no Qwen planner run - `examples/carbon_bonding/script.json`
is a ready-made example. It synthesizes narration, renders the video sized to
match it, and muxes them into one file, printing the final path at the end
(`examples/carbon_bonding/final_with_audio.mp4`). Open that file - video and
audio play together automatically, nothing else needed.

Flags:
- `--backend mock|cuda_probe|wan` (default `mock` - no GPU required)
- `--tts-python /path/to/python` if `tts/`'s venv isn't at the default
  location (`tts/venv/bin/python` on macOS/Linux, `tts/venv/Scripts/python.exe`
  on Windows - this is auto-detected per OS)
- `--seed N` to make the voice pick reproducible
- `--no-mux` to skip step 3 and leave a silent video + separate WAV
- any other flag (e.g. `--scene 1`) passes through to `render_short_script.py`

## How to get a script.json

Every run starts from one `script.json` - you never write a separate "audio
script" and "video script". Three ways to produce one, today:

### Option A: hand-write it

Use `examples/carbon_bonding/script.json` as a template and the schema
reference below. Straightforward for a short, simple video.

### Option B: generate it with any AI assistant (works today, no GPU needed)

The full automated planner (`backend/scripts/plan_short_pdf.py`, using the
local Qwen model) needs a GPU. Until you have one available, you can get a
correctly-formatted `script.json` from any capable AI chat assistant (Claude,
ChatGPT, etc.) using the prompt below - same schema, same word-budget rules,
same output the automated planner will eventually produce. Paste this in,
filling in your own source text where marked:

```text
Generate a script.json file for a 15-second educational short-form video
pipeline. Output ONLY valid JSON matching the schema below - no markdown
code fences, no explanation, no extra fields beyond what's listed.

SOURCE TEXT / TOPIC:
<paste your story, poem, or topic description here>

SCHEMA:
Top-level object has exactly these keys: title, summary, character, style,
scenes, source_sentences, duration_seconds.

- "duration_seconds": must be the integer 15, always.
- "summary": an 8-48 word plain-English summary of the whole video.
- "character": ONE recurring character/narrator for the entire video -
  {"name": str, "kind": str (e.g. "human narrator", "plant", "animal"),
   "visual_identity": str (one sentence, their fixed look),
   "outfits": {"default": str}, "states": {"default": str}}
  (age/hair/eyes/clothes are optional - omit if not relevant)
- "style" (optional, fine to omit): {"type", "palette", "shapes", "lighting",
  "outline"} as short descriptive strings.
- "scenes": a list of 4 to 8 scenes (choose a count that fits the content -
  fewer scenes means more words allowed per scene, see the word budget
  below). Each scene:
  {
    "scene": <1-based index, sequential: 1, 2, 3, ...>,
    "action": "<what visually happens, one sentence>",
    "location": "<where>",
    "weather": "<weather/lighting, or \"clear\" if not applicable>",
    "camera": "<framing, e.g. \"wide\", \"medium\", \"close\">",
    "motion": "<what moves/animates in this shot, one sentence>",
    "narration": "<the exact line spoken aloud for this scene>",
    "outfit_id": "default",
    "state_id": "default",
    "source_sentence_ids": [<ids from source_sentences this scene covers>],
    "text": "<the source text this scene is based on>",
    "pages": [1],
    "narration_seconds": <float, see formula below>
  }
- Word budget for "narration" PER SCENE, based on total scene count:
    4 scenes or fewer -> 6 to 9 words
    5 scenes          -> 5 to 7 words
    6 scenes          -> 4 to 6 words
    7 or 8 scenes     -> 3 to 6 words
  The sum of every scene's narration word count must not exceed 36 words
  total across the whole video.
- "narration_seconds" per scene = max(3.0, word_count / 2.5), rounded to 1
  decimal place.
- "source_sentences": a list of {"id": <int>, "page": <int>, "text":
  "<one source sentence>"}. Every "source_sentence_ids" value used across
  all scenes combined must appear exactly once, and must exactly match the
  full set of ids in "source_sentences" (no sentence skipped, none reused).

Here is a correctly-formatted example (different topic, same shape) -
match this structure exactly:
<paste the contents of examples/carbon_bonding/script.json here as a
few-shot example>
```

After generating, validate it actually matches the schema before spending
time on TTS/video generation (catches typos, missing fields, word-budget
violations):

```bash
# from the repo root, with backend/'s venv active
python -c "
from backend.app.models.shorts import ShortPlan
import json
ShortPlan.model_validate(json.load(open('path/to/script.json')))
print('valid')
"
```

### Option C: the automated planner (needs a GPU)

```bash
python -m backend.scripts.plan_short_pdf "path/to/your.pdf"
```

Uses the local Qwen model to read a PDF and produce `script.json` directly.
See the root `README.md`'s "Two-Stage PDF Demo" section for full details.
This is the eventual default once GPU availability isn't a constraint;
Option B produces the same shape in the meantime.

## Manual TTS usage (no video pipeline)

If you just want multi-speaker audio from a plain dialogue script (no
`backend/` video involved):

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

## Syncing with the video pipeline (the three steps individually)

If you want to run or inspect the steps one at a time instead of using
`run_synced_short_demo.py`:

```bash
# Step 1 - from tts/'s venv
python sync_storyboard_audio.py path/to/script.json --gender female
# -> path/to/script.synced.json, path/to/script.narration.wav

# Step 2 - from backend/'s venv, repo root
python -m backend.scripts.render_short_script path/to/script.synced.json --backend mock
# -> path/to/final.mp4 (silent)

# Step 3 - either venv, needs only ffmpeg
python tts/mux_narration_video.py path/to/final.mp4 path/to/script.narration.wav \
  --output path/to/final_with_audio.mp4
```

`mux_narration_video.py` isn't specific to `backend/`'s output - see the next
section for using it with a video this pipeline never generated.

## Alternate workflow: narrating a pre-existing video

The flow above generates both the audio and the video. If you already have a
finished video (recorded, licensed, or made some other way) and just want to
add matching narration, the direction reverses: the video's length is fixed,
so only the narration can be shaped to fit it - `sync_storyboard_audio.py`
and the `ShortPlan` schema don't apply here, since there's no video to size.

1. Watch the video and note the rough timestamp of each visual beat you want
   narration to land on.
2. Hand-write a plain turns script (the same `{speaker, gender, line}` shape
   `tts_cli.py` always uses - a `video_cue_s` field is optional, just a note
   to yourself, `tts_cli.py` ignores unknown fields):
   ```json
   [
     {"speaker": "narrator", "gender": "female", "line": "...", "video_cue_s": 0},
     {"speaker": "narrator", "gender": "female", "line": "...", "video_cue_s": 6}
   ]
   ```
3. Generate the audio and check the real timing against the video:
   ```bash
   python tts_cli.py generate your_script.json --output narration.wav --manifest manifest.json
   ```
   `manifest.json`'s `turns[i].start_s`/`end_s` are the real synthesized
   timestamps - compare these to your video's actual beats and rewrite lines
   that drift too far (longer/shorter wording shifts timing).
4. Mux once you're happy with the timing:
   ```bash
   python mux_narration_video.py your_video.mp4 narration.wav --output final_with_audio.mp4
   ```

Total narration length is naturally capped by what a video's runtime can
hold at a normal speaking pace (roughly 2.3-2.7 words/second) - a 30-second
video has room for about 75-90 words of continuous narration, not
dramatically more, however many lines you split it across.

## Cross-platform notes

This pipeline is written to run on macOS, Linux, and Windows, but has
primarily been exercised on macOS so far. Known platform-specific points:

- **Device selection** (`chatterbox_device.py`): tries CUDA first (Linux/
  Windows with an NVIDIA GPU + a CUDA-enabled torch install), then Apple's
  MPS (Mac), then falls back to CPU everywhere else. CPU works everywhere,
  just slower per line - expect this on a Windows machine without CUDA set
  up, or any machine without a GPU.
- **venv layout**: `bin/python` + `source venv/bin/activate` on macOS/Linux
  vs. `Scripts\python.exe` + `venv\Scripts\Activate.ps1` on Windows.
  `run_synced_short_demo.py`'s default `--tts-python` path is chosen
  automatically based on the OS it's running on.
- **ffmpeg**: must be on `PATH` on every OS (`mux_narration_video.py` and
  `backend/`'s video assembly both shell out to it directly). `backend/`'s
  `mock_video.py` additionally falls back to the bundled `imageio-ffmpeg`
  binary if a system `ffmpeg` isn't found; `mux_narration_video.py` does not
  have that fallback yet, so install ffmpeg separately.
- **Path separators**: all paths are built with Python's `pathlib`
  (OS-agnostic) and `voice_map.json`'s stored paths use forward slashes,
  which `pathlib` resolves correctly on Windows too - no manual path editing
  needed.
- PowerShell's line-continuation character is a backtick (`` ` ``), not `\` -
  the multi-line `bash` examples above won't paste directly into PowerShell;
  either run them as one line or use `` ` `` instead of `\`.

## Troubleshooting

**No sound when playing a muxed file.** First confirm the file itself is
fine: `ffprobe -show_entries stream=codec_type,codec_name <file>` should
list both a `video` and an `audio` stream. If it does, the file's fine and
it's a playback-side issue - check the player's own volume control (some
players, like QuickTime, have a volume slider independent of the system
volume), make sure you're not looking at a stale window left open from a
previous file, and check your OS's audio output device selection.

**"narration is longer than the video" warning from `mux_narration_video.py`.**
Expected, not a bug: the narration WAV includes the silence gaps `tts_cli.py`
inserts *between* turns (350ms by default, see `--silence-ms`), but
consecutive video clips concatenate back-to-back with no equivalent gap. For
an N-scene video that's `(N-1) * 0.35s` of narration beyond the video's
length - e.g. 4 scenes -> 3 gaps -> ~1.05s, which is exactly what you'll see.
The last fraction of a second of trailing silence gets cut off by ffmpeg's
`-shortest`; since it's silence, not speech, this is harmless. If actual
spoken words are getting cut off instead, that's a different problem - a
narration line ran long enough to eat into the next gap, worth trimming that
line and re-running step 1.

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
