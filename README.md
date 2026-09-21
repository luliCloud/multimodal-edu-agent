# multimodal-edu-agent
Multimodal AI Agent for Automated Educational Content & Interactive Companion Learning

This repository is a single-GPU development prototype for the EduTok-style
multimodal pipeline:

```text
script segments -> job queue -> worker pool -> video artifacts -> API playback
```

The default generator is a CPU mock, and an optional synthetic CUDA probe is
available. Four in-process workers accept jobs; a
single GPU slot serializes video-generation calls. Run one API process for this
prototype (`uvicorn backend.app.main:app` without `--workers`). The lock does not
coordinate multiple processes or machines. The optional Wan backend runs real
model inference; Redis/Celery integration remains future work.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cuda]"
```

`.[dev]` on its own is enough to run the API, the mock backend and the test
suite; the `cuda` and `wan` extras are only needed for GPU generation.

Run a local demo that generates short mock `.mp4` files with `ffmpeg`:

```bash
python -m backend.scripts.run_local_demo
```

## Two-Stage PDF Demo

The main demo is deliberately split at the review boundary. Stage 1 runs PDF extraction
and Qwen, then saves an editable script. Stage 2 consumes that saved script and never
reruns the planner.

### Stage 1: PDF to Script

```bash
python -m backend.scripts.plan_short_pdf "test_pdfs/After the Rain.pdf"
```

This writes:

```text
storage/demo/after_the_rain/script.json
storage/demo/after_the_rain/image_prompts.json
```

For a recurring human character, the Wan renderer now uses a real visual bible rather
than treating the character description as sufficient. Put one canonical image at
`storage/demo/after_the_rain/assets/character_reference.png`. Optional scene images use
the stable names `scene_01_keyframe.png`, `scene_02_keyframe.png`, and so on. Every scene
receives the same reference ID and image; when a keyframe exists, VACE fixes it as frame
zero and generates only the later motion. This keeps identity in an image asset while
scene prompts describe action, location, weather, camera, and motion.

Review `summary`, `character`, `style`, all `scenes`, narration, outfits and states
in `script.json` before spending GPU time on video.
While Qwen runs, the CLI reports tokenizer and model loading, generation attempts,
generated-token milestones every 100 tokens, validation failures and retries, and final
schema success. Run the planner and Wan sequentially on a 16 GB GPU; starting both at
once can exhaust GPU memory.

### Stage 2: Reviewed Script to Video

Fast mock demo:

```bash
python -m backend.scripts.render_short_script \
  storage/demo/after_the_rain/script.json --backend mock
```

Real Wan GPU generation:

```bash
python -m backend.scripts.render_short_script \
  storage/demo/after_the_rain/script.json --backend wan
```

The renderer automatically discovers the reference and keyframes under the sibling
`assets/` directory. Use `--character-reference PATH` and `--keyframe-dir DIR` to point
at other assets, or `--no-character-reference` for an explicit T2V comparison. With a
human reference, Wan uses `Wan-AI/Wan2.1-VACE-1.3B-diffusers`; that checkpoint downloads
on its first run. The CLI reports both the bible path and how many selected scenes have
keyframes, so a reference-conditioned run cannot be confused with plain T2V.

Reviewed keyframes default to `WAN_KEYFRAME_MOTION=stable`: CUDA applies a gentle camera
move and weather overlays to the original pixels, so the face and body cannot be redrawn
between frames. Set `WAN_KEYFRAME_MOTION=vace` for a learned-motion comparison. VACE can
produce larger character motion, but the 1.3B checkpoint may drift from the reference;
the stable mode is the consistency-first default for the single-GPU prototype.

For actual layered motion, add `scene_01_background.png` and
`scene_01_character.png` beside `scene_01_keyframe.png` (and the matching files for each
scene). Character PNGs must have a real alpha channel. The CUDA compositor then moves the
character independently of the background: the puddle scene follows a jump arc and adds
an expanding landing ripple, while the other scenes use character sway, moving rain,
sunlight changes, and rainbow highlights. If a scene has no layers, it falls back to the
whole-keyframe camera move.

Both commands generate the selected scene clips and one `*-combined.mp4` under the script's
`videos/` directory, then copy the combined result to the stable path
`storage/demo/after_the_rain/final.mp4`. Use `--scene 1` to render one reviewed scene as
a cheaper smoke test; that single clip is also published as `final.mp4`.
The CLI prints the validated title, backend, selected scenes, Wan settings, each scene's
start and output path, and the final assembly step. During a blocking model call it also
prints an elapsed-time heartbeat every 15 seconds, so model loading and inference are
visibly different from a stopped process.

Run both stages with one command when no manual review is needed:

```bash
python -m backend.scripts.run_short_demo \
  "test_pdfs/After the Rain.pdf" --backend mock
```

Replace `mock` with `wan` for real inference. Qwen exits before video generation, so both
models do not occupy the single GPU simultaneously. The combined MP4 currently contains
the visual clips; `narration` and timing are present in the script, while TTS, burned-in
captions and transitions remain the next media-composition stage.

Start the API:

```bash
uvicorn backend.app.main:app
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

```text
GET  /health
POST /upload
GET  /status/{job_id}
GET  /videos/{doc_id}
GET  /media/{filename}
POST /pdf/keywords
POST /pdf/plan
POST /pdf/jobs
POST /scripts/jobs
```

For the same reviewable flow over HTTP, start one API process with:

```bash
PLANNER_BACKEND=qwen GENERATOR_BACKEND=mock uvicorn backend.app.main:app
```

Upload a PDF and save the returned script:

```bash
mkdir -p storage/demo/api
curl -X POST http://127.0.0.1:8000/pdf/plan \
  -H 'Content-Type: application/pdf' \
  --data-binary @"test_pdfs/After the Rain.pdf" \
  -o storage/demo/api/script.json
```

Submit that reviewed script without replanning:

```bash
curl -X POST http://127.0.0.1:8000/scripts/jobs \
  -H 'Content-Type: application/json' \
  --data-binary @storage/demo/api/script.json
```

Copy the returned `job_id` into `/status/{job_id}` and its `doc_id` into
`/videos/{doc_id}`. The final entry is the combined video and its `url` can be opened
through `/media/<filename>.mp4`.

Example upload request:

```json
{
  "title": "Parallel GPU Inference Demo",
  "segments": [
    {"title": "Seed", "text": "A little seed slept in dark soil."},
    {"title": "Flower", "text": "A yellow flower appeared and a bee visited."}
  ]
}
```

The first PDF ingestion step extracts candidate video topics with page references. It
reads text with `pdftotext` from poppler-utils when that binary is installed and falls
back to the bundled `pdfplumber` otherwise, so no system package is required. It accepts
a text-based PDF as the raw request body (up to 10 MB):

```bash
curl -X POST http://127.0.0.1:8000/pdf/keywords \
  -H 'Content-Type: application/pdf' \
  --data-binary @test_pdfs/The_Little_Seed.pdf
```

This CPU-only heuristic is a baseline for selecting topics. It filters infrastructure
terms, pronouns, standalone colors and accidental prose spans while retaining page
evidence. Scanned PDFs need OCR first.
`/pdf/plan` preserves narrative order, page references, and per-scene keywords;
`/pdf/jobs` submits those scenes and keywords as video tasks. With the Wan
backend, local Qwen3-4B-Instruct first summarizes the extracted sentences and plans
4-8 scenes for a 15-second short. Ordinary stories use four scenes; milestone-dense plant
lifecycle stories use eight single-action scenes. Every plan separates global character,
outfit, lifecycle-state and style definitions from per-scene action, weather, camera,
motion and narration. Each scene retains exact source-sentence IDs and page evidence.
The planner assigns every source sentence to an ordered group before asking Qwen to
design that group's shot, so short actions and the ending cannot be silently dropped.
This semantic plan is reviewable in `/pdf/plan`. The
planner runs in a separate GPU process that exits before Wan inference, so the
two models do not occupy GPU memory together. Set `PLANNER_BACKEND=extractive`
to use the earlier sentence-grouping baseline. The Qwen planner currently
supports short stories with up to 32 narrative sentences; longer PDFs are
rejected with 422, and a planner that cannot produce a usable storyboard
returns 502. Text extraction currently supports English text-based PDFs, not
scanned pages. Check the generated visual prompts before an expensive video
run: model-written details can still go beyond the PDF even when the scene
order is grounded.

Every scene is checked against the Pydantic contract in
`backend/app/models/shorts.py`: the plan has 4-8 consecutively ordered scenes, full source
coverage, valid global `outfit_id` and `state_id` references, and a per-scene narration
budget fitted to scene count with no more than 36 words total. Action, narration and
summary fields must contain source anchors from their assigned groups. Botanical stories cannot
silently become human characters. Explicit clothing changes are assigned deterministic
before/after outfits, and lifecycle stories must use multiple global character states.
When generation fails a check,
the worker re-prompts the model with the specific error while it is still
loaded, up to `PLANNER_MAX_ATTEMPTS` times (default 3), rather than losing
the run and its model load. Each scene also carries `narration` and a
`narration_seconds` estimate at roughly 150 words per minute, which is the
script input the audio stage needs. `visual_prompt` is ready for plain text-to-video;
`keyframe_prompt` is used to create a scene image from the canonical character asset.
The Wan VACE path then receives both that character image and, when available, the scene
keyframe. The extractive baseline fills the shared source, keyword and
narration fields without loading Qwen.

To exercise the single-GPU worker with a synthetic CUDA-generated MP4 (this is
**not** a text-to-video model), install the optional dependencies and run:

```bash
GENERATOR_BACKEND=cuda_probe python -m backend.scripts.run_cuda_pdf_demo test_pdfs/The_Little_Seed.pdf
```

The default backend remains `mock`. The CUDA probe confirms GPU assignment,
CUDA tensor execution, scheduling, and MP4 encoding.

For a real text-to-video test, install `.[wan]` and generate selected PDF scenes with
Wan2.1 T2V 1.3B. Model weights download to the Hugging Face cache on first use;
the official Diffusers repository is roughly 29 GB. The defaults use portrait 320×576,
9 fps and 20 inference steps, with CPU model offload for a 16 GB GPU. Frame count is
fitted to scene count: four scenes use 33 frames each and eight scenes use 17 frames each.
The local Qwen planner also downloads its model on first use (roughly 8 GB).
By default, plant lifecycle scenes use one consistent CUDA-rendered 2D style for the
dormant seed, rain, root, stem, leaves, flower, bee, and new seeds. This keeps growth
direction, framing, anatomy, and palette deterministic. Set `WAN_CONTROLLED_MOTION=0`
to compare pure Wan generation; other stories continue to use Wan normally.

```bash
GENERATOR_BACKEND=wan python -m backend.scripts.run_wan_pdf_demo test_pdfs/The_Little_Seed.pdf --scene 2 --scene 4
```

The scene numbers refer to the Qwen storyboard. This command saves the summary,
source evidence, prompts and motion descriptions in
`storage/plans/The_Little_Seed-storyboard.json`, then generates the selected clips.
To revise or rerender those scenes without rerunning Qwen, pass
`--plan-file storage/plans/The_Little_Seed-storyboard.json`.

Use `/pdf/jobs` with `GENERATOR_BACKEND=wan` to process all scenes, or run
`GENERATOR_BACKEND=wan WAN_STEPS=12 python -m backend.scripts.run_cuda_pdf_demo test_pdfs/The_Little_Seed.pdf`
to generate and concatenate the full test PDF locally. This can take
several minutes for the PDF; the 15-second latency target is not yet met. For a
smaller smoke test, set `WAN_NUM_FRAMES=17 WAN_STEPS=8 WAN_HEIGHT=448 WAN_WIDTH=256`.
The demo script also concatenates completed MP4 scenes into one `*-combined.mp4`
file in `storage/videos`. Wan PDF jobs also add the combined file to their
`/videos/{doc_id}` result. With the API running, open `/media/<filename>.mp4`
in a browser to play a generated clip or the combined video.

## Next Implementation Step

Generate the single character/state reference and one consistent keyframe per scene, then run
image-to-video animation for each keyframe. Synthesize audio from `narration`
and `narration_seconds`, add captions and transitions, and assemble the 15-second result.
Keep the PDF plan reviewable before expensive image or video generation.

## Tests

```bash
pytest
```

The suite runs without a GPU, poppler or the optional extras; the Wan
assembly test skips itself when `imageio-ffmpeg` is absent. The adaptive planner's
retry, grounding, duration, state and outfit contracts are covered with a stubbed
generator; the model path itself still needs a CUDA box to exercise end to end.
