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
```

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

This CPU-only heuristic is a baseline for selecting topics and filters infrastructure
terms such as GPU, worker, and parallelism. Scanned PDFs need OCR first.
`/pdf/plan` preserves narrative order, page references, and per-scene keywords;
`/pdf/jobs` submits those scenes and keywords as video tasks. With the Wan
backend, local Qwen3-4B-Instruct first summarizes the extracted sentences and plans
short scenes with exact source-sentence IDs, a visual prompt, and visible
start-to-end motion. The planner assigns every source sentence to an ordered
scene group before asking Qwen to design that group's shot, so short actions
and the story ending cannot be silently dropped. This semantic plan is
reviewable in `/pdf/plan`. The
planner runs in a separate GPU process that exits before Wan inference, so the
two models do not occupy GPU memory together. Set `PLANNER_BACKEND=extractive`
to use the earlier sentence-grouping baseline. The Qwen planner currently
supports short stories with up to 32 narrative sentences; longer PDFs are
rejected with 422, and a planner that cannot produce a usable storyboard
returns 502. Text extraction currently supports English text-based PDFs, not
scanned pages. Check the generated visual prompts before an expensive video
run: model-written details can still go beyond the PDF even when the scene
order is grounded.

Every scene is checked against the pydantic contract in
`backend/app/services/storyboard_schema.py`: the summary and every
`visual_prompt`, `motion` and `narration` must be present and non-empty, the
scene count must match the planned groups, and narration must land in the
6-18 word band that fits a short scene. When a generation fails that check,
the worker re-prompts the model with the specific error while it is still
loaded, up to `PLANNER_MAX_ATTEMPTS` times (default 3), rather than losing
the run and its model load. Each scene also carries `narration` and a
`narration_seconds` estimate at roughly 150 words per minute, which is the
script input the audio stage needs. The extractive baseline fills the same
two fields with its source text, so both planners hand downstream stages the
same shape.

To exercise the single-GPU worker with a synthetic CUDA-generated MP4 (this is
**not** a text-to-video model), install the optional dependencies and run:

```bash
GENERATOR_BACKEND=cuda_probe python -m backend.scripts.run_cuda_pdf_demo test_pdfs/The_Little_Seed.pdf
```

The default backend remains `mock`. The CUDA probe confirms GPU assignment,
CUDA tensor execution, scheduling, and MP4 encoding.

For a real text-to-video test, install `.[wan]` and generate selected PDF scenes with
Wan2.1 T2V 1.3B. Model weights download to the Hugging Face cache on first use;
the official Diffusers repository is roughly 29 GB. The defaults use 33 frames
at 576×320 and 20 inference steps, with CPU model offload for a 16 GB GPU.
The local Qwen planner also downloads its model on first use (roughly 8 GB).
By default, the Wan backend uses Wan for every scene. For the seed-growth
story, set `WAN_CONTROLLED_MOTION=1` to use simple CUDA-rendered 2D animations
for root, stem, leaves, and bee scenes. Those scenes make growth direction,
leaf attachment, and the bee's head direction deterministic, but are specific
to that story and look different from the Wan clips.

```bash
GENERATOR_BACKEND=wan python -m backend.scripts.run_wan_pdf_demo test_pdfs/The_Little_Seed.pdf --scene 2 --scene 7
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
smaller smoke test, set `WAN_NUM_FRAMES=17 WAN_STEPS=8 WAN_HEIGHT=256 WAN_WIDTH=448`.
The demo script also concatenates completed MP4 scenes into one `*-combined.mp4`
file in `storage/videos`. Wan PDF jobs also add the combined file to their
`/videos/{doc_id}` result. With the API running, open `/media/<filename>.mp4`
in a browser to play a generated clip or the combined video.

## Next Implementation Step

Evaluate scene quality and visual consistency, then synthesize audio from the
`narration` and `narration_seconds` the planner now emits, and reconcile the
spoken length with the generated clip length. Keep the PDF plan reviewable
before generation.

## Tests

```bash
pytest
```

The suite runs without a GPU, poppler or the optional extras; the Wan
assembly test skips itself when `imageio-ffmpeg` is absent. The planner's
retry and validation contract is covered with a stubbed generator, so the
model path itself still needs a CUDA box to exercise end to end.
