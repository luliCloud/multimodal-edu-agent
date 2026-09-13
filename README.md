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
coordinate multiple processes or machines; real model inference and Redis/Celery
integration remain future work.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cuda]"
```

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
requires `pdftotext` from poppler-utils and accepts a text-based PDF as the raw request
body (up to 10 MB):

```bash
curl -X POST http://127.0.0.1:8000/pdf/keywords \
  -H 'Content-Type: application/pdf' \
  --data-binary @test_pdfs/The_Little_Seed.pdf
```

This CPU-only heuristic is a baseline for selecting topics and filters infrastructure
terms such as GPU, worker, and parallelism. It does not yet create
generation prompts; scanned PDFs need OCR first. `/pdf/plan` preserves narrative
order and page references; `/pdf/jobs` submits its scenes as video tasks.

To exercise the single-GPU worker with a synthetic CUDA-generated MP4 (this is
**not** a text-to-video model), install the optional dependencies and run:

```bash
GENERATOR_BACKEND=cuda_probe python -m backend.scripts.run_cuda_pdf_demo test_pdfs/The_Little_Seed.pdf
```

The default backend remains `mock`. The CUDA probe confirms GPU assignment,
CUDA tensor execution, scheduling, and MP4 encoding. Connect an actual video
model after the PDF scene output is reviewed.

## Next Implementation Step

Connect a selected text-to-video model to the single GPU worker, then evaluate
clip latency and memory use. Keep the PDF plan reviewable before generation.

## Tests

```bash
pytest
```
