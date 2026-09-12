# multimodal-edu-agent
Multimodal AI Agent for Automated Educational Content & Interactive Companion Learning

This repository is being rebuilt as a Mac-friendly development skeleton for the
EduTok-style multimodal pipeline:

```text
script segments -> job queue -> worker pool -> video artifacts -> API playback
```

The local skeleton does not require CUDA. It uses a mock video generator so the
API, job lifecycle, and storage contract can be developed on macOS before the
real AnimateDiff/SVD worker is connected on a CUDA machine.

## Local Mac Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run a local demo that generates short mock `.mp4` files with `ffmpeg`:

```bash
python -m backend.scripts.run_local_demo
```

Start the API:

```bash
uvicorn backend.app.main:app --reload
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
```

Example upload request:

```json
{
  "title": "Parallel GPU Inference Demo",
  "segments": [
    {
      "title": "Segment-level parallelism",
      "text": "Each script segment becomes an independent generation job."
    },
    {
      "title": "GPU worker",
      "text": "The mock generator can later be replaced by a CUDA video worker."
    }
  ]
}
```

## Project Direction

Use the Mac for:

- FastAPI backend and schemas
- scheduler and queue contracts
- local mock worker
- tests and documentation
- frontend integration

Use NVIDIA CUDA hardware for:

- AnimateDiff / Stable Video Diffusion
- Wav2Lip or other lip-sync inference
- multi-GPU benchmarking
- latency and throughput measurements

## Tests

```bash
pytest
```
