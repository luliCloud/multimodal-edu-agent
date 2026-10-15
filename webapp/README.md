# EduTok Studio (webapp)

A web wrapper around the two-stage PDF → script → video pipeline in `backend/`.

```text
webapp/
  backend/   FastAPI API that calls the existing pipeline (no planner or generator code is duplicated)
  frontend/  Next.js 16 + React 19 + Tailwind 4 studio UI
```

Flow: drop one PDF → watch text extraction and scene planning → review and edit the
scene script (narration, action, location, weather, camera, motion, character, style,
reorder, add or remove scenes) → generate the video → watch it, download it, or edit and
re-render.

## Run it locally

One command from the repository root. It installs missing dependencies, starts both servers,
and Ctrl+C stops both:

```bash
./webapp/start.sh                                         # dev mode, hot reload
./webapp/start.sh --prod                                  # production frontend build
PLANNER_BACKEND=qwen RENDER_BACKEND=wan ./webapp/start.sh # GPU models
API_PORT=8001 WEB_PORT=3001 ./webapp/start.sh             # other ports
```

Or run the two servers by hand in two terminals, both from the **repository root**:

```bash
# 1. API (port 8000)
source .venv/bin/activate
pip install -e ".[dev]" -r webapp/backend/requirements.txt
uvicorn webapp.backend.app.main:app --reload --port 8000
```

```bash
# 2. UI (port 3000)
cd webapp/frontend
npm install
npm run dev
```

Open http://localhost:3000. The Next server proxies `/api/*` to FastAPI (`BACKEND_URL`,
default `http://127.0.0.1:8000`), so uploads, polling and video seeking are same-origin.

Use the pipeline's existing environment variables to pick the models:

```bash
# CPU only: extractive planner + ffmpeg mock clips (the defaults)
uvicorn webapp.backend.app.main:app --port 8000

# GPU box: Qwen planner + Wan video
PLANNER_BACKEND=qwen RENDER_BACKEND=wan uvicorn webapp.backend.app.main:app --port 8000
```

Run one API process (no `--workers`). The GPU lock lives in-process, as it does for the
original API.

## Storage

Everything lives on local disk for now, and every path can be changed with an environment variable:

| Variable | Default | Contents |
| --- | --- | --- |
| `WEBAPP_DATA_DIR` | `storage/webapp/projects` | `<id>/project.json` (status, editable script, render state), `<id>/source.pdf`, `<id>/assets/` |
| `WEBAPP_VIDEO_DIR` | `storage/webapp/videos` | Rendered videos: `<id>/<render_id>/final.mp4` and `scene-NN.mp4` |
| `WEBAPP_RENDER_DIR` | `storage/webapp/render-scratch` | Pipeline scratch output, moved into the video directory after each render |
| `PRERENDERED_DIR` | `storage/demo` | Used only by `RENDER_BACKEND=directory` (below) |

**Videos are served from the video directory.** The Library page lists every video file
there, including `.mp4` files you copy in by hand (shown as *Imported*). `/api/media/<key>`
streams them with HTTP Range support, so seeking works.

**Moving to cloud storage later:** all video I/O goes through the `VideoStorage` protocol
in `backend/app/services/storage.py` (`put_file`, `url`, `local_path`, `list`,
`delete_prefix`). To switch, write an S3 or GCS class with the same methods, have `url()`
return a signed or CDN URL, and construct it in `main.py` in place of
`LocalDirectoryStorage`. The frontend only ever uses the `url` values the API returns.

### `RENDER_BACKEND`

| Value | Behaviour |
| --- | --- |
| `mock` (default) | ffmpeg colour clips, the same as `render_short_script --backend mock` |
| `cuda_probe` | Synthetic CUDA MP4, for checking the GPU path |
| `wan` | Real Wan 2.1 generation. For human characters, `assets/character_reference.png` and keyframes in the project's `assets/` dir are used exactly as the CLI does |
| `directory` | Generates nothing. Picks up a video already rendered elsewhere from `PRERENDERED_DIR/<pdf_slug>/final.mp4` or `PRERENDERED_DIR/<pdf_slug>.mp4` (the path `render_short_script` writes to). Useful when the GPU box is separate from the web host |

## How the script editor maps onto the pipeline

- Planning calls `story_planner.plan_story`, the same function `/pdf/plan` uses.
- On every save, the server renumbers scenes, recomputes `narration_seconds`, and rebuilds
  each scene's `visual_prompt` with the planner's own `scene_visual_prompt` template.
  Scenes whose prompt was hand-edited in the UI (`custom_prompt: true`) keep their text.
- Review notes come from the strict `ShortPlan` contract (word budgets, coverage and so on).
  They are hints and never block rendering:
  - If the edited script still satisfies `ShortPlan`, rendering goes through
    `upload_request_from_script`, exactly like the CLI.
  - Otherwise each scene is sent directly as a `SegmentRequest`.
- **Export script.json** downloads a file in the CLI's format, so a reviewed script can
  be rendered on another machine with `python -m backend.scripts.render_short_script`.

## API

```text
GET    /api/health                       planner/render backends, video dir
GET    /api/projects                     project summaries
POST   /api/projects                     multipart "file" (one PDF, ≤10 MB) → 202, planning starts
GET    /api/projects/{id}                full project (poll while status is extracting/planning/rendering)
PUT    /api/projects/{id}/script         {script, base_revision} → saved project + review notes (409 on stale revision)
POST   /api/projects/{id}/replan         re-run conversion from the stored PDF
POST   /api/projects/{id}/render         {scenes?: [int]} → 202, rendering starts
DELETE /api/projects/{id}                removes PDF, script and videos
GET    /api/projects/{id}/pdf            original PDF
GET    /api/projects/{id}/script.json    CLI-compatible export
GET    /api/library                      every video in the video directory
GET    /api/media/{key}[?download=true]  stream or download a video
```

Status lifecycle: `extracting → planning → draft → rendering → rendered`. A failed render
returns the project to `draft` (or `rendered` if an older video exists) and sets `error`.
A failed conversion sets `failed`. If the server restarts mid-task, the project is marked
interrupted instead of being left spinning.

## Tests

```bash
pytest webapp/backend/tests
```

The tests cover upload → plan → edit → render → range-served media → library → re-render
cleanup → delete, the `directory` backend, the round trip of a Qwen-shaped script through
the editor, and input validation.
