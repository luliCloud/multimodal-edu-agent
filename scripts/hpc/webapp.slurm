#!/usr/bin/env bash
#SBATCH --job-name=edutok-web
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit with bash scripts/hpc/submit.sh; do not run on the login node.' >&2
  exit 1
fi
root="${HPC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -f "$root/scripts/hpc/env.sh" ]]; then
  echo 'Project checkout not found. Submit with scripts/hpc/submit.sh.' >&2
  exit 1
fi
cd -- "$root"
source "$root/scripts/hpc/env.sh"
python="$root/.venv/bin/python"
api_pid=''
tunnel_pid=''
cleanup() {
  trap - EXIT
  # Stop only children started by this job, never unrelated Python/SSH processes.
  if [[ -n "$tunnel_pid" ]]; then kill "$tunnel_pid" 2>/dev/null || true; fi
  if [[ -n "$api_pid" ]]; then kill "$api_pid" 2>/dev/null || true; fi
  if [[ -n "$tunnel_pid" ]]; then wait "$tunnel_pid" 2>/dev/null || true; fi
  if [[ -n "$api_pid" ]]; then wait "$api_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$python" - <<'PY'
import os
import socket
from pathlib import Path

port = int(os.environ["HPC_API_PORT"])
if not 10000 <= port <= 63999:
    raise SystemExit("Choose HPC_API_PORT between 10000 and 63999.")
with socket.socket() as listener:
    listener.bind(("127.0.0.1", port))

required = []
if os.environ["PLANNER_BACKEND"] == "qwen":
    required.append("PLANNER_MODEL_ID")
if os.environ["RENDER_BACKEND"] == "wan":
    required += ["WAN_MODEL_ID", "WAN_REFERENCE_MODEL_ID"]
if os.environ["HF_HUB_OFFLINE"] == "1":
    for name in required:
        if not os.environ.get(name) or not Path(os.environ[name]).is_dir():
            raise SystemExit(f"Missing local {name}; configure its snapshot in .hpc.env.")
if os.environ["TTS_ENABLED"].lower() in {"1", "true", "yes", "on"}:
    if not Path(os.environ["TTS_PYTHON"]).is_file():
        raise SystemExit("TTS_PYTHON is missing.")
    # No model inference here. TTS imports/cache must be tested separately first.
    if not Path("tts/venv/bin/python").is_file():
        raise SystemExit("Install the isolated tts/venv before enabling narration.")
PY

echo "Job $SLURM_JOB_ID on $(hostname); API localhost:$HPC_API_PORT"
# A tiny allocation checks usable CUDA, not merely GPU visibility. It runs
# through Slurm on the assigned device before any large model is loaded.
srun --ntasks=1 "$python" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one Slurm-visible GPU.")
value = torch.ones(1, device="cuda:0")
torch.cuda.synchronize()
print("CUDA verified:", torch.cuda.get_device_name(0), value.item(), flush=True)
PY

srun --ntasks=1 "$python" -m uvicorn webapp.backend.app.main:app \
  --host 127.0.0.1 --port "$HPC_API_PORT" &
api_pid=$!

# Wait for this job's API to be ready before advertising a tunnel.
"$python" - "$HPC_API_PORT" "$api_pid" <<'PY'
import json
import os
import sys
import time
import urllib.request

port, pid = map(int, sys.argv[1:])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(60):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("API process exited during startup; inspect this job's log.")
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
            health = json.load(response)
        if health.get("status") == "ok":
            print("Backend health:", health, flush=True)
            break
    except (OSError, ValueError):
        pass
    time.sleep(1)
else:
    raise SystemExit("API did not become healthy within the startup timeout.")
PY

# Compute-node SSH logins were rejected during setup. The assigned node makes
# an authenticated reverse connection to the login node; both listeners remain
# loopback-only. A collision fails rather than stopping someone else's tunnel.
ssh -N -T -i "$HPC_SSH_KEY" \
  -o BatchMode=yes -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R "127.0.0.1:$HPC_API_PORT:127.0.0.1:$HPC_API_PORT" \
  "$HPC_LOGIN_HOST" &
tunnel_pid=$!
echo "Reverse tunnel starting on $HPC_LOGIN_HOST localhost:$HPC_API_PORT."
echo 'Verify /api/health through your Mac tunnel before opening the frontend.'

set +e
wait -n "$api_pid" "$tunnel_pid"
status=$?
set -e
echo "Backend or tunnel ended (status $status); stopping this job's other child."
if [[ "$status" -eq 0 ]]; then status=1; fi
exit "$status"
