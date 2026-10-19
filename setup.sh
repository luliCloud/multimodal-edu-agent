#!/usr/bin/env bash
# One-time environment setup for the audio-video sync pipeline (macOS/Linux).
# Creates both venvs (tts/'s and backend/'s) and installs the right
# dependencies for the backend you intend to use. Run this once; after it
# completes, use backend.scripts.run_synced_short_demo any time without
# repeating setup.
#
# Usage (from the repo root):
#   ./setup.sh           # backend/'s venv gets the 'wan' extra (real GPU video generation)
#   ./setup.sh mock      # lighter install, no GPU/diffusion deps
#   ./setup.sh cuda_probe

set -euo pipefail

BACKEND="${1:-wan}"
case "$BACKEND" in
  wan|mock|cuda_probe) ;;
  *) echo "Usage: $0 [wan|mock|cuda_probe]" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Checking ffmpeg ==="
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not found on PATH. Install it: 'brew install ffmpeg' (macOS) or 'apt install ffmpeg' (Linux)."
else
  echo "ffmpeg found."
fi

echo
echo "=== [2/4] Setting up tts/ venv (Chatterbox TTS) ==="
cd "$REPO_ROOT/tts"
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo
echo "=== [3/4] Setting up backend/ venv (backend = $BACKEND) ==="
cd "$REPO_ROOT"
python3 -m venv .venv
case "$BACKEND" in
  wan) EXTRAS=".[dev,wan]" ;;
  cuda_probe) EXTRAS=".[dev,cuda]" ;;
  *) EXTRAS=".[dev]" ;;
esac
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e "$EXTRAS"

if [ "$BACKEND" != "mock" ]; then
  echo
  echo "=== [4/4] Checking CUDA/MPS availability ==="
  .venv/bin/python -c "
import torch
cuda = torch.cuda.is_available()
mps = torch.backends.mps.is_available()
print('CUDA available:', cuda, '-', torch.cuda.get_device_name(0) if cuda else 'none')
print('MPS available:', mps, '(Apple Silicon; --backend wan needs CUDA, not MPS)')
"
  echo
  echo "On an 8GB GPU (e.g. RTX 4060), if --backend wan runs out of memory, try a smaller config:"
  echo "  WAN_NUM_FRAMES=17 WAN_STEPS=8 WAN_HEIGHT=448 WAN_WIDTH=256 python -m backend.scripts.run_synced_short_demo ..."
fi

echo
echo "Done. Try it:"
echo "  source .venv/bin/activate"
echo "  python -m backend.scripts.run_synced_short_demo examples/carbon_bonding/script.json --gender female --backend $BACKEND"
