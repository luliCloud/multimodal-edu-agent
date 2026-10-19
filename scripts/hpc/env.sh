#!/usr/bin/env bash
# Source this file from Bash; it changes only the current shell's environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source scripts/hpc/env.sh' >&2
  exit 1
fi

_hpc_environment() {
  local root python shared candidate variable repository resolved
  root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)" || return
  export HPC_PROJECT_ROOT="$root"
  # Optional, trusted shell configuration. This file must never contain passwords.
  if [[ -f "$root/.hpc.env" ]]; then
    source "$root/.hpc.env" || return
  fi
  python="$root/.venv/bin/python"
  if [[ ! -x "$python" ]]; then
    echo "Backend environment missing: $python" >&2
    return 1
  fi
  shared="${HPC_SHARED_SITE_PACKAGES:-}"
  if [[ -z "$shared" ]]; then
    # Discovery reads module metadata; it does not import torch or initialize CUDA.
    shared="$("$python" -c 'import importlib.util; from pathlib import Path; s=importlib.util.find_spec("torch"); print(Path(s.origin).parent.parent if s and s.origin else "")' 2>/dev/null)" || shared=""
  fi
  if [[ -n "$shared" && ! -d "$shared" ]]; then
    echo "HPC_SHARED_SITE_PACKAGES is not a directory: $shared" >&2
    return 1
  fi
  unset PYTHONHOME
  export VIRTUAL_ENV="$root/.venv"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  # Project overrides must precede the read-only module packages (e.g. Pillow).
  export PYTHONPATH="$root:$VIRTUAL_ENV/lib64/python3.11/site-packages:$VIRTUAL_ENV/lib/python3.11/site-packages${shared:+:$shared}"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  export USE_TF=0
  export HF_HOME="${HPC_MODEL_CACHE:-$root/cache/huggingface}"
  export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
  export HF_ASSETS_CACHE="$HF_HOME/assets"
  export XDG_CACHE_HOME="$root/cache" PIP_CACHE_DIR="$root/cache/pip"
  export TMPDIR="$root/tmp" TORCH_HOME="$root/cache/torch"
  export TRITON_CACHE_DIR="$root/cache/triton" NUMBA_CACHE_DIR="$root/cache/numba"
  export MPLCONFIGDIR="$root/cache/matplotlib" GRADIO_TEMP_DIR="$root/tmp/gradio"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export PLANNER_BACKEND="${PLANNER_BACKEND:-qwen}"
  export RENDER_BACKEND="${RENDER_BACKEND:-wan}"
  export PLANNER_MAX_ATTEMPTS="${PLANNER_MAX_ATTEMPTS:-6}"
  export WAN_CONTROLLED_MOTION="${WAN_CONTROLLED_MOTION:-1}"
  export WAN_HEIGHT="${WAN_HEIGHT:-576}" WAN_WIDTH="${WAN_WIDTH:-320}"
  export WAN_STEPS="${WAN_STEPS:-20}"
  # Fixed frames caused full-story runs to exceed their target duration.
  if [[ -n "${HPC_FIXED_NUM_FRAMES:-}" ]]; then
    export WAN_NUM_FRAMES="$HPC_FIXED_NUM_FRAMES"
  else
    unset WAN_NUM_FRAMES
  fi
  export TTS_ENABLED="${TTS_ENABLED:-0}" TTS_GENDER="${TTS_GENDER:-female}"
  export TTS_PYTHON="${TTS_PYTHON:-$root/scripts/hpc/tts-python}"
  export HPC_API_PORT="${HPC_API_PORT:-18000}"
  export HPC_LOGIN_HOST="${HPC_LOGIN_HOST:-coe-hpc3}"
  export HPC_SSH_KEY="${HPC_SSH_KEY:-$HOME/.ssh/id_ed25519}"
  export WEBAPP_DATA_DIR="$root/storage/webapp/projects"
  export WEBAPP_VIDEO_DIR="$root/storage/webapp/videos"
  export WEBAPP_RENDER_DIR="$root/storage/webapp/render-scratch"
  mkdir -p "$root/logs" "$TMPDIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" \
    "$HF_ASSETS_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
    "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$GRADIO_TEMP_DIR" || return
  # Reuse pinned, downloaded snapshots without querying the network. Explicit
  # paths in .hpc.env take precedence. Ambiguous caches require an explicit path.
  for candidate in \
    'PLANNER_MODEL_ID|Qwen/Qwen3-4B-Instruct-2507' \
    'WAN_MODEL_ID|Wan-AI/Wan2.1-T2V-1.3B-Diffusers' \
    'WAN_REFERENCE_MODEL_ID|Wan-AI/Wan2.1-VACE-1.3B-diffusers'; do
    variable="${candidate%%|*}"
    repository="${candidate#*|}"
    if [[ -z "${!variable:-}" ]]; then
      if resolved="$("$python" "$root/scripts/hpc/model-path.py" "$HF_HUB_CACHE" "$repository")"; then
        printf -v "$variable" '%s' "$resolved"
        export "$variable"
      fi
    fi
  done
}

if _hpc_environment; then
  unset -f _hpc_environment
else
  unset -f _hpc_environment
  return 1
fi
