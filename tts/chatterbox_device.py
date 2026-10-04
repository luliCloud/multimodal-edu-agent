"""Device selection for Chatterbox TTS: CUDA -> MPS -> CPU fallback.

CUDA (Linux/Windows with an NVIDIA GPU) is Chatterbox's primary supported
backend, so it's used directly when available, no smoke test needed.

Apple's MPS backend is available on Mac, but some ops in Chatterbox's
inference graph aren't implemented for it, so a model can *load* fine on
"mps" and then raise a "not implemented for MPS" RuntimeError the first
time .generate() actually runs. ChatterboxTTS.from_pretrained() only
checks whether MPS exists at all, not whether it can run this model, so
we do a real smoke-test generation once and cache the verdict to disk -
otherwise every single CLI invocation would re-pay for a full sampling
pass just to re-confirm what we already know.

Everywhere else (including Windows without a usable CUDA/torch build)
falls back to CPU - slower, but works the same, just takes longer per line.
"""
import json
import sys
from pathlib import Path

import torch

DEVICE_CACHE_PATH = Path(__file__).parent / ".chatterbox_device_cache.json"


def _read_cached_device():
    if not DEVICE_CACHE_PATH.exists():
        return None
    try:
        return json.loads(DEVICE_CACHE_PATH.read_text()).get("device")
    except (json.JSONDecodeError, OSError):
        return None


def _write_cached_device(device):
    try:
        DEVICE_CACHE_PATH.write_text(json.dumps({"device": device}))
    except OSError:
        pass  # non-fatal - we just re-detect next time


def load_chatterbox_model(quiet=False):
    from chatterbox.tts import ChatterboxTTS

    def log(msg):
        if not quiet:
            print(msg, file=sys.stderr)

    cached_device = _read_cached_device()
    if cached_device == "cuda" and torch.cuda.is_available():
        log("Using cached device: cuda (delete "
            f"{DEVICE_CACHE_PATH.name} to re-check)")
        return ChatterboxTTS.from_pretrained(device="cuda"), "cuda"
    if cached_device == "mps" and torch.backends.mps.is_available():
        log("Using cached device: mps (skipping smoke test; delete "
            f"{DEVICE_CACHE_PATH.name} to re-check)")
        return ChatterboxTTS.from_pretrained(device="mps"), "mps"
    if cached_device == "cpu":
        log("Using cached device: cpu (skipping MPS/CUDA check; delete "
            f"{DEVICE_CACHE_PATH.name} to re-check)")
        return ChatterboxTTS.from_pretrained(device="cpu"), "cpu"

    if torch.cuda.is_available():
        log("CUDA GPU detected, using it.")
        model = ChatterboxTTS.from_pretrained(device="cuda")
        _write_cached_device("cuda")
        return model, "cuda"

    if torch.backends.mps.is_available():
        try:
            log("Attempting to load Chatterbox on MPS (one-time smoke test)...")
            model = ChatterboxTTS.from_pretrained(device="mps")
            model.generate("Testing one two three.")  # smoke test
            log("MPS works, using it. Caching this so future runs skip the check.")
            _write_cached_device("mps")
            return model, "mps"
        except Exception as exc:
            log(f"MPS failed ({exc}); falling back to CPU.")

    log("Loading Chatterbox on CPU...")
    model = ChatterboxTTS.from_pretrained(device="cpu")
    _write_cached_device("cpu")
    return model, "cpu"
