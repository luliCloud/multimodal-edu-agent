# One-time environment setup for the audio-video sync pipeline (Windows).
# Creates both venvs (tts/'s and backend/'s) and installs the right
# dependencies for the backend you intend to use. Run this once; after it
# completes, use backend.scripts.run_synced_short_demo any time without
# repeating setup.
#
# Usage (from the repo root, in PowerShell):
#   .\setup.ps1                # backend/'s venv gets the 'wan' extra (real GPU video generation)
#   .\setup.ps1 -Backend mock  # lighter install, no GPU/diffusion deps

param(
    [ValidateSet("wan", "mock", "cuda_probe")]
    [string]$Backend = "wan"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "=== [1/4] Checking ffmpeg ===" -ForegroundColor Cyan
if (-not (Test-Command "ffmpeg")) {
    Write-Warning "ffmpeg not found on PATH. Install it before running the pipeline: 'winget install ffmpeg', or download a build and add it to PATH manually."
} else {
    Write-Host "ffmpeg found."
}

Write-Host "`n=== [2/4] Setting up tts/ venv (Chatterbox TTS) ===" -ForegroundColor Cyan
Push-Location "$RepoRoot\tts"
python -m venv venv
& ".\venv\Scripts\pip.exe" install --upgrade pip
& ".\venv\Scripts\pip.exe" install -r requirements.txt
Pop-Location

Write-Host "`n=== [3/4] Setting up backend/ venv (backend = $Backend) ===" -ForegroundColor Cyan
Push-Location $RepoRoot
python -m venv .venv
$extras = switch ($Backend) {
    "wan"         { ".[dev,wan]" }
    "cuda_probe"  { ".[dev,cuda]" }
    default       { ".[dev]" }
}
& ".\.venv\Scripts\pip.exe" install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -e $extras
Pop-Location

if ($Backend -ne "mock") {
    Write-Host "`n=== [4/4] Checking CUDA availability ===" -ForegroundColor Cyan
    & "$RepoRoot\.venv\Scripts\python.exe" -c "import torch; avail = torch.cuda.is_available(); print('CUDA available:', avail); print('Device:', torch.cuda.get_device_name(0) if avail else 'none - will fail for --backend wan/cuda_probe, which require a CUDA GPU')"
    Write-Host "`nIf CUDA available is False but you have an NVIDIA GPU: your driver or torch build is mismatched." -ForegroundColor Yellow
    Write-Host "Reinstall torch with the right CUDA version from https://pytorch.org/get-started/locally/ inside .venv." -ForegroundColor Yellow
    Write-Host "`nOn an 8GB GPU (e.g. RTX 4060), if --backend wan runs out of memory, try a smaller config:" -ForegroundColor Yellow
    Write-Host '  $env:WAN_NUM_FRAMES=17; $env:WAN_STEPS=8; $env:WAN_HEIGHT=448; $env:WAN_WIDTH=256' -ForegroundColor Yellow
}

Write-Host "`nDone. Try it:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m backend.scripts.run_synced_short_demo examples\carbon_bonding\script.json --gender female --backend $Backend"
