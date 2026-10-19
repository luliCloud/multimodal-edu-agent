# Running the webapp on SJSU HPC3

These scripts reuse an existing Python 3.11 video environment and downloaded
models. They do not install software, alter drivers/networking, download model
weights, change GPU visibility, or publish the app to the internet.

The HPC1 download host, HPC3 login host (`coe-hpc3`, prompt `g17`), and GPU
compute nodes have different jobs. Prepare downloads on HPC1. Submit jobs from
HPC3. Run models only inside a Slurm allocation. Run the frontend on your Mac.

## Files and environments

| Path | Purpose |
| --- | --- |
| `.venv/` | Existing backend, Qwen, Wan, and VACE environment |
| `tts/venv/` | Separate, fully isolated Chatterbox environment |
| `.hpc.env` | Local settings; never commit credentials or tokens |
| `cache/huggingface/` | Existing model downloads; reused without copying |
| `downloads/` | Offline wheel packages |
| `storage/` | Uploaded PDFs, scripts, visual assets, generated video/audio |
| `logs/webapp-JOB_ID.log` | Backend, Slurm step, and reverse-tunnel output |
| `tmp/`, `cache/` | Project-local temporary files and other caches |

Do not recreate either environment when reconnecting. The environment script
keeps project package overrides before the shared Python module packages. The
TTS wrapper removes that `PYTHONPATH` before starting Chatterbox, because its
Transformers/Diffusers versions differ from the video environment.

## One-time configuration

In your existing checkout on HPC3:

```bash
cd "$HOME/projects/multimodal-edu-agent"
cp scripts/hpc/config.example.sh .hpc.env
```

Edit `.hpc.env` locally. It is trusted Bash configuration, not a dotenv file.
For the Python module used during the October 2026 setup, add:

```bash
HPC_SHARED_SITE_PACKAGES=/app/rl9.x/python3/3.11.7/lib/python3.11/site-packages
```

Confirm the directory still contains the cluster's working CUDA-enabled torch
installation; module locations can change. If the active module already exposes
torch, automatic discovery also works. No personal user ID is embedded in the
scripts. Paths are derived from the checkout.

`HPC_MODEL_CACHE` defaults to this checkout's `cache/huggingface`. Point it at
an existing approved cache if needed. The environment uses a cached `refs/main`
revision or the sole snapshot for each model. If several pinned snapshots exist
without a `refs/main`, set `PLANNER_MODEL_ID`, `WAN_MODEL_ID`, and
`WAN_REFERENCE_MODEL_ID` to explicit local snapshot directories in `.hpc.env`.
This does not select the newest directory by timestamp or contact Hugging Face.

The model repositories used during setup were:

- `Qwen/Qwen3-4B-Instruct-2507`
- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-VACE-1.3B-diffusers`

`TTS_ENABLED=0` is the default. Enable it only after installing and testing
Chatterbox in `tts/venv` and downloading its required model weights. The presence
of Qwen/Wan weights does not provide Chatterbox weights.

Check paths and dependencies without starting inference:

```bash
source scripts/hpc/env.sh
python -m pip check
scripts/hpc/tts-python -m pip check  # after the TTS environment is installed
```

The scripts do not edit `.bashrc`, shared modules, or other users' files. Most
runtime outputs are directed into this checkout; OS/Slurm/SSH accounting and
normal SSH host-key records remain system-managed. Downloaded model snapshots
and virtual environments can contain symlinks: do not follow them when cleaning
up project files.

## Package and model preparation

HPC3 HTTPS connections failed during setup, while HPC1 could reach PyPI and
Hugging Face. Download Linux x86_64 CPython 3.11 wheels on HPC1 into `downloads/`
and install them offline on HPC3. Native wheels must match HPC3, not a Mac or
HPC1's older CentOS runtime. Pure-Python source dependencies can be built into
wheels separately. Keep TTS wheels in `downloads/tts` and its packages isolated
from `.venv`. Do not run the generic `setup.sh` over an already working HPC
installation just to reconnect.

Before downloading models, confirm your quota and approved storage location
with your advisor/HPC support. Shared free space is not your personal quota.
Use HPC1's separate `.download-venv` for downloads, not the HPC3 runtime venvs.
Set `HF_HOME` and `HF_XET_CACHE` into the approved project cache on the download
host too. The runtime defaults to offline mode, so missing files fail rather
than silently starting a large download on a compute/login node.

## Submit one GPU backend job

Connect to the SJSU VPN, SSH to HPC1 using your own SJSU ID, then SSH to HPC3.
From the HPC3 checkout:

```bash
bash scripts/hpc/submit.sh
```

The default requests one A100, four CPU cores, 32 GiB of host RAM, and two hours
in `gpuqs`. Slurm assigns the node. These are starting resources from the tested
setup, not a guarantee for every model/workload. Check `sinfo` for current queues
and limits. Override resources with normal `sbatch` options:

```bash
bash scripts/hpc/submit.sh --time=01:00:00 --mem=48G
```

During setup, CUDA allocation failed on `cs002` but succeeded on other A100
nodes. That was a point-in-time observation, not a permanent blacklist. If the
same failure recurs, report it and optionally exclude that node for the test:

```bash
bash scripts/hpc/submit.sh --exclude=cs002
```

Monitor only your job:

```bash
squeue -u "$USER"
tail -f logs/webapp-JOB_ID.log
```

The job verifies an actual tiny CUDA operation, starts one Uvicorn process on
the compute node's `127.0.0.1:18000`, and establishes an authenticated reverse
SSH tunnel to the HPC3 login node's `127.0.0.1:18000`. This avoids the rejected
compute-node SSH login observed during setup. SSH uses the existing key and
normal host-key checks. It needs to authenticate without an interactive prompt;
preconfigure the authorized host/key relationship using the cluster's guidance.
The scripts do not turn off host-key checking or circumvent Slurm allocations.

If the API or tunnel exits, the job stops its other child. Cancellation/time
limit signals also trigger cleanup. A port collision fails; the script never
kills another process to claim the port. Choose a different `HPC_API_PORT` in
`.hpc.env` if needed, and adapt the forwarding destinations below.

## Connect the Mac frontend

The following defaults assume the job's reverse tunnel on HPC3 uses port 18000.
Local port 18001 avoids the stale port 18000 tunnel encountered during setup.
You need your VPN, one Mac tunnel tab, and one frontend tab.

In a Mac terminal, set your own ID and run the nested forward:

```bash
SJSU_ID=YOUR_SJSU_ID
ssh -t -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18001:127.0.0.1:18001 \
  "$SJSU_ID@coe-hpc1.sjsu.edu" \
  'ssh -N -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18001:127.0.0.1:18000 coe-hpc3'
```

Enter your campus password only at the SSH prompt. Leave the tunnel running;
silence is normal. No API listener binds to `0.0.0.0`. Loopback forwarding is
for your development session, not a substitute for authentication if this app
is later offered to other users or publicly hosted.

In another Mac terminal:

```bash
curl --max-time 10 http://127.0.0.1:18001/api/health
cd ~/multimodal-edu-agent/webapp/frontend
npm ci  # first install, or when package-lock.json changes
BACKEND_URL=http://127.0.0.1:18001 npm run dev -- --hostname 127.0.0.1
```

Open the address printed by Next.js, normally `http://127.0.0.1:3000`. If an old
frontend is already running, stop it with Ctrl+C in its own tab and restart with
the new `BACKEND_URL`. Do not start multiple Next.js processes in the same checkout.
Keep the Mac frontend's code in sync with the backend version before testing new
features. Installing on HPC does not update your Mac checkout.

## Audio, timing, and visual quality

- Enable `TTS_ENABLED=1` in `.hpc.env` only after isolated TTS import and offline
  generation tests pass. `TTS_PYTHON` defaults to the isolation wrapper, not the
  raw TTS interpreter. The backend passes its environment to subprocesses, so
  simply having two venv directories is not enough to prevent path contamination.
- Narration is embedded in the final combined MP4; scene previews remain silent.
- `WAN_CONTROLLED_MOTION=1` retains the repository default for recognized plant
  lifecycle animation. Set it to `0` deliberately when testing learned Wan
  inference instead. Record which path produced a demonstration.
- The environment clears `WAN_NUM_FRAMES` for adaptive scene timing. Use
  `HPC_FIXED_NUM_FRAMES` only for a deliberate fixed-length test. Setting 33
  frames at 9 fps gives about 3.67 seconds **per scene**, not per story.
- Review scenes before rendering. An exported seed story had ten scenes with
  two duplicated opening scenes; removing the duplicates restored eight stages.
  The web editor can render scripts outside the strict planner contract, so
  read its review warnings. An output file is not proof of story accuracy,
  character consistency, or exact duration, especially with real speech timings.

## Reconnect and stop

After VPN loss, reconnect and check `squeue -u "$USER"` on HPC3 first. An `sbatch`
job can outlive the terminal, so do not assume it ended or request another GPU
without checking. If the job is still running, recreate the Mac tunnel and test
health. If it ended, submit one new job. Existing files remain saved.

For a deliberate shutdown, stop the Mac frontend and tunnel with Ctrl+C in
their respective tabs, then cancel your own job on HPC3:

```bash
scancel YOUR_JOB_ID
```

Cancel/restart the job after changing `.hpc.env`; a running API does not inherit
later shell changes. No broad `pkill`, firewall edits, GPU resets, or forwarding
sysctl changes are part of this workflow.

## Local script checks

```bash
bash -n scripts/hpc/env.sh scripts/hpc/tts-python scripts/hpc/submit.sh scripts/hpc/webapp.slurm
python3.11 -m unittest discover -s scripts/hpc/tests -v
```

The tests use temporary environments and stubbed Slurm/SSH/GPU commands. They
verify path precedence, TTS isolation, local snapshot selection, refusal to run
the launcher outside an allocation, and backend cleanup when the tunnel fails.
They do not establish actual cluster resource access, network policy approval,
Chatterbox model compatibility, or successful live Slurm/SSH operation. Perform
a fresh short cluster job before relying on this launcher for a demonstration.
