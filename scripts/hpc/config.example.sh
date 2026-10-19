# Copy to .hpc.env in the repository root. This is trusted Bash code, not dotenv.
# Do not put passwords or tokens here.

# If the cluster's Python module sets PYTHONPATH, source env.sh after loading it.
# Otherwise set this to the read-only module site-packages directory that contains
# the working CUDA-enabled torch installation:
# HPC_SHARED_SITE_PACKAGES="/path/to/module/lib/python3.11/site-packages"

# The default is cache/huggingface inside this checkout. You can point at an
# approved existing cache to avoid copying model weights.
# HPC_MODEL_CACHE="/path/to/existing/project/cache/huggingface"
# Set explicit snapshot paths when the cache contains multiple revisions:
# PLANNER_MODEL_ID="/path/to/qwen/snapshots/REVISION"
# WAN_MODEL_ID="/path/to/wan/snapshots/REVISION"
# WAN_REFERENCE_MODEL_ID="/path/to/vace/snapshots/REVISION"

PLANNER_BACKEND=qwen
RENDER_BACKEND=wan
PLANNER_MAX_ATTEMPTS=6
WAN_CONTROLLED_MOTION=1
# Leave HPC_FIXED_NUM_FRAMES unset for adaptive timing.

# Enable only after isolated TTS packages and weights pass an offline GPU test.
TTS_ENABLED=0
TTS_GENDER=female

HPC_LOGIN_HOST=coe-hpc3
HPC_API_PORT=18000
# HPC_SSH_KEY="$HOME/.ssh/id_ed25519"
