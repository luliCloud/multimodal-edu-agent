#!/usr/bin/env bash
# Submit from the real checkout; Slurm copies the batch script to a spool directory.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo 'sbatch is unavailable. Submit from the HPC3 login node.' >&2
  exit 1
fi
mkdir -p "$root/logs"
export HPC_PROJECT_ROOT="$root"
exec sbatch --chdir="$root" --output="$root/logs/webapp-%j.log" \
  "$@" "$root/scripts/hpc/webapp.slurm"
