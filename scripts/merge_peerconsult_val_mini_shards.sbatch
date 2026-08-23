#!/bin/bash
# CPU-only post-processing job.  It receives only the array job ID as an
# argument and deliberately does not inherit the submitting shell's API key.
#SBATCH --job-name=partnr-peer-merge
# Use the same readily available cluster partition, but request no GPU.
#SBATCH --partition=4090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch merge_peerconsult_val_mini_shards.sbatch <array-job-id>" >&2
  exit 2
fi

cd /data/user/hd68631/projects/partnr-planner
array_job_id="$1"
python3 scripts/merge_peerconsult_val_mini_shards.py \
  "logs/peerconsult-tatu-val-mini-${array_job_id}_*"
