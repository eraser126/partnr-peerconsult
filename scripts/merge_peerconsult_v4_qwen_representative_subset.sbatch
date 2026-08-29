#!/bin/bash
# CPU-only merger for the fixed PARTNR representative subset.
#SBATCH --job-name=partnr-peer-v4-representative-merge
#SBATCH --partition=4090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! -f "$2" ]]; then
  echo "Usage: sbatch merge_peerconsult_v4_qwen_representative_subset.sbatch <array-job-id> <manifest.json>" >&2
  exit 2
fi

cd /data/user/hd68631/projects/partnr-planner
array_job_id="$1"
manifest="$2"
python3 scripts/merge_peerconsult_val_mini_shards.py \
  "logs/peerconsult-v4-qwen-representative-${array_job_id}_*" \
  --episode-indices-file "$manifest" \
  --method "PeerConsult V4 / Qwen3-VL-4B representative PARTNR subset"
