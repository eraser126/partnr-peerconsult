#!/bin/bash
# CPU-only merger for PeerConsult V4 Qwen val_mini array shards.
# The merge job deliberately receives no API credentials.
#SBATCH --job-name=partnr-peer-v4-merge
#SBATCH --partition=4090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch merge_peerconsult_v4_qwen_val_mini_shards.sbatch <array-job-id> <subset-episodes>" >&2
  exit 2
fi

cd /data/user/hd68631/projects/partnr-planner
array_job_id="$1"
subset_episodes="$2"
merge_args=(
  "logs/peerconsult-v4-qwen-val-mini-${array_job_id}_*"
  --method "PeerConsult V4 / Qwen3-VL-4B via OpenAI-compatible API"
)
if (( subset_episodes > 0 )); then
  merge_args+=(--max-episodes "$subset_episodes")
fi
python3 scripts/merge_peerconsult_val_mini_shards.py \
  "${merge_args[@]}"
