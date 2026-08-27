#!/bin/bash
# Submit 1, 2, or 4 one-GPU PeerConsult V4 Qwen val_mini shards and a
# dependent CPU-only result merger. Credentials remain in the submitting shell.

set -euo pipefail

cards="${1:-}"
if [[ ! "$cards" =~ ^(1|2|4)$ ]]; then
  echo "Usage: bash scripts/submit_peerconsult_v4_qwen_val_mini.sh {1|2|4}" >&2
  exit 2
fi
: "${OPENAI_API_KEY:?Source scripts/peerconsult_v4_qwen_env.sh first.}"
: "${OPENAI_BASE_URL:?Source scripts/peerconsult_v4_qwen_env.sh first.}"

if [[ "$OPENAI_API_KEY" == *$'\n'* || "$OPENAI_API_KEY" == *$'\r'* ]]; then
  echo "OPENAI_API_KEY must be a single line. No Slurm job was submitted." >&2
  exit 2
fi

root="$HOME/projects/partnr-planner"
cd "$root"
mkdir -p logs

array_job_raw="$(sbatch --parsable --array="0-$((cards - 1))%${cards}" scripts/peerconsult_v4_qwen_val_mini_array.sbatch)"
array_job_id="${array_job_raw%%;*}"
merge_job_raw="$(sbatch --parsable --export=NONE --dependency="afterok:${array_job_id}" scripts/merge_peerconsult_v4_qwen_val_mini_shards.sbatch "$array_job_id")"
merge_job_id="${merge_job_raw%%;*}"

echo "Submitted ${cards} one-GPU V4 Qwen shard(s) as array job: ${array_job_id}"
echo "Submitted CPU result merger: ${merge_job_id}"
echo "Monitor: squeue -j ${array_job_id},${merge_job_id}"
echo "Merged result: logs/peerconsult-v4-qwen-val-mini-${array_job_id}_merged/results/summary.json"
