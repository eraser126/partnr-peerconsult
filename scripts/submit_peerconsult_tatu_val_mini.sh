#!/bin/bash
# Submit 1, 2, or 4 one-GPU PARTNR PeerConsult val_mini shards and a dependent
# CPU-only result merger.  API credentials remain only in the submitting shell.
set -euo pipefail

cards="${1:-}"
if [[ ! "$cards" =~ ^(1|2|4)$ ]]; then
  echo "Usage: bash scripts/submit_peerconsult_tatu_val_mini.sh {1|2|4}" >&2
  exit 2
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in this terminal first.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL in this terminal first.}"

# A pasted command block at an interactive `read` prompt becomes part of the
# secret.  Reject it before any Slurm allocation is requested, and never echo
# the value back to the terminal or a log.
if [[ "$OPENAI_API_KEY" == *$'\n'* || "$OPENAI_API_KEY" == *$'\r'* ]]; then
  echo "OPENAI_API_KEY must be a single line. No Slurm job was submitted." >&2
  exit 2
fi

root="${HOME}/projects/partnr-planner"
cd "$root"

array_job_raw="$(sbatch --parsable --array="0-$((cards - 1))%${cards}" scripts/peerconsult_tatu_val_mini_array.sbatch)"
array_job_id="${array_job_raw%%;*}"
merge_job_raw="$(sbatch --parsable --export=NONE --dependency="afterok:${array_job_id}" scripts/merge_peerconsult_val_mini_shards.sbatch "$array_job_id")"
merge_job_id="${merge_job_raw%%;*}"

echo "Submitted ${cards} one-GPU shards as array job: ${array_job_id}"
echo "Submitted CPU result merger (runs only after every shard succeeds): ${merge_job_id}"
echo "Monitor: squeue -j ${array_job_id},${merge_job_id}"
echo "Merged result: logs/peerconsult-tatu-val-mini-${array_job_id}_merged/results/summary.json"
