#!/bin/bash
# Submit centralized Oracle full-observation Zero-shot ReAct on the fixed
# 56-episode PARTNR representative subset using the same Qwen3 endpoint.

set -euo pipefail

cards="${1:-2}"
manifest="${2:-scripts/subsets/partnr_representative_v1.json}"
if [[ ! "$cards" =~ ^(1|2|4)$ ]]; then
  echo "Usage: bash scripts/submit_react_centralized_fullobs_qwen_representative_subset.sh [1|2|4] [manifest.json]" >&2
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
test -f "$manifest"
manifest="$(cd "$(dirname "$manifest")" && pwd)/$(basename "$manifest")"
episode_count="$(python3 - "$manifest" <<'PY'
import json
import sys

indices = json.load(open(sys.argv[1], encoding="utf-8"))["episode_indices"]
if not isinstance(indices, list) or not indices or len(indices) != len(set(indices)):
    raise SystemExit("Manifest must contain non-empty, unique episode_indices")
print(len(indices))
PY
)"
if (( episode_count < cards )); then
  echo "Manifest has fewer episodes than requested GPU shards." >&2
  exit 2
fi

mkdir -p logs
array_job_raw="$(sbatch --parsable --export="ALL,PARTNR_REPRESENTATIVE_MANIFEST=${manifest}" --array="0-$((cards - 1))%${cards}" scripts/react_centralized_fullobs_qwen_representative_array.sbatch)"
array_job_id="${array_job_raw%%;*}"
merge_job_raw="$(sbatch --parsable --export=NONE --dependency="afterok:${array_job_id}" scripts/merge_react_centralized_fullobs_qwen_representative_subset.sbatch "$array_job_id" "$manifest")"
merge_job_id="${merge_job_raw%%;*}"

echo "Submitted ${episode_count}-episode centralized Oracle full-observation ReAct + Qwen3 subset on ${cards} GPU shard(s): ${array_job_id}"
echo "Submitted CPU result merger: ${merge_job_id}"
echo "Monitor: squeue -j ${array_job_id},${merge_job_id}"
echo "Merged result: logs/react-qwen-fullobs-representative-${array_job_id}_merged/results/summary.json"
