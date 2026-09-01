#!/bin/bash
# One GPU processes one disjoint shard of the fixed PARTNR representative subset.
# Submit through submit_peerconsult_v4_qwen_representative_subset.sh.
#SBATCH --job-name=partnr-peer-v4-representative
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

: "${OPENAI_API_KEY:?Source scripts/peerconsult_v4_qwen_env.sh first.}"
: "${OPENAI_BASE_URL:?Source scripts/peerconsult_v4_qwen_env.sh first.}"
export PARTNR_MODEL="${PARTNR_MODEL:-qwen/qwen3-vl-4b}"

shard_index="${SLURM_ARRAY_TASK_ID:-0}"
shard_count="${SLURM_ARRAY_TASK_COUNT:-1}"
run_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${shard_index}"
manifest="${PARTNR_REPRESENTATIVE_MANIFEST:-scripts/subsets/partnr_representative_v1.json}"

if [[ ! "$shard_index" =~ ^[0-9]+$ || ! "$shard_count" =~ ^[1-9][0-9]*$ ]] || (( shard_index >= shard_count )); then
  echo "Invalid Slurm array shard: $shard_index/$shard_count" >&2
  exit 2
fi

module load anaconda3
module load shangwang
source activate "$HOME/env/partnr"
cd "$HOME/projects/partnr-planner"
source scripts/configure_local_llm_no_proxy.sh
test -f "$manifest"
mkdir -p logs

export HABITAT_NVIDIA_GL="$HOME/opt/nvidia-gl/570.211.01/extracted/NVIDIA-Linux-x86_64-570.211.01-no-compat32"
test -f "$HABITAT_NVIDIA_GL/libEGL.so.1"
test -f "$HABITAT_NVIDIA_GL/libOpenGL.so.0"
export LD_LIBRARY_PATH="$HABITAT_NVIDIA_GL:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
test -n "${https_proxy:-}"

mapfile -t episode_indices < <(python - "$manifest" "$shard_index" "$shard_count" <<'PY'
import json
import sys

manifest, shard, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
indices = json.load(open(manifest, encoding="utf-8"))["episode_indices"]
if not isinstance(indices, list) or not indices:
    raise SystemExit("Representative manifest has no episode_indices")
for position, index in enumerate(indices):
    if position % count == shard:
        print(int(index))
PY
)
if (( ${#episode_indices[@]} == 0 )); then
  echo "Shard $shard_index has no assigned representative episodes" >&2
  exit 2
fi

echo "Started: $(date -Is)"
echo "Shard: $shard_index/$shard_count"
echo "Model: $PARTNR_MODEL"
echo "Manifest: $manifest"
echo "Episode indices: ${episode_indices[*]}"
nvidia-smi

python -m habitat_llm.examples.planner_demo \
  --config-name=baselines/peer_consult_v4_zero_shot_react_summary \
  hydra.run.dir="logs/peerconsult-v4-qwen-representative-${run_id}" \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
  +episode_indices="[$(IFS=,; echo "${episode_indices[*]}")]" \
  num_proc=1 \
  llm@evaluation.agents.agent_0.planner.plan_config.llm=peerconsult_v4_qwen_chat \
  llm@evaluation.agents.agent_1.planner.plan_config.llm=peerconsult_v4_qwen_chat \
  evaluation.agents.agent_1.planner.plan_config.llm.initial_request_delay_s=3 \
  evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
  evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
  evaluation.log_data=True \
  evaluation.log_detailed_traces=True

echo "Finished: $(date -Is)"
