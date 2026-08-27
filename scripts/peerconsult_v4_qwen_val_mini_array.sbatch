#!/bin/bash
# One Slurm array task = one GPU = one disjoint PARTNR val_mini V4 shard.
# Submit only through submit_peerconsult_v4_qwen_val_mini.sh.
#SBATCH --job-name=partnr-peer-v4-qwen
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

case "$shard_count:$shard_index" in
  *[!0-9:]*|:*|0:*|*:) echo "Invalid Slurm array shard: $shard_index/$shard_count" >&2; exit 2 ;;
esac
if (( shard_index >= shard_count )); then
  echo "Invalid Slurm array shard: $shard_index/$shard_count" >&2
  exit 2
fi

module load anaconda3
module load shangwang
source activate "$HOME/env/partnr"
cd "$HOME/projects/partnr-planner"
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

echo "Started: $(date -Is)"
echo "Node: $(hostname)"
echo "Shard: $shard_index/$shard_count"
echo "Model: $PARTNR_MODEL"
nvidia-smi

python -m habitat_llm.examples.planner_demo \
  --config-name=baselines/peer_consult_v4_zero_shot_react_summary \
  hydra.run.dir="logs/peerconsult-v4-qwen-val-mini-${run_id}" \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
  +episode_mod_filter="[$shard_index,$shard_count]" \
  num_proc=1 \
  llm@evaluation.agents.agent_0.planner.plan_config.llm=tatu_chat \
  llm@evaluation.agents.agent_1.planner.plan_config.llm=tatu_chat \
  evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
  evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
  evaluation.log_data=True \
  evaluation.log_detailed_traces=True

echo "Finished: $(date -Is)"
