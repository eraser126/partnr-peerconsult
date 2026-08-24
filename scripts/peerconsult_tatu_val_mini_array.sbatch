#!/bin/bash
# One array task = one GPU = one disjoint PARTNR val_mini episode shard.
# Submit via scripts/submit_peerconsult_tatu_val_mini.sh; do not submit this
# worker directly unless SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT are set.
#SBATCH --job-name=partnr-peer-tatu
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the submitting shell; never put it in this file.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL in the submitting shell.}"
export PARTNR_MODEL="${PARTNR_MODEL:-deepseek-v4-flash-0731}"

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
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/scripts/nvidia_egl_vendor.json"
test -f "$__EGL_VENDOR_LIBRARY_FILENAMES"

# Habitat-Sim matches its requested CUDA index to an EGL physical device. On
# this shared node, make the GPU assigned by Slurm explicit.
if [[ "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ]]; then
  export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
  export GPU_DEVICE_ORDINAL="$SLURM_JOB_GPUS"
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
# `shangwang` is the platform's compute-node egress proxy.  It is not the
# user's local desktop proxy and must remain enabled for external API calls.
test -n "${https_proxy:-}"

echo "Started: $(date -Is)"
echo "Node: $(hostname)"
echo "Shard: $shard_index/$shard_count"
echo "Slurm GPU mapping: job=${SLURM_JOB_GPUS:-unset} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model: $PARTNR_MODEL"
echo "Tatu API base URL configured: yes; platform egress proxy: enabled"
nvidia-smi

# The external OpenAI-compatible API has no PARTNR grammar-constrained decoder.
# Apply this same override to the official decentralized ReAct comparison.
python -m habitat_llm.examples.planner_demo \
  --config-name=baselines/peer_consult_v2_zero_shot_react_summary \
  hydra.run.dir="logs/peerconsult-tatu-val-mini-${run_id}" \
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
