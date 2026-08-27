#!/bin/bash
# One episode / one GPU smoke test for the current PeerConsult V4 configuration.
# Credentials are inherited from the login shell; source peerconsult_v4_qwen_env.sh first.
#SBATCH --job-name=partnr-peer-v4-smoke
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${OPENAI_API_KEY:?Source scripts/peerconsult_v4_qwen_env.sh first.}"
: "${OPENAI_BASE_URL:?Source scripts/peerconsult_v4_qwen_env.sh first.}"
export PARTNR_MODEL="${PARTNR_MODEL:-qwen/qwen3-vl-4b}"

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
echo "Episode: dataset index 0"
echo "Model: $PARTNR_MODEL"
nvidia-smi

python -m habitat_llm.examples.planner_demo \
  --config-name=baselines/peer_consult_v4_zero_shot_react_summary \
  hydra.run.dir="logs/peerconsult-v4-qwen-smoke-${SLURM_JOB_ID}" \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
  +episode_indices='[0]' \
  num_proc=1 \
  llm@evaluation.agents.agent_0.planner.plan_config.llm=peerconsult_v4_qwen_chat \
  llm@evaluation.agents.agent_1.planner.plan_config.llm=peerconsult_v4_qwen_chat \
  evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
  evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
  evaluation.log_data=True \
  evaluation.log_detailed_traces=True

echo "Finished: $(date -Is)"
