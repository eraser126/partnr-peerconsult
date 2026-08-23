#!/bin/bash
# Three-episode, one-GPU diagnostic run for API completion length and usage.
# Credentials are inherited from the terminal that invokes sbatch.
#SBATCH --job-name=partnr-peer-token-smoke
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the submitting shell.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL in the submitting shell.}"
export PARTNR_MODEL="${PARTNR_MODEL:-deepseek-v4-flash-0731}"

module load anaconda3
module load shangwang
source activate "$HOME/env/partnr"
cd "$HOME/projects/partnr-planner"
mkdir -p logs

export HABITAT_NVIDIA_GL="$HOME/opt/nvidia-gl/570.211.01/extracted/NVIDIA-Linux-x86_64-570.211.01-no-compat32"
export LD_LIBRARY_PATH="$HABITAT_NVIDIA_GL:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
test -n "${https_proxy:-}"

echo "Started: $(date -Is)"
echo "Episodes: dataset indices [0, 1, 2]"
echo "API max_tokens: omitted (provider default)"

python -m habitat_llm.examples.planner_demo \
  --config-name=baselines/peer_consult_v2_zero_shot_react_summary \
  hydra.run.dir="logs/peerconsult-tatu-token-smoke-${SLURM_JOB_ID}" \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
  +episode_indices='[0,1,2]' \
  num_proc=1 \
  llm@evaluation.agents.agent_0.planner.plan_config.llm=tatu_chat \
  llm@evaluation.agents.agent_1.planner.plan_config.llm=tatu_chat \
  evaluation.agents.agent_0.planner.plan_config.constrained_generation=False \
  evaluation.agents.agent_1.planner.plan_config.constrained_generation=False \
  evaluation.log_data=True \
  evaluation.log_detailed_traces=True

echo "Finished: $(date -Is)"
