#!/bin/bash
# Re-run only the five episodes that raised PARTNR edge-case exceptions.
#SBATCH --job-name=partnr-heuristic-repair
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load anaconda3
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

echo "Started: $(date -Is)"
nvidia-smi

python -m habitat_llm.examples.planner_demo \
  --config-name=baselines/heuristic_full_obs \
  hydra.run.dir="logs/heuristic-val-mini-repair-${SLURM_JOB_ID}" \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini_repair_5.json.gz" \
  num_proc=1

echo "Finished: $(date -Is)"
