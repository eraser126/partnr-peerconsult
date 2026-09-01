#!/usr/bin/env bash
#SBATCH --job-name=qwen3-vllm-setup
#SBATCH --partition=9654
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen3_vllm_setup_%j.out
#SBATCH --error=logs/qwen3_vllm_setup_%j.err

set -euo pipefail
cd /data/user/hd68631/projects/partnr-planner
mkdir -p logs
bash scripts/setup_qwen3_vllm_env.sh
