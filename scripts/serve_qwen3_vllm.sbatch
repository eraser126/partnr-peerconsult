#!/usr/bin/env bash
#SBATCH --job-name=qwen3-vllm-server
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_vllm_server_%j.out
#SBATCH --error=logs/qwen3_vllm_server_%j.err

# Run this job separately from PARTNR simulator jobs.  It reserves one GPU for
# the model and exposes an OpenAI-compatible endpoint to other compute nodes.
set -euo pipefail

REPO_DIR=/data/user/hd68631/projects/partnr-planner
ENV_DIR=/data/user/hd68631/env/qwen3-vllm
MODEL_DIR=/data/user/hd68631/models/Qwen3-VL-4B-Instruct
PORT="${QWEN3_VLLM_PORT:-8000}"
SERVED_MODEL="${QWEN3_VLLM_MODEL_NAME:-qwen3-vl-4b-local}"

cd "${REPO_DIR}"
mkdir -p logs
module load anaconda3
source activate "${ENV_DIR}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing Qwen3-VL weights: ${MODEL_DIR}" >&2
  exit 2
fi

HOST="$(hostname -f)"
echo "PARTNR endpoint: http://${HOST}:${PORT}/v1"
echo "PARTNR model: ${SERVED_MODEL}"
echo "Set OPENAI_API_KEY=local-qwen3-vllm in the PARTNR job."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4
