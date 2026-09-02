#!/usr/bin/env bash
#SBATCH --job-name=qwen3-vllm-server
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
# Keep the serving endpoint alive throughout a multi-shard evaluation.  The
# renderer arrays may run for several hours and cannot recover from a server
# that exits at the scheduler's default short limit.
#SBATCH --time=48:00:00
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
# 16K admits the long full-observation PARTNR prompts while retaining enough
# KV-cache capacity for the two-agent V4 setup on one RTX 4090.
CONTEXT_LEN="${QWEN3_VLLM_CONTEXT_LEN:-16384}"
MAX_NUM_SEQS="${QWEN3_VLLM_MAX_NUM_SEQS:-3}"

if [[ ! "${CONTEXT_LEN}" =~ ^[1-9][0-9]*$ || ! "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "QWEN3_VLLM_CONTEXT_LEN and QWEN3_VLLM_MAX_NUM_SEQS must be positive integers." >&2
  exit 2
fi

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
echo "PARTNR context window: ${CONTEXT_LEN}; max concurrent sequences: ${MAX_NUM_SEQS}"
echo "Set OPENAI_API_KEY=local-qwen3-vllm in the PARTNR job."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len "${CONTEXT_LEN}" \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs "${MAX_NUM_SEQS}"
