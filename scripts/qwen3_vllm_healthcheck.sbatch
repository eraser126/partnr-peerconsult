#!/usr/bin/env bash
#SBATCH --job-name=qwen3-vllm-check
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/qwen3_vllm_check_%j.out
#SBATCH --error=logs/qwen3_vllm_check_%j.err

# One-GPU end-to-end check: load the downloaded weights, start an OpenAI-
# compatible local endpoint, and complete a short request. It exits afterward.
set -euo pipefail

REPO_DIR=/data/user/hd68631/projects/partnr-planner
ENV_DIR=/data/user/hd68631/env/qwen3-vllm
MODEL_DIR=/data/user/hd68631/models/Qwen3-VL-4B-Instruct
PORT="${QWEN3_VLLM_CHECK_PORT:-18000}"
CONTEXT_LEN="${QWEN3_VLLM_CONTEXT_LEN:-16384}"
MAX_NUM_SEQS="${QWEN3_VLLM_MAX_NUM_SEQS:-2}"

cd "${REPO_DIR}"
mkdir -p logs
module load anaconda3
source activate "${ENV_DIR}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

vllm serve "${MODEL_DIR}" \
  --served-model-name qwen3-vl-4b-local \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len "${CONTEXT_LEN}" \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  > "logs/qwen3_vllm_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  if curl --silent --fail "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done
curl --silent --fail "http://127.0.0.1:${PORT}/v1/models"
curl --silent --fail \
  -H 'Content-Type: application/json' \
  -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -d '{"model":"qwen3-vl-4b-local","messages":[{"role":"user","content":"Reply with only: OK"}],"temperature":0,"max_tokens":8}'
echo
python - "${PORT}" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

port = sys.argv[1]
# A 9K-token-style text prompt verifies that the configured 16K context is
# genuinely accepted, rather than merely checking that the server can start.
payload = {
    "model": "qwen3-vl-4b-local",
    "messages": [{"role": "user", "content": " token" * 9000}],
    "temperature": 0,
    "max_tokens": 8,
}
request = Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    assert response.status == 200, response.status
print("Long-context local vLLM request passed.")
PY
echo "Qwen3-VL local vLLM health check passed."
