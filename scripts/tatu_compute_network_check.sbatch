#!/bin/bash
# Minimal no-key diagnostic: verifies network transport from an allocated GPU
# node to the Tatu OpenAI-compatible endpoint.  It never calls a model.
#SBATCH --job-name=tatu-net-check
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load anaconda3
module load shangwang
source activate "$HOME/env/partnr"
cd "$HOME/projects/partnr-planner"
mkdir -p logs

echo "node=$(hostname)"
echo "proxy variables:"
env | grep -iE "(^|_)(http|https|all)_proxy" || true

for host in maas.tatucloud.com maas.tiangongsuanli.com; do
  endpoint="https://${host}/v1"
  echo "--- ${host}: DNS ---"
  getent hosts "$host" || true
  echo "--- ${host}: GET /v1 ---"
  curl -sS -o /dev/null -w "http=%{http_code} connect=%{time_connect}s total=%{time_total}s\n" \
    --connect-timeout 15 --max-time 30 "$endpoint" || true
  echo "--- ${host}: POST /v1/chat/completions without credentials (401/403 proves transport) ---"
  curl -sS -o /tmp/tatu-net-response-"$SLURM_JOB_ID".txt \
    -w "http=%{http_code} connect=%{time_connect}s total=%{time_total}s\n" \
    --connect-timeout 15 --max-time 30 -X POST "$endpoint/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' || true
  head -c 400 /tmp/tatu-net-response-"$SLURM_JOB_ID".txt 2>/dev/null || true
  echo
  rm -f /tmp/tatu-net-response-"$SLURM_JOB_ID".txt
done

echo "--- Python httpx 0.27 transport to official documented endpoint ---"
python - <<'PY'
import httpx

try:
    response = httpx.post(
        "https://maas.tiangongsuanli.com/v1/chat/completions",
        json={"model": "deepseek-v4-flash-0731", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
        timeout=30,
        trust_env=True,
    )
    print(f"httpx={httpx.__version__} status={response.status_code}")
    print(response.text[:400])
except Exception as exc:
    print(f"httpx={httpx.__version__} error={type(exc).__name__}: {exc}")
PY
