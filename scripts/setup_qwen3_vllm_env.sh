#!/usr/bin/env bash
# Create an isolated local Qwen3-VL inference environment on the cluster.
# This intentionally does not change the existing PARTNR runtime.
set -euo pipefail

ENV_DIR="${QWEN3_VLLM_ENV_DIR:-/data/user/hd68631/env/qwen3-vllm}"
MODEL_DIR="${QWEN3_VL_MODEL_DIR:-/data/user/hd68631/models/Qwen3-VL-4B-Instruct}"
WHEELHOUSE_DIR="${QWEN3_VLLM_WHEELHOUSE:-/data/user/hd68631/wheelhouse/qwen3-vllm-wheelhouse}"

module load shangwang
module load anaconda3

# Prefer a complete, local wheelhouse.  It avoids the cluster's slow external
# proxy and makes installation reproducible.  Retain the mirror fallback only
# for an initial setup before a wheelhouse has been uploaded.
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model weights are missing: ${MODEL_DIR}" >&2
  exit 2
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  conda create --yes --prefix "${ENV_DIR}" python=3.10 pip
fi

source activate "${ENV_DIR}"
PIP_ARGS=(--upgrade --retries 12)
if compgen -G "${WHEELHOUSE_DIR}/*.whl" >/dev/null; then
  echo "Installing entirely offline from ${WHEELHOUSE_DIR}"
  PIP_ARGS+=(--no-index --find-links "${WHEELHOUSE_DIR}")
else
  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
  echo "Wheelhouse unavailable; using ${PIP_INDEX_URL} through shangwang"
  python -m pip install --upgrade --retries 12 pip setuptools wheel
fi

# vLLM supplies a CUDA-enabled PyTorch build compatible with the installed
# driver. Qwen3-VL requires a recent Transformers release.
python -m pip install "${PIP_ARGS[@]}" \
  "vllm>=0.10.0,<0.12.0" \
  "transformers>=4.57.0,<5" \
  "qwen-vl-utils>=0.0.14"

python - <<'PY'
import importlib.util
import sys

import torch
import transformers

assert importlib.util.find_spec("vllm") is not None, "vLLM import is unavailable"
assert hasattr(transformers, "Qwen3VLForConditionalGeneration"), (
    "Installed Transformers does not provide Qwen3VLForConditionalGeneration"
)
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print("vllm=installed")
PY

echo "Local Qwen3-VL environment is ready: ${ENV_DIR}"
