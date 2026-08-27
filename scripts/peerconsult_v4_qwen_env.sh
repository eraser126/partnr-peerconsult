#!/bin/bash
# Source this file from the login shell before submitting a PeerConsult V4 job.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source scripts/peerconsult_v4_qwen_env.sh" >&2
  exit 2
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp 'Qwen API Key: ' OPENAI_API_KEY
  echo
fi

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "OPENAI_API_KEY cannot be empty." >&2
  return 2
fi
if [[ "${OPENAI_API_KEY}" == *$'\n'* || "${OPENAI_API_KEY}" == *$'\r'* ]]; then
  echo "OPENAI_API_KEY must be a single line." >&2
  unset OPENAI_API_KEY
  return 2
fi

export OPENAI_API_KEY
export OPENAI_BASE_URL='https://nucbox.boar-sirius.ts.net/v1'
export PARTNR_MODEL='qwen/qwen3-vl-4b'
