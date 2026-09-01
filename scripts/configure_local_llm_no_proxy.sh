#!/usr/bin/env bash
# `module load shangwang` resets no_proxy.  Restore the bypass only for a
# cluster-local HTTP vLLM endpoint, while leaving public HTTPS providers alone.

if [[ "${OPENAI_BASE_URL:-}" =~ ^http://([^/:]+)(:[0-9]+)?(/|$) ]]; then
    endpoint_host="${BASH_REMATCH[1]}"
    existing_no_proxy="${NO_PROXY:-${no_proxy:-}}"
    export NO_PROXY="${existing_no_proxy:+${existing_no_proxy},}${endpoint_host},localhost,127.0.0.1"
    export no_proxy="${NO_PROXY}"
    echo "Bypassing proxy for local LLM endpoint host: ${endpoint_host}"
fi
