#!/bin/bash
# Download the official Qwen3-VL-4B-Instruct weights to Northwest-1 storage.
# This job uses CPU only; it does not start model inference.
#SBATCH --job-name=qwen3-vl-4b-download
#SBATCH --partition=9654
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
MODEL_DIR="/data/user/hd68631/models/Qwen3-VL-4B-Instruct"
USER_BIN="/data/user/hd68631/.local/bin"

module load shangwang
module load anaconda3
source activate /data/user/hd68631/env/partnr

mkdir -p logs "$MODEL_DIR" "$USER_BIN"
export PATH="$USER_BIN:$PATH"
export MODELSCOPE_CACHE="/data/user/hd68631/.cache/modelscope"

echo "Started: $(date -Is)"
echo "Downloading: $MODEL_ID"
echo "Destination: $MODEL_DIR"

# Install the downloader privately, without modifying the PARTNR Conda env.
python -m pip install --user --upgrade modelscope
modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"

test -f "$MODEL_DIR/config.json"
find "$MODEL_DIR" -maxdepth 1 -name 'model-*.safetensors' -type f | grep -q .
du -sh "$MODEL_DIR"
echo "Finished: $(date -Is)"
