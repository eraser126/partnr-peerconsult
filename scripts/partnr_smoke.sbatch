#!/bin/bash
#SBATCH --job-name=partnr-smoke
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load anaconda3
source activate "$HOME/env/partnr"
cd "$HOME/projects/partnr-planner"

# Use matching NVIDIA EGL/OpenGL user-space libraries only for this job.
# They are extracted, not installed; no system library is changed.
export HABITAT_NVIDIA_GL="$HOME/opt/nvidia-gl/570.211.01/extracted/NVIDIA-Linux-x86_64-570.211.01-no-compat32"
test -f "$HABITAT_NVIDIA_GL/libEGL.so.1"
test -f "$HABITAT_NVIDIA_GL/libOpenGL.so.0"
test -f "$HABITAT_NVIDIA_GL/libGLdispatch.so.0"
export LD_LIBRARY_PATH="$HABITAT_NVIDIA_GL:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
# Only the smoke verifier uses this module.  It leaves the installed
# Habitat-Lab source untouched and prevents an invalid navmesh from stalling
# the entire val_mini verification run.
export PYTHONPATH="$PWD/scripts/smoke_patch:${PYTHONPATH:-}"
export PARTNR_SMOKE_SNAP_MAX_ITER=2
export PARTNR_SMOKE_SNAP_NUM_SAMPLES=5

nvidia-smi

binding="$CONDA_PREFIX/lib/python3.9/site-packages/habitat_sim-0.3.3-py3.9-linux-x86_64.egg/habitat_sim/_ext/habitat_sim_bindings.cpython-39-x86_64-linux-gnu.so"
test -f "$binding"
echo "Habitat-Sim GL library resolution:"
ldd "$binding" | grep -E 'libEGL\.so\.1|libOpenGL\.so\.0|libGLdispatch\.so\.0|libGL\.so\.1' || true

python - <<'PY'
import torch
import habitat_sim
import habitat

print("torch =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("gpu =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("habitat_sim = imported")
print("habitat = imported")
PY

python -m habitat_llm.examples.verify_episodes \
  --config-name examples/planner_multi_agent_demo_config.yaml \
  hydra.run.dir="logs/verify-${SLURM_JOB_ID}" \
  evaluation=centralized_evaluation_runner_multi_agent \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
  mode="data" \
  world_model.partial_obs="False" \
  evaluation.type="centralized" \
  num_proc=1
