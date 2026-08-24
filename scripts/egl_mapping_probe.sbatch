#!/bin/bash
# One-GPU, no-API probe for CUDA/EGL device mapping under Slurm.
# It creates a minimal Habitat-Sim headless context and exits.
#SBATCH --job-name=partnr-egl-probe
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load anaconda3
source activate "$HOME/env/partnr"
# The 12.8 toolkit is loaded only for this diagnosed CUDA-runtime failure.
# It does not change the installed PyTorch (CUDA 12.4) package.
module load cuda/12.8
cd "$HOME/projects/partnr-planner"
mkdir -p logs

export HABITAT_NVIDIA_GL="$HOME/opt/nvidia-gl/570.211.01/extracted/NVIDIA-Linux-x86_64-570.211.01-no-compat32"
test -f "$HABITAT_NVIDIA_GL/libEGL.so.1"
test -f "$HABITAT_NVIDIA_GL/libOpenGL.so.0"
export LD_LIBRARY_PATH="$HABITAT_NVIDIA_GL:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/scripts/nvidia_egl_vendor.json"
test -f "$__EGL_VENDOR_LIBRARY_FILENAMES"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "host=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
env | grep -E '^(CUDA|NVIDIA|SLURM_.*GPU|GPU_DEVICE)' | sort || true
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv,noheader
echo "--- device files ---"
ls -l /dev/nvidia* 2>/dev/null || true

python - <<'PY'
import ctypes
import torch
import habitat_sim

cuda_driver = ctypes.CDLL("libcuda.so.1")
device_count = ctypes.c_int()
print("libcuda.cuInit =", cuda_driver.cuInit(0))
print("libcuda.cuDeviceGetCount =", cuda_driver.cuDeviceGetCount(ctypes.byref(device_count)))
print("libcuda.device_count =", device_count.value)
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch.cuda.device_0 =", torch.cuda.get_device_name(0))
    print("torch.cuda.current_device =", torch.cuda.current_device())
else:
    try:
        torch.cuda.init()
    except Exception as error:
        print("torch.cuda.init_error =", repr(error))

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "NONE"
sim_cfg.enable_physics = False
sim_cfg.gpu_device_id = 0
agent_cfg = habitat_sim.agent.AgentConfiguration()
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
print("HABITAT_EGL_CONTEXT_OK")
sim.close()
PY
