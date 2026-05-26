#!/bin/bash
#SBATCH --job-name=Pred_DDPM
#SBATCH -A rrg-pgmartin
#SBATCH --output=logs/pred_%j.out
#SBATCH --error=logs/pred_%j.err
#SBATCH --nodes=1                     # Multi-GPU on a single node
#SBATCH --gpus-per-node=1             # Request all 4 H100s
#SBATCH --cpus-per-task=24            # 8 cores per GPU (match num_workers in YAML)
#SBATCH --time=0:59:59               # Max time per job
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=astroduo.xu@utoronto.ca

# --- 1. CACHE & PERFORMANCE ENV ---
export TORCH_HOME="/scratch/xuduo117/torch_cache"
export PYTORCH_KERNEL_CACHE_PATH="/scratch/xuduo117/torch_cache/kernels"
export XDG_CACHE_HOME="/scratch/xuduo117/xdg_cache"

# H100 Optimization: Enable Tensor Cores and TF32
export TORCH_CUDNN_V8_API_ENABLED=1

# --- 2. ENVIRONMENT ---
module load python/3.13.2
module load cuda/12.6
module load opencv
source /home/xuduo117/py312/bin/activate


python inference_real_data.py --config configs/SuperRes_ddpm_v2.yaml  --input_map Test_data/ml_input_planck.npy --output_map Test_data/ml_ddpm_prediction_v2_2.npy --stride 8
