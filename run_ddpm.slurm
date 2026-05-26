#!/bin/bash
#SBATCH --job-name=SupRes_DDPMV3
#SBATCH -A rrg-pgmartin
#SBATCH --output=logs/ddpmv2_%j.out
#SBATCH --error=logs/ddpmv2_%j.err
#SBATCH --nodes=1                     # Multi-GPU on a single node
#SBATCH --gpus-per-node=4             # Request all 4 H100s
#SBATCH --cpus-per-task=32            # 8 cores per GPU (match num_workers in YAML)
#SBATCH --time=23:59:59               # Max time per job
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

# --- 3. RUN TRAINING ---
# We use torchrun to launch 4 processes (one per GPU)
# --nproc_per_node=4 corresponds to the number of GPUs requested
RANDOM_PORT=$(shuf -i 20000-65000 -n 1)

torchrun \
    --nproc_per_node=4 \
    --master_port=$RANDOM_PORT \
    main.py --config configs/SuperRes_ddpm_v3.yaml
