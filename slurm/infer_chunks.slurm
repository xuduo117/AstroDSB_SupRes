#!/bin/bash
# Example job script from the runs of the paper (Trillium, SciNet). Submit from the repository
# root, e.g. `sbatch slurm/infer_chunks.slurm`, and edit the account, modules, environment and paths
# for your cluster.
#SBATCH --job-name=Pred_DSB
#SBATCH -A rrg-pgmartin
#SBATCH --output=logs/pred_%j.out
#SBATCH --error=logs/pred_%j.err
#SBATCH --partition=compute          # Ensure you point to the H100 partition
#SBATCH --nodes=1                     # Multi-GPU on a single node
#SBATCH --gpus-per-node=1             # Request all 4 H100s
#SBATCH --cpus-per-task=24            # 8 cores per GPU (match num_workers in YAML)
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


# python inference_chunks.py --config configs/SuperRes_dsb.yaml  --data_dir ./cropped_inference_data/ --batch_size 64
# python inference_chunks.py --config configs/SuperRes_dsb_0701.yaml  --data_dir ./cropped_inference_data_0701/ --batch_size 64
python inference_chunks.py --config configs/SuperRes_dsb_0910.yaml  --data_dir ./cropped_inference_data_0910/ --batch_size 64
