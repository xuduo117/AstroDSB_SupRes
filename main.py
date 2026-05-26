import os
import yaml
import argparse
import torch
import torch.distributed as dist
from easydict import EasyDict as edict

# Modular Components
from data.factory import get_dataloaders
from core.factory import get_model, get_trainer
from core.diffusion import AstroDiffusionBridge, DDPMHandler
from torch.utils.data import DistributedSampler
from utils.util import set_seed 
from logger import Logger       

# Optimization for H100: Set TF32 precision for faster matmuls
torch.set_float32_matmul_precision('high')

def train_worker(cfg):
    # 1. Capture environment variables from torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"Local rank {local_rank} exceeds visible GPUs. Check torchrun parameters.")

    torch.cuda.set_device(local_rank)
    
    # 2. Init DDP
    if cfg.training.get('distributed', False):
        cfg.local_rank, cfg.global_rank, cfg.global_size = local_rank, global_rank, world_size
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
    else:
        cfg.local_rank, cfg.global_rank, cfg.global_size = 0, 0, 1

    log = Logger(cfg.log_dir if cfg.global_rank == 0 else None)
    log.info(f"Worker {cfg.global_rank} started. Local Rank: {cfg.local_rank} | Device: cuda:{cfg.local_rank}")

    set_seed(cfg.experiment.seed + cfg.global_rank)

    # 3. Build Pipeline
    log.info("Initializing Dataloaders (Planck -> ACT)...")
    train_loader, val_loader = get_dataloaders(cfg, log)
    
    log.info("Initializing U-Net Architecture...")
    model = get_model(cfg, train_loader, log)
    
    model_type = cfg.model.get('type', 'dsb').lower()
    log.info(f"Initializing {model_type.upper()} Framework...")
    if model_type == 'ddpm':
        diffusion = DDPMHandler(cfg, device=f"cuda:{cfg.local_rank}")
    else:
        diffusion = AstroDiffusionBridge(cfg, device=f"cuda:{cfg.local_rank}")

    trainer = get_trainer(cfg, model, diffusion, train_loader, val_loader, log)
    trainer.load_checkpoint()
    
    if cfg.global_rank == 0:
        log.info(f"Starting training loop from Epoch {trainer.start_epoch} to {cfg.training.epochs}...")
    
    # ==========================================
    # NEW: GRACEFUL CRASH HANDLER
    # Prevents NCCL timeouts and zombie GPUs on the cluster!
    # ==========================================
    try:
        for epoch in range(trainer.start_epoch, cfg.training.epochs):
            if cfg.training.get('distributed', False):
                if isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                else:
                    if cfg.global_rank == 0:
                        log.info(f"[WARNING] Sampler is {type(train_loader.sampler)}. Skipping set_epoch.")
                
            train_loss = trainer.train_epoch(epoch)
            
            if cfg.global_rank == 0:
                log.info(f"Epoch [{epoch}/{cfg.training.epochs}] | Avg Loss: {train_loss:.6f}")
                
    except Exception as e:
        log.info(f"[Worker {cfg.global_rank}] Caught exception: {e}. Initiating graceful shutdown...")
        raise e
    
    finally:
        # This guarantees the process group is destroyed even if the code crashes!
        if cfg.training.get('distributed', False) and dist.is_initialized():
            dist.destroy_process_group()
            if cfg.global_rank == 0:
                log.info("Distributed process group destroyed safely.")

def main():
    parser = argparse.ArgumentParser(description="Astro-DSB Planck-to-ACT Super-Resolution Trainer")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = edict(yaml.safe_load(f))

    cfg.log_dir = os.path.join(cfg.experiment.base_log_dir, cfg.experiment.name)
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)

    train_worker(cfg)

if __name__ == "__main__":
    main()