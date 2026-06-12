import os
import yaml
import argparse
import torch
import numpy as np
import h5py
from tqdm import tqdm
from easydict import EasyDict as edict
from torch_ema import ExponentialMovingAverage

# Modular Components
from data.factory import get_dataloaders
from core.factory import get_model
from core.diffusion import AstroDiffusionBridge, DDPMHandler
from utils.util import set_seed 
from logger import Logger       

def clean_state_dict(state_dict):
    """Removes the 'module.' prefix if the model was saved via DDP."""
    clean_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        clean_dict[new_key] = v
    return clean_dict

# ==========================================
# NEW: PHYSICAL UNIT RECOVERY
# ==========================================
def unnormalize_prediction(pred_tensor, memory_tensor):
    """
    Rebuilds true physical units from the U-Net prediction using universal arcsinh inversion.
    """
    i_median = memory_tensor[:, 0].view(-1, 1, 1, 1)
    i_std = memory_tensor[:, 1].view(-1, 1, 1, 1)
    tf_min = memory_tensor[:, 2].view(-1, 1, 1, 1)
    N_scale = memory_tensor[:, 3].view(-1, 1, 1, 1)
    
    # 1. Remove the 10% Headroom 
    tf_data = ((pred_tensor - 0.1) / 0.8) * N_scale + tf_min
    
    # 2. Invert the arcsinh transform
    raw_physical_data = torch.sinh(tf_data) * i_std + i_median
    return raw_physical_data

def evaluate(cfg):
    # 1. Setup Single-GPU Environment for Evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cfg.training.distributed = False
    cfg.local_rank = 0
    cfg.global_rank = 0
    cfg.global_size = 1
    
    log = Logger(None)
    log.info(f"Starting Full Dataset Evaluation on {device}")
    set_seed(cfg.experiment.seed)

    # 2. Setup Data
    _, val_loader = get_dataloaders(cfg, log)
    
    # 3. Setup Model & Diffusion
    model = get_model(cfg, None, log).to(device)
    model.eval()

    model_type = cfg.model.get('type', 'dsb').lower()
    if model_type == 'ddpm':
        diffusion = DDPMHandler(cfg, device=device)
    else:
        diffusion = AstroDiffusionBridge(cfg, device=device)

    # 4. Load Checkpoint and EMA Weights
    ckpt_path = os.path.join(cfg.experiment.base_log_dir, cfg.experiment.name, "latest.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")
    
    log.info(f"Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(clean_state_dict(state_dict))
    
    if 'ema_state' in ckpt:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.get('ema_decay', 0.999))
        ema.load_state_dict(ckpt['ema_state'])
        ema.copy_to(model.parameters())
        log.info("Successfully applied EMA weights.")

    # 5. Output Directory Setup
    eval_dir = os.path.join(cfg.experiment.base_log_dir, cfg.experiment.name, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    obs_ch = cfg.model.get('cond_channels', cfg.model.get('in_channels', 1))
    target_ch = cfg.model.get('out_channels', 1)  

    h5_path = os.path.join(eval_dir, f"{model_type}_predictions_full.h5")
    csv_path = os.path.join(eval_dir, f"{model_type}_file_mapping.csv")

    log.info(f"Saving file mapping to: {csv_path}")
    with open(csv_path, 'w') as f:
        f.write("H5_Index,Filepath\n")
        
        # Safely extract file lists from the new CombinedSyntheticDataset wrapper
        if hasattr(val_loader.dataset, 'dust_dataset') and hasattr(val_loader.dataset, 'skirt_dataset'):
            # Concatenate both file lists for the CSV mapping
            file_list = val_loader.dataset.dust_dataset.file_list + val_loader.dataset.skirt_dataset.file_list
        elif hasattr(val_loader.dataset, 'synthetic_dataset'):
            file_list = val_loader.dataset.synthetic_dataset.file_list
        else:
            file_list = val_loader.dataset.file_list
            
        for idx, filepath in enumerate(file_list):
            f.write(f"{idx},{filepath}\n")

    # 6. Evaluation Loop (Streaming to Disk)
    with h5py.File(h5_path, 'w') as h5f:
        dset_in = h5f.create_dataset('inputs', shape=(0, obs_ch, 256, 256), maxshape=(None, obs_ch, 256, 256), dtype='float32')
        dset_gt = h5f.create_dataset('ground_truth', shape=(0, target_ch, 256, 256), maxshape=(None, target_ch, 256, 256), dtype='float32')
        dset_pred = h5f.create_dataset('prediction', shape=(0, target_ch, 256, 256), maxshape=(None, target_ch, 256, 256), dtype='float32')
        dset_idx = h5f.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int64')

        ptr = 0 
        
        with torch.no_grad():
            # ==========================================
            # FIX: Unpack memory_raw from the 6-element tuple
            # ==========================================
            for i, (x_prim_raw, x_cond_raw, y_raw, mask_raw, memory_raw, index_raw) in enumerate(tqdm(val_loader, desc=f"Evaluating {model_type.upper()}")):
                
                if x_prim_raw.dim() == 5:
                    num_patches = x_prim_raw.shape[1]
                    
                    x_prim = x_prim_raw.flatten(0, 1).to(device)
                    x_cond = x_cond_raw.flatten(0, 1).to(device)
                    y_gt = y_raw.flatten(0, 1).to(device)
                    memory = memory_raw.flatten(0, 1).to(device)
                    
                    index = index_raw.repeat_interleave(num_patches).to(device)
                else:
                    x_prim = x_prim_raw.to(device)
                    x_cond = x_cond_raw.to(device)
                    y_gt = y_raw.to(device)
                    memory = memory_raw.to(device)
                    index = index_raw.to(device)

                shape = (x_prim.shape[0], target_ch, x_prim.shape[2], x_prim.shape[3])

                # 1. Run the Model Sampler
                if model_type == 'ddpm':
                    prediction = diffusion.solve_sde(model, shape=shape, cond_img=x_cond, skip_steps=50)
                else:
                    prediction = diffusion.ddpm_sampling(model, x_prim, cond_img=x_cond, skip_steps=50, verbose=False)
                
                # ==========================================
                # NEW: Convert everything back to True Physics!
                # ==========================================
                x_cond_phys = unnormalize_prediction(x_cond, memory)
                y_gt_phys = unnormalize_prediction(y_gt, memory)
                pred_phys = unnormalize_prediction(prediction, memory)

                x_cond_np = x_cond_phys.cpu().numpy()
                y_gt_np = y_gt_phys.cpu().numpy()
                pred_np = pred_phys.cpu().numpy()

                batch_size = x_cond_np.shape[0]

                # Dynamically resize the H5 datasets
                dset_in.resize(ptr + batch_size, axis=0)
                dset_gt.resize(ptr + batch_size, axis=0)
                dset_pred.resize(ptr + batch_size, axis=0)
                dset_idx.resize(ptr + batch_size, axis=0)

                # Write data
                dset_in[ptr:ptr+batch_size] = x_cond_np
                dset_gt[ptr:ptr+batch_size] = y_gt_np
                dset_pred[ptr:ptr+batch_size] = pred_np
                dset_idx[ptr:ptr+batch_size] = index.cpu().numpy()

                ptr += batch_size

    log.info(f"Evaluation complete! {ptr} physical patches packed into {h5_path}. Mapping saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Astro-DSB Bulk Inference & Save")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = edict(yaml.safe_load(f))

    evaluate(cfg)

if __name__ == "__main__":
    main()