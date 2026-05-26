import os
import glob
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

# Modular Components (Ensure these point to your actual codebase)
from core.factory import get_model
from core.diffusion import AstroDiffusionBridge, DDPMHandler

# =========================================================
# ASYMMETRIC DILUTION-AWARE ARCSINH [0, 1] NATIVE
# =========================================================
def apply_patch_asymmetric_arcsinh(patch_centered, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0):
    c_math = 0.0 
    s = np.nanstd(patch_centered) + 1e-8
    alpha = s * transition_sigmas
    tf_patch = np.arcsinh(patch_centered / alpha)
    
    abs_min, abs_max = np.nanmin(patch_centered), np.nanmax(patch_centered)
    perc_min, perc_max = np.nanpercentile(patch_centered, 0.1), np.nanpercentile(patch_centered, 99.9)
    sigma_min, sigma_max = c_math - max_sigma * s, c_math + max_sigma * s
    
    p_min = max(abs_min, min(perc_min, sigma_min))
    p_max = min(abs_max, max(perc_max, sigma_max))
    
    max_phys = headroom_multiplier * max(0, p_max)
    min_phys = headroom_multiplier * min(0, p_min) 
    
    t_min = np.arcsinh(min_phys / alpha)
    t_max = np.arcsinh(max_phys / alpha)
    if t_max - t_min < 1e-6: t_max = t_min + 1.0
        
    final_norm = (tf_patch - t_min) / (t_max - t_min)
    final_norm = np.clip(final_norm, 0.0, 1.0)
    
    return final_norm, alpha, t_min, t_max

def reverse_patch_asymmetric_arcsinh(pred_norm, c_median, alpha, t_min, t_max):
    tf_data = pred_norm * (t_max - t_min) + t_min
    pred_centered = np.sinh(tf_data) * alpha
    pred_physical = pred_centered + c_median
    return pred_physical

# =========================================================
# INFERENCE ENGINE
# =========================================================
@torch.no_grad()
def process_chunks(cfg, input_dir, model, diffusion, device, batch_size=16):
    """Iterates over chunked data, processes in GPU batches, and saves physical predictions."""
    
    chunk_files = sorted(glob.glob(os.path.join(input_dir, "chunk_*.npy")))
    if not chunk_files:
        print(f"[Error] No chunks found in {input_dir}")
        return

    print(f"\nFound {len(chunk_files)} chunks in {input_dir}.")
    
    for chunk_file in chunk_files:
        chunk_name = os.path.basename(chunk_file)
        pred_name = chunk_name.replace("chunk_", "pred_chunk_")
        save_path = os.path.join(input_dir, pred_name)
        
        if os.path.exists(save_path):
            print(f"Skipping {chunk_name} - Prediction already exists.")
            continue

        print(f"Processing {chunk_name}...")
        chunk_data = np.load(chunk_file) # Shape: [N_patches, H, W]
        num_patches = chunk_data.shape[0]
        
        physical_predictions = []
        
        # Process patches in mini-batches to max out GPU without OOM
        for i in tqdm(range(0, num_patches, batch_size), desc="  Batches", leave=False):
            batch_raw = chunk_data[i : i + batch_size]
            
            batch_norm_tensors = []
            batch_memory = []
            
            # 1. Normalize each patch in the CPU batch
            for patch in batch_raw:
                c_median = float(np.nanmedian(patch))
                if np.isnan(c_median): c_median = 0.0
                patch_centered = patch - c_median
                
                p_norm, alpha, t_min, t_max = apply_patch_asymmetric_arcsinh(patch_centered)
                batch_norm_tensors.append(p_norm)
                batch_memory.append((c_median, alpha, t_min, t_max))
                
            # 2. Push to GPU and Run Model
            t_cond = torch.tensor(np.array(batch_norm_tensors), dtype=torch.float32).unsqueeze(1).to(device)
            
            if cfg.model.get('type', 'dsb').lower() == 'ddpm':
                pred_tensor = diffusion.solve_sde(model, shape=t_cond.shape, cond_img=t_cond, skip_steps=50, eta=0.0)
            else:
                pred_tensor = diffusion.ddpm_sampling(model, x_start=t_cond, cond_img=t_cond, skip_steps=50, verbose=False)
                
            pred_numpy = pred_tensor.squeeze(1).cpu().numpy() # Back to CPU [B, H, W]
            
            # 3. Un-normalize back to Physical Space instantly
            for j in range(len(pred_numpy)):
                c, alpha, t_min, t_max = batch_memory[j]
                pred_phys = reverse_patch_asymmetric_arcsinh(pred_numpy[j], c, alpha, t_min, t_max)
                physical_predictions.append(pred_phys)

        temp_save_path = save_path.replace(".npy", "_tmp.npy")
        
        # 1. Save to the temporary file
        np.save(temp_save_path, np.array(physical_predictions, dtype=np.float32))
        
        # 2. Rename it instantly (this is an atomic OS operation)
        os.rename(temp_save_path, save_path)
        
        print(f"  -> Saved {pred_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Path containing the target subfolders (e.g. ./cropped_inference_data/)")
    parser.add_argument("--batch_size", type=int, default=16, help="Patches per GPU pass.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = edict(yaml.safe_load(f))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Init Model
    model = get_model(cfg, None, None).to(device)
    model.eval()

    if cfg.model.get('type', 'dsb').lower() == 'ddpm':
        diffusion = DDPMHandler(cfg, device=device)
    else:
        diffusion = AstroDiffusionBridge(cfg, device=device)

    # Load Weights
    ckpt_path = os.path.join(cfg.experiment.base_log_dir, cfg.experiment.name, "latest.pt")
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)

    # Process all subdirectories (Targets) in the data directory
    targets = sorted([f for f in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, f))])
    
    for target in targets:
        target_path = os.path.join(args.data_dir, target)
        process_chunks(cfg, target_path, model, diffusion, device, batch_size=args.batch_size)

if __name__ == "__main__":
    main()