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
# INFERENCE ENGINE FOR INDIVIDUAL .NPY FILES
# =========================================================
@torch.no_grad()
def process_files(cfg, input_dir, model, diffusion, device, batch_size=16):
    """Iterates over specific .npy files in a directory and saves physical predictions."""
    
    # Grab all .npy files in the directory
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    
    # Filter out files that already start with "Recon_" to avoid processing outputs
    target_files = [f for f in all_files if not os.path.basename(f).startswith("Recon_")]
    
    if not target_files:
        print(f"[Warning] No valid target .npy files found in {input_dir}")
        return

    print(f"\nFound {len(target_files)} files to process in {input_dir}.")
    
    for file_path in target_files:
        filename = os.path.basename(file_path)
        pred_name = f"Recon_{filename}"
        save_path = os.path.join(input_dir, pred_name)
        
        if os.path.exists(save_path):
            print(f"Skipping {filename} - Prediction already exists.")
            continue

        print(f"Processing {filename}...")
        data = np.load(file_path) 
        
        # Track original shape to restore it later if it was 2D
        original_ndim = data.ndim
        
        # Force shape to [N_patches, H, W] for the batching loop
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        elif data.ndim != 3:
            print(f"  [Error] Unsupported shape {data.shape} for {filename}. Must be 2D or 3D.")
            continue
            
        num_patches = data.shape[0]
        physical_predictions = []
        
        # Process patches in mini-batches to max out GPU without OOM
        for i in tqdm(range(0, num_patches, batch_size), desc="  Batches", leave=False):
            batch_raw = data[i : i + batch_size]
            
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

        # Convert back to numpy array
        pred_out = np.array(physical_predictions, dtype=np.float32)
        
        # Restore original dimension shape if the input was just [H, W]
        if original_ndim == 2:
            pred_out = pred_out.squeeze(0)

        temp_save_path = save_path.replace(".npy", "_tmp.npy")
        
        # 1. Save to the temporary file
        np.save(temp_save_path, pred_out)
        
        # 2. Rename it instantly (this is an atomic OS operation)
        os.rename(temp_save_path, save_path)
        
        print(f"  -> Saved {pred_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Path containing the target .npy files")
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

    # Process all files in the targeted data directory
    process_files(cfg, args.data_dir, model, diffusion, device, batch_size=args.batch_size)

if __name__ == "__main__":
    main()