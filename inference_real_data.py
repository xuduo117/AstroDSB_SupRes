import os
import yaml
import argparse
import torch
import numpy as np
import math
from tqdm import tqdm
from easydict import EasyDict as edict
import torch.nn.functional as F

# Modular Components
from core.factory import get_model
from core.diffusion import AstroDiffusionBridge, DDPMHandler

def create_hann_window(window_size=256):
    """Creates a 2D bell curve to seamlessly blend overlapping patches."""
    window_1d = np.hanning(window_size)
    window_2d = window_1d[:, None] * window_1d[None, :]
    return window_2d

# =========================================================
# ASYMMETRIC DILUTION-AWARE ARCSINH (SMART BOUNDS)
# =========================================================
def apply_patch_asymmetric_arcsinh(patch_centered, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0):
    """
    Applies the exact Smart Bounds normalizer used during training.
    Expects the patch to already have its median subtracted.
    """
    c_math = 0.0  # Training math anchors to 0.0
    
    s = np.nanstd(patch_centered) + 1e-8
    alpha = s * transition_sigmas
    tf_patch = np.arcsinh(patch_centered / alpha)
    
    # 1. Get raw statistics
    abs_min, abs_max = np.nanmin(patch_centered), np.nanmax(patch_centered)
    perc_min, perc_max = np.nanpercentile(patch_centered, 0.1), np.nanpercentile(patch_centered, 99.9)
    sigma_min, sigma_max = c_math - max_sigma * s, c_math + max_sigma * s
    
    # 2. SMART BOUNDS
    p_min = max(abs_min, min(perc_min, sigma_min))
    p_max = min(abs_max, max(perc_max, sigma_max))
    
    # 3. Apply the 2.5x Super-Resolution Headroom
    max_phys = headroom_multiplier * max(0, p_max)
    min_phys = headroom_multiplier * min(0, p_min) 
    
    t_min = np.arcsinh(min_phys / alpha)
    t_max = np.arcsinh(max_phys / alpha)
    
    if t_max - t_min < 1e-6:
        t_max = t_min + 1.0
        
    # 4. Squeeze exactly into [0.0, 1.0]
    final_norm = (tf_patch - t_min) / (t_max - t_min)
    final_norm = np.clip(final_norm, 0.0, 1.0)
    
    return final_norm, alpha, t_min, t_max

def reverse_patch_asymmetric_arcsinh(pred_norm, c_median, alpha, t_min, t_max):
    """Reverses the [0, 1] squeeze, then reverses Arcsinh, then restores the background."""
    # 1. Reverse the [0, 1] squeeze
    tf_data = pred_norm * (t_max - t_min) + t_min
    
    # 2. Reverse Arcsinh 
    pred_centered = np.sinh(tf_data) * alpha
    
    # 3. Add the absolute physical background back!
    pred_physical = pred_centered + c_median
    return pred_physical

# =========================================================

@torch.no_grad()
def process_large_map(cfg, raw_map_linear, model, diffusion, device, stride=64, upsample_factor=1.0):
    """Handles data sanitization, dynamic sliding window, and seamless stitching."""
    
    # ---------------------------------------------------------
    # 1. DATA SANITIZATION
    # ---------------------------------------------------------
    print("Sanitizing raw data (Removing NaNs and Infs)...")
    # Clean anomalies, but DO NOT clip negative values!
    raw_map_linear[~np.isfinite(raw_map_linear)] = np.nan
    raw_map_linear = np.nan_to_num(raw_map_linear, nan=0.0)
    
    # ---------------------------------------------------------
    # 2. CONDITIONAL UPSAMPLING
    # ---------------------------------------------------------
    h_raw, w_raw = raw_map_linear.shape
    
    if upsample_factor != 1.0:
        h_hr = int(math.ceil(h_raw * upsample_factor))
        w_hr = int(math.ceil(w_raw * upsample_factor))
        print(f"Upsampling map by {upsample_factor}x (from {h_raw}x{w_raw} to {h_hr}x{w_hr})...")
        
        raw_tensor = torch.from_numpy(raw_map_linear).float().unsqueeze(0).unsqueeze(0)
        hr_base_tensor = F.interpolate(raw_tensor, size=(h_hr, w_hr), mode='bicubic', align_corners=False)
        hr_base = hr_base_tensor.squeeze().numpy()
    else:
        print(f"Map is already at target resolution ({h_raw}x{w_raw}). Skipping upsample step.")
        hr_base = raw_map_linear
        h_hr, w_hr = h_raw, w_raw
        
    # ---------------------------------------------------------
    # 3. DYNAMIC PADDING FOR ARBITRARY SHAPES
    # ---------------------------------------------------------
    window_size = 256
    
    # Calculate exact padding needed so the sliding window fits perfectly
    pad_h = 0
    if h_hr < window_size:
        pad_h += window_size - h_hr
    if (h_hr + pad_h - window_size) % stride != 0:
        pad_h += stride - ((h_hr + pad_h - window_size) % stride)

    pad_w = 0
    if w_hr < window_size:
        pad_w += window_size - w_hr
    if (w_hr + pad_w - window_size) % stride != 0:
        pad_w += stride - ((w_hr + pad_w - window_size) % stride)
    
    # Reflect padding creates a mirror image at the borders, preventing harsh lines
    hr_padded = np.pad(hr_base, ((0, pad_h), (0, pad_w)), mode='reflect')
    padded_h, padded_w = hr_padded.shape
    
    # ---------------------------------------------------------
    # 4. INITIALIZE CANVAS
    # ---------------------------------------------------------
    output_canvas = np.zeros_like(hr_padded, dtype=np.float32)
    weight_canvas = np.zeros_like(hr_padded, dtype=np.float32)
    hann_window = create_hann_window(window_size)
    
    y_steps = range(0, padded_h - window_size + 1, stride)
    x_steps = range(0, padded_w - window_size + 1, stride)
    total_patches = len(y_steps) * len(x_steps)
    
    print(f"Processing {total_patches} overlapping patches with stride {stride}...")
    
    with tqdm(total=total_patches, desc="Inferring") as pbar:
        for y in y_steps:
            for x in x_steps:
                patch = hr_padded[y:y+window_size, x:x+window_size]
                
                # --- A. CENTER THE PATCH ---
                c_median = float(np.nanmedian(patch))
                if np.isnan(c_median): c_median = 0.0
                patch_centered = patch - c_median
                
                # --- B. NORMALIZE ---
                patch_norm, alpha, t_min, t_max = apply_patch_asymmetric_arcsinh(patch_centered)
                t_cond = torch.from_numpy(patch_norm).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # --- C. INFERENCE ---
                if cfg.model.get('type', 'dsb').lower() == 'ddpm':
                    # DDPM now natively handles [0, 1] data thanks to our loss weighting!
                    # eta=0.0 ensures deterministic (noise-free) sampling.
                    pred_tensor = diffusion.solve_sde(model, shape=t_cond.shape, cond_img=t_cond, skip_steps=50, eta=0.0)
                else:
                    # DSB naturally runs in [0, 1] bounds!
                    pred_tensor = diffusion.ddpm_sampling(model, x_start=t_cond, cond_img=t_cond, skip_steps=50, verbose=False)
                
                pred_norm = pred_tensor.squeeze().cpu().numpy()
                
                # --- D. REVERSE MATH (Including adding median back) ---
                pred_physical = reverse_patch_asymmetric_arcsinh(pred_norm, c_median, alpha, t_min, t_max)
                
                # --- E. BLEND ---
                pred_blended = pred_physical * hann_window
                
                output_canvas[y:y+window_size, x:x+window_size] += pred_blended
                weight_canvas[y:y+window_size, x:x+window_size] += hann_window
                
                pbar.update(1)
                
    # ---------------------------------------------------------
    # 5. FINAL STITCHING & CROPPING
    # ---------------------------------------------------------
    # Divide fluxes by weights to perfectly normalize the arbitrary strides
    final_stitched = output_canvas / (weight_canvas + 1e-8)
    
    # Crop off the padding to return the EXACT dimensions of the input map
    final_act_map = final_stitched[:h_hr, :w_hr]
    
    return final_act_map

def main():
    parser = argparse.ArgumentParser(description="Real Observation Super-Resolution Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input_map", type=str, required=True, help="Path to raw Planck .npy file")
    parser.add_argument("--output_map", type=str, default="predicted_ACT_map.npy")
    parser.add_argument("--stride", type=int, default=64, help="Sliding window step size (Lower = smoother but slower)")
    parser.add_argument("--upsample_factor", type=float, default=1.0, help="Multiply input resolution by this factor. 1.0 skips interpolation.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = edict(yaml.safe_load(f))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = get_model(cfg, None, None).to(device)
    model.eval()

    model_type = cfg.model.get('type', 'dsb').lower()
    if model_type == 'ddpm':
        diffusion = DDPMHandler(cfg, device=device)
    else:
        diffusion = AstroDiffusionBridge(cfg, device=device)

    ckpt_path = os.path.join(cfg.experiment.base_log_dir, cfg.experiment.name, "latest.pt")
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    
    print(f"Loading raw observation from {args.input_map}...")
    raw_map = np.load(args.input_map)
    if raw_map.ndim > 2:
        raw_map = raw_map.squeeze()

    predicted_act_map = process_large_map(cfg, raw_map, model, diffusion, device, stride=args.stride, upsample_factor=args.upsample_factor)
    
    np.save(args.output_map, predicted_act_map)
    print(f"Successfully saved high-resolution physical map to {args.output_map}")

if __name__ == "__main__":
    main()