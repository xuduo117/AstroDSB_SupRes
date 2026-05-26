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
# NEW: UNIVERSAL ARCSINH NORMALIZATION
# =========================================================
def apply_arcsinh_norm(patch_raw):
    """Applies the 10% Headroom Arcsinh transform."""
    i_median = np.nanmedian(patch_raw)
    i_std = np.nanstd(patch_raw) + 1e-8
    
    i_min_bound = np.nanpercentile(patch_raw, 0.1)
    i_max_bound = np.nanpercentile(patch_raw, 99.9)
    
    in_tf = np.arcsinh((patch_raw - i_median) / i_std)
    
    tf_min = np.arcsinh((i_min_bound - i_median) / i_std)
    tf_max = np.arcsinh((i_max_bound - i_median) / i_std)
    N_scale = (tf_max - tf_min) + 1e-8
    
    patch_norm = 0.1 + 0.8 * (in_tf - tf_min) / N_scale
    patch_norm = np.clip(patch_norm, 0.0, 1.0)
    
    return patch_norm, i_median, i_std, tf_min, N_scale

def reverse_arcsinh_norm(pred_norm, i_median, i_std, tf_min, N_scale):
    """Reverses the prediction back to physical linear scale."""
    tf_data = ((pred_norm - 0.1) / 0.8) * N_scale + tf_min
    pred_physical = np.sinh(tf_data) * i_std + i_median
    return pred_physical
# =========================================================

@torch.no_grad()
def process_large_map(cfg, raw_map_linear, model, diffusion, device, stride=64, upsample_factor=1.0):
    """Handles data sanitization, dynamic sliding window, and seamless stitching."""
    
    # ---------------------------------------------------------
    # 1. DATA SANITIZATION
    # ---------------------------------------------------------
    print("Sanitizing raw data (Removing NaNs and Infs)...")
    # Notice: We NO LONGER clip negative values! Arcsinh needs them for the CMB.
    raw_map_linear = np.nan_to_num(raw_map_linear, nan=0.0, posinf=0.0, neginf=0.0)
    
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
                
                # Apply the Arcsinh math
                patch_norm, i_median, i_std, tf_min, N_scale = apply_arcsinh_norm(patch)
                t_cond = torch.from_numpy(patch_norm).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # Run Model
                if cfg.model.get('type', 'dsb').lower() == 'ddpm':
                    pred_tensor = diffusion.solve_sde(model, shape=t_cond.shape, cond_img=t_cond, skip_steps=50)
                else:
                    pred_tensor = diffusion.ddpm_sampling(model, x_start=t_cond, cond_img=t_cond, skip_steps=50, verbose=False)
                
                pred_norm = pred_tensor.squeeze().cpu().numpy()
                
                # Reverse the math to true physical units
                pred_physical = reverse_arcsinh_norm(pred_norm, i_median, i_std, tf_min, N_scale)
                
                # Blend
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