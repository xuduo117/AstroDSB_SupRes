import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
import numpy as np

def compute_power_spectrum(image):
    """
    Computes the 1D radial power spectrum of a 2D image.
    Mean-subtracts to remove the DC component (k=0 spike).
    """
    img_clean = image - np.mean(image)
    
    f = np.fft.fft2(img_clean)
    fshift = np.fft.fftshift(f)
    power_2d = np.abs(fshift)**2
    
    h, w = power_2d.shape
    y, x = np.indices((h, w))
    center = (h // 2, w // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    tbin = np.bincount(r.ravel(), power_2d.ravel())
    nr = np.bincount(r.ravel())
    
    radial_profile = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr!=0)
    
    return radial_profile

def save_snapshot(epoch, obs, pred, gt, mask, log_dir, prefix="val"):
    """
    Saves a 6-column comparison plot using Asymmetric SymLogNorm for high dynamic range visualization.
    """
    obs_img = obs[0, 0].cpu().numpy() 
    pred_img = pred[0, 0].cpu().numpy()
    gt_img = gt[0, 0].cpu().numpy()
    
    mask_img = mask.cpu().numpy().squeeze() 
    if mask_img.ndim > 2:
        mask_img = mask_img[0]
    
    error = pred_img - gt_img
    
    # ========================================================
    # DYNAMIC RANGE COLOR NORMALIZATION (Asymmetric SymLogNorm)
    # ========================================================
    # 1. Asymmetric Bounds: Just like our dataloader normalizer, we DO NOT force 
    # the color scale to be symmetric. We map exactly to the physical min/max.
    vmin = np.nanpercentile(gt_img, 0.1)
    vmax = np.nanpercentile(gt_img, 99.9)
    
    # Safety catch for flat images
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
        
    # linthresh is the point where the plot switches from Linear to Log.
    # The standard deviation is the perfect physical boundary for this!
    linthresh = np.nanstd(gt_img) + 1e-8 
    
    shared_norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
    
    # 2. Residuals SHOULD remain symmetric (errors are distributed around zero).
    err_vmax = max(np.nanpercentile(np.abs(error), 99.9), 1e-6)
    err_norm = SymLogNorm(linthresh=linthresh, vmin=-err_vmax, vmax=err_vmax, base=10)
    
    # 3. Mask is strictly [0, 1]
    mask_norm = Normalize(vmin=0, vmax=1)
    # ========================================================

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
    images = [obs_img, pred_img, gt_img, error, mask_img]
    titles = [
        "Observation (Input)", 
        "Astro-DSB (Pred)", 
        "Sim (Ground Truth)", 
        "Residual (P-G)",
        "Validity Mask"
    ]
    cmaps = ['viridis', 'viridis', 'viridis', 'RdBu_r', 'gray'] 
    norms = [shared_norm, shared_norm, shared_norm, err_norm, mask_norm]

    # --- 1. PLOT THE IMAGES ---
    for i in range(5):
        im = axes[i].imshow(images[i], cmap=cmaps[i], origin='lower', norm=norms[i])
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].axis('off')
        
        # Add colorbar, and explicitly format log ticks if using SymLogNorm
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        if i < 4: 
            cbar.ax.set_ylabel(r'Amplitude ($\mu K$)', rotation=270, labelpad=15)

    # --- 2. PLOT THE POWER SPECTRUM (Raw Data) ---
    ps_obs = compute_power_spectrum(obs_img)
    ps_pred = compute_power_spectrum(pred_img)
    ps_gt = compute_power_spectrum(gt_img)
    
    max_k = obs_img.shape[0] // 2
    k = np.arange(1, max_k) 
    
    axes[5].loglog(k, ps_obs[1:max_k], label='Input', color='blue', alpha=0.7)
    axes[5].loglog(k, ps_pred[1:max_k], label='Prediction', color='red', lw=2)
    axes[5].loglog(k, ps_gt[1:max_k], label='Ground Truth', color='black', linestyle='--', lw=2)
    
    axes[5].set_title("1D Power Spectrum $P(k)$", fontsize=14)
    axes[5].set_xlabel("Spatial Frequency $k$", fontsize=12)
    axes[5].set_ylabel("Power", fontsize=12)
    axes[5].grid(True, which="both", ls="--", alpha=0.3)
    axes[5].legend()

    # --- 3. SAVE AND CLOSE ---
    plt.suptitle(f"Epoch {epoch} Snapshot (Physical Units & Asymmetric SymLogNorm)", fontsize=16, y=1.05)
    
    snapshot_dir = os.path.join(log_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    save_path = os.path.join(snapshot_dir, f"{prefix}_epoch_{epoch:03d}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()