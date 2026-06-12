import os
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from easydict import EasyDict as edict

# --- UPDATED: Import BOTH Synthetic datasets ---
from data.datasets import SyntheticAstroDataset, SkirtAstroDataset
from data.factory import scan_and_split_files

# =========================================================
# ASYMMETRIC REVERSAL FUNCTION
# =========================================================

def reverse_patch_asymmetric_arcsinh(pred_tensor, memory_tensor):
    """Reverses the [0, 1] Asymmetric Arcsinh prediction back into physical units."""
    c = memory_tensor[:, 0].view(-1, 1, 1, 1)
    alpha = memory_tensor[:, 1].view(-1, 1, 1, 1)
    t_min = memory_tensor[:, 2].view(-1, 1, 1, 1)
    t_max = memory_tensor[:, 3].view(-1, 1, 1, 1)
    
    tf_data = pred_tensor * (t_max - t_min) + t_min
    pred_physical = torch.sinh(tf_data) * alpha + c
    return pred_physical

def plot_16_samples(dataset, name, log_dir):
    """Pulls 16 samples in Training Mode and plots them in a massive 16x5 grid."""
    print(f"\nGenerating 16-sample diagnostic for {name} (Training Mode)...")
    
    fig, axes = plt.subplots(16, 5, figsize=(25, 4 * 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    col_titles = [
        "Phys Low-Res Input\n(True Units)", 
        "Norm Low-Res\n(Network Input)", 
        "Phys High-Res Target\n(True Units)", 
        "Norm High-Res\n(Network Target)", 
        "Network Target Histogram\n(Dynamic Range Anchor)"
    ]

    for i in range(16):
        ax_row = axes[i]
        try:
            # 1. Pull the i-th sample from the dataset
            x_prim_list, x_cond_list, y_list, mask_list, memory_list, idx = dataset[i]
            
            # 2. Extract the first patch from the returned list
            x_cond = x_cond_list[0:1]
            y_gt = y_list[0:1]       
            memory = memory_list[0:1]
            
            # 3. Reverse the math to recover true Physical Units
            x_cond_phys = reverse_patch_asymmetric_arcsinh(x_cond, memory)
            y_gt_phys = reverse_patch_asymmetric_arcsinh(y_gt, memory)
            
            # Convert to Numpy for plotting
            x_cond_np = x_cond[0, 0].numpy()
            y_gt_np = y_gt[0, 0].numpy()
            x_cond_phys_np = x_cond_phys[0, 0].numpy()
            y_gt_phys_np = y_gt_phys[0, 0].numpy()
            
            # 4. Setup Dynamic SymLogNorm for the Physical Data (NaN Safe!)
            vmax = np.nanpercentile(np.abs(y_gt_phys_np), 99.9)
            if np.isnan(vmax) or vmax <= 0: 
                vmax = 1.0  # Fallback to prevent crashing
                
            linthresh = np.nanstd(y_gt_phys_np) + 1e-8
            if np.isnan(linthresh) or linthresh <= 0: 
                linthresh = 1e-5
                
            phys_norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
            
            # --- COL 0: Physical Low-Res Input ---
            im0 = ax_row[0].imshow(x_cond_phys_np, cmap='viridis', origin='lower', norm=phys_norm)
            ax_row[0].axis('off')
            plt.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)
            ax_row[0].text(-0.1, 0.5, f"Sample {i+1}", transform=ax_row[0].transAxes, 
                           fontsize=16, fontweight='bold', va='center', ha='right', rotation=90)

            # --- COL 1: Normalized Low-Res (Network Input) ---
            im1 = ax_row[1].imshow(x_cond_np, cmap='viridis', origin='lower', vmin=0, vmax=1)
            ax_row[1].axis('off')
            plt.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)

            # --- COL 2: Physical High-Res Target ---
            im2 = ax_row[2].imshow(y_gt_phys_np, cmap='viridis', origin='lower', norm=phys_norm)
            ax_row[2].axis('off')
            plt.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)

            # --- COL 3: Normalized High-Res (Network Target) ---
            im3 = ax_row[3].imshow(y_gt_np, cmap='viridis', origin='lower', vmin=0, vmax=1)
            ax_row[3].axis('off')
            plt.colorbar(im3, ax=ax_row[3], fraction=0.046, pad=0.04)

            # --- COL 4: Histogram of Normalized High-Res ---
            ax_row[4].hist(y_gt_np.flatten(), bins=100, color='purple', alpha=0.7, density=True)
            ax_row[4].set_xlim(0, 1)
            
            # Visually prove where physical 0.0 (the background) landed in the network's eyes!
            c_val = memory[0, 0].item()
            alpha_val = memory[0, 1].item()
            t_min_val = memory[0, 2].item()
            t_max_val = memory[0, 3].item()
            
            # Map physical 0.0 through the normalizer to find its plot position
            zero_tf = np.arcsinh((0.0 - c_val) / alpha_val)
            zero_norm = (zero_tf - t_min_val) / (t_max_val - t_min_val)
            
            ax_row[4].axvline(zero_norm, color='red', linestyle='--', alpha=0.8, linewidth=2)
            if i == 0:
                align = 'left' if zero_norm < 0.5 else 'right'
                pad = 0.02 if zero_norm < 0.5 else -0.02
                ax_row[4].text(zero_norm + pad, ax_row[4].get_ylim()[1]*0.9, 'Physical 0.0\n(Background)', 
                               color='red', ha=align, fontweight='bold')
            ax_row[4].grid(True, alpha=0.3)
            
            # Apply column titles only to the top row
            if i == 0:
                for col_idx in range(5):
                    ax_row[col_idx].set_title(col_titles[col_idx], fontsize=16, pad=20, fontweight='bold')
            
        except Exception as e:
            print(f"Failed on sample {i}: {e}")
            for ax in ax_row:
                ax.axis('off')
                ax.text(0.5, 0.5, "ERROR", ha='center', va='center', color='red')

    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, f"train_check_{name.replace(' ', '_')}.png")
    plt.suptitle(f"{name} Dataset - 16 Training Samples (Asymmetric)", fontsize=24, y=0.90)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f"✅ Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = edict(yaml.safe_load(f))
        
    # FORCE TRAINING MODE
    cfg.dataset.augment = True
    
    print("Initializing datasets in STRICT TRAINING MODE...")
    split_mode = cfg.dataset.get('split_mode', 'physical_ood')
    ood_keywords = cfg.dataset.get('ood_keywords', ['I100'])
    
    # 1. Init Raw Dust Dataset
    print("\n--- Scanning Raw Dust Data ---")
    dust_train_files, _ = scan_and_split_files(
        cfg.dataset.roots, 
        pattern="**/*850um_tensor.npy",
        split_mode=split_mode,
        ood_keywords=ood_keywords
    )
    
    # 2. Init SKIRT Multiband Dataset
    print("\n--- Scanning SKIRT Synthetic Data ---")
    skirt_roots = cfg.dataset.get('skirt_roots', ['/scratch/xuduo117/project_SKIRT/Observational_Galleries/'])
    skirt_train_files, _ = scan_and_split_files(
        skirt_roots, 
        pattern="**/*_gallery.npy",
        split_mode=split_mode,
        ood_keywords=ood_keywords
    )

    # Instantiate datasets
    ds_dust = SyntheticAstroDataset(cfg, dust_train_files, mode='train')
    ds_skirt = SkirtAstroDataset(cfg, skirt_train_files, mode='train')

    # 3. Setup Output Directory
    out_dir = "diagnostics"
    os.makedirs(out_dir, exist_ok=True)
    
    # 4. Generate the 16-sample grids for BOTH
    if len(ds_dust) > 0:
        plot_16_samples(ds_dust, "Raw_Dust_NoiseToClean", out_dir)
    else:
        print("⚠️ No Raw Dust training files found.")

    if len(ds_skirt) > 0:
        plot_16_samples(ds_skirt, "SKIRT_Multiband_NoiseToClean", out_dir)
    else:
        print("⚠️ No SKIRT training files found.")
    
    print("\n🎉 All 16-sample training diagnostics generated successfully!")

if __name__ == "__main__":
    main()