import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from easydict import EasyDict as edict
from torch_ema import ExponentialMovingAverage

sys.path.append(os.getcwd())

from data.factory import get_dataloaders
from core.factory import get_model
from core.diffusion import AstroDiffusionBridge 
from utils.util import set_seed

# ==========================================
# PHYSICAL UNIT RECOVERY
# ==========================================
def unnormalize_prediction(pred_tensor, memory_tensor):
    """
    Rebuilds true physical units from the DSB model output using the 
    Asymmetric Dilution-Aware Arcsinh inversion scheme.
    
    Assumes memory_tensor contains:
      - memory_tensor[:, 0]: c_median
      - memory_tensor[:, 1]: alpha
      - memory_tensor[:, 2]: t_min
      - memory_tensor[:, 3]: t_max
    """
    # Reshape tracking constants for broadcast across [B, C, H, W]
    c_median = memory_tensor[:, 0].view(-1, 1, 1, 1)
    alpha    = memory_tensor[:, 1].view(-1, 1, 1, 1)
    t_min    = memory_tensor[:, 2].view(-1, 1, 1, 1)
    t_max    = memory_tensor[:, 3].view(-1, 1, 1, 1)
    
    # 1. Reverse the [0, 1] scaling back to arcsinh latent space
    tf_data = pred_tensor * (t_max - t_min) + t_min
    
    # 2. Invert the arcsinh transform to get centered data
    pred_centered = torch.sinh(tf_data) * alpha
    
    # 3. Add back the patch median to recover absolute physical units
    pred_physical = pred_centered + c_median
    return pred_physical


class DummyLogger:
    def info(self, msg): print(msg)
    def error(self, msg): print(f"ERR: {msg}")

def load_model_helper(model, ckpt_path, device, ema_decay=0.999):
    print(f"Loading weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    if 'ema_state' in ckpt:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        ema.load_state_dict(ckpt['ema_state'])
        ema.copy_to(model.parameters())
    return model.eval()

class DSBUncertaintyAnalysis:
    def __init__(self, cfg, checkpoint_path, device):
        self.cfg = cfg
        self.device = device
        log = DummyLogger()
        _, val_loader = get_dataloaders(cfg, log=log)
        self.dataset = val_loader.dataset
        self.model = get_model(cfg, val_loader, log).to(device)
        self.model = load_model_helper(self.model, checkpoint_path, device)
        self.bridge = AstroDiffusionBridge(cfg, device)

    def get_random_sample(self):
        idx = random.randint(0, len(self.dataset) - 1)
        x_prim_raw, x_cond_raw, y_raw, mask_raw, memory_raw, _ = self.dataset[idx]
        
        s_idx = 0 
        x_obs = x_cond_raw[s_idx:s_idx+1].to(self.device)
        y_gt = y_raw[s_idx:s_idx+1].to(self.device)
        memory = memory_raw[s_idx:s_idx+1].to(self.device)
        
        return x_obs, y_gt, memory, f"File_{idx}_Patch_{s_idx}"

    def compute_uncertainty(self, x_obs, memory, num_samples, skip_steps=50, mode='aleatoric'):
        if mode == 'epistemic': 
            self.model.train() 
        else: 
            self.model.eval()  
            
        predictions_phys = []
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc=f"Sampling {mode}", leave=False):
                pred_latent = self.bridge.ddpm_sampling(self.model, x_start=x_obs, cond_img=x_obs, verbose=False, skip_steps=skip_steps)
                pred_phys = unnormalize_prediction(pred_latent, memory)
                predictions_phys.append(pred_phys.cpu())
                
        stack = torch.cat(predictions_phys, dim=0)
        self.model.eval() 
        
        mean_map = torch.mean(stack, dim=0)
        std_map = torch.std(stack, dim=0)
        return mean_map, std_map

    def _save_and_plot_results(self, x_obs_phys, y_gt_phys, mean_pred, ale_std, epi_std, total_std, s_id):
        # 1. Extract 2D Numpy Arrays
        inp = x_obs_phys[0, 0].cpu().numpy()
        gt = y_gt_phys[0, 0].cpu().numpy()
        pred = mean_pred[0].numpy()
        ale = ale_std[0].numpy()
        epi = epi_std[0].numpy()
        tot = total_std[0].numpy()

        os.makedirs("uncertainty_results", exist_ok=True)

        # 2. SAVE ALL 3 UNCERTAINTIES TO DISK
        data_path = f"uncertainty_results/UncertaintyData_{s_id}.npz"
        np.savez_compressed(
            data_path,
            input_low_res=inp,
            ground_truth=gt,
            pred_mean=pred,
            aleatoric_std=ale,
            epistemic_std=epi,
            total_std=tot  # <-- Added Total
        )
        print(f"   -> Saved raw numpy arrays to {data_path}")

        # 3. Setup Bounds
        valid_gt = gt[gt > 0.0]
        vmin = np.nanpercentile(valid_gt, 1) if len(valid_gt) > 0 else 0.0
        vmax = np.nanpercentile(valid_gt, 99.5) if len(valid_gt) > 0 else 1.0
        alpha = np.nanstd(valid_gt) if len(valid_gt) > 0 else 0.1
        norm = colors.AsinhNorm(linear_width=alpha, vmin=vmin, vmax=vmax)

        # Lock the uncertainty colorbars to the Total Uncertainty max so they are comparable
        u_vmax = np.nanpercentile(tot, 99.5)

        # 4. Plot 6 Panels
        fig, axes = plt.subplots(1, 6, figsize=(36, 6))
        plt.suptitle(f"Super-Resolution Uncertainty Analysis: {s_id}", fontsize=20, y=1.05)
        
        im0 = axes[0].imshow(inp, norm=norm, cmap='magma', origin='lower')
        axes[0].set_title("Input (Low Res)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(gt, norm=norm, cmap='magma', origin='lower')
        axes[1].set_title("Ground Truth (High Res)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(pred, norm=norm, cmap='magma', origin='lower')
        axes[2].set_title("Predicted Mean")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(ale, cmap='viridis', origin='lower', vmin=0, vmax=u_vmax)
        axes[3].set_title("Aleatoric (Data Limits)")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        im4 = axes[4].imshow(epi, cmap='plasma', origin='lower', vmin=0, vmax=u_vmax)
        axes[4].set_title("Epistemic (Model Ignorance)")
        plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        
        im5 = axes[5].imshow(tot, cmap='inferno', origin='lower', vmin=0, vmax=u_vmax)
        axes[5].set_title("Total Predictive Uncertainty")
        plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plot_path = f'uncertainty_results/UncertaintyPlot_{s_id}.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"   -> Saved 6-panel plot to {plot_path}")

    def run_analysis(self, num_patches_to_test=5, mc_samples_per_patch=32, seed=42):
        print(f"\n[INIT] Locking random seed to {seed} for deterministic evaluation...")
        
        # 1. Lock all random number generators
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for i in range(num_patches_to_test):
            torch.cuda.empty_cache()
            
            # Because the seed is locked, this will pull the EXACT same patches every run
            x_latent, y_latent, memory, s_id = self.get_random_sample()
            print(f"\n--- Analyzing Patch {i+1}/{num_patches_to_test}: {s_id} ---")
            
            x_phys = unnormalize_prediction(x_latent, memory)
            y_phys = unnormalize_prediction(y_latent, memory)
            
            # ========================================================
            # COMMON RANDOM NUMBERS (CRN)
            # ========================================================
            # Because the global seed is locked, this master_seed will also 
            # be exactly the same every time you run the script!
            master_seed = random.randint(0, 2**32 - 1)
            
            # 1. Calculate Aleatoric (Pure diffusion stochasticity)
            torch.manual_seed(master_seed)
            mean_pred, ale_std = self.compute_uncertainty(x_latent, memory, num_samples=mc_samples_per_patch, mode='aleatoric')
            
            # 2. Calculate Total (Diffusion + MC Dropout)
            torch.manual_seed(master_seed) 
            _, total_std = self.compute_uncertainty(x_latent, memory, num_samples=mc_samples_per_patch, mode='epistemic')
            
            # 3. Clean Subtraction
            ale_var = ale_std ** 2
            total_var = total_std ** 2
            
            epi_var = torch.clamp(total_var - ale_var, min=0.0)
            epi_std = torch.sqrt(epi_var)
            
            self._save_and_plot_results(x_phys, y_phys, mean_pred, ale_std, epi_std, total_std, s_id)

if __name__ == "__main__":
    CHECKPOINT = 'results/astro_supres_dsb/latest.pt'
    CONFIG = 'configs/SuperRes_dsb.yaml'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(CONFIG, 'r') as f: 
        cfg = edict(yaml.safe_load(f))
        
    cfg.dataset.batch_size = 1
    
    # --- ANALYSIS SETTINGS ---
    # num_patches_to_test: How many different sky regions to evaluate
    # mc_samples_per_patch: How many times to run the diffusion model per patch. 
    #                       8 = Fast approximation. 32-64 = High-quality, smooth convergence.
    
    interp = DSBUncertaintyAnalysis(cfg, CHECKPOINT, device)
    interp.run_analysis(num_patches_to_test=5, mc_samples_per_patch=128, seed=42)