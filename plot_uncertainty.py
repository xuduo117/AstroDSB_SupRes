import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ==========================================
# 1. LOAD THE SAVED DATA
# ==========================================
# Point this to your downloaded .npz file
file_path = "uncertainty_results/UncertaintyData_File_29_Patch_0.npz"
data = np.load(file_path)

inp = data['input_low_res']
gt = data['ground_truth']
pred = data['pred_mean']
ale = data['aleatoric_std']
epi = data['epistemic_std']
tot = data['total_std']

# ==========================================
# 2. CALCULATE RELATIVE UNCERTAINTY
# ==========================================
# We add a small epsilon (1e-3) to the denominator to strictly prevent divide-by-zero infinities.
epsilon = 1e-3
fractional_uncertainty = tot / (np.clip(pred, epsilon, None))

# ==========================================
# 3. ADAPTIVE COLORBAR LOGIC
# ==========================================
def get_adaptive_bounds(arr, p_min=1.0, p_max=99.5):
    """Safely extracts percentile bounds, ignoring NaNs and zeros."""
    valid_data = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(valid_data) == 0:
        return 0.0, 1.0
    return np.nanpercentile(valid_data, p_min), np.nanpercentile(valid_data, p_max)

# A. Physical Maps Bounds (Shared so we can visually compare Input vs GT vs Pred)
phys_vmin, phys_vmax = get_adaptive_bounds(gt, p_min=1.0, p_max=99.5)
phys_alpha = np.nanstd(gt[gt > 0.0]) if len(gt[gt > 0.0]) > 0 else 0.1
norm_phys = colors.AsinhNorm(linear_width=phys_alpha, vmin=phys_vmin, vmax=phys_vmax)

# B. Uncertainty Maps Bounds (Independent to maximize contrast)
ale_vmin, ale_vmax = get_adaptive_bounds(ale, p_min=1.0, p_max=99.5)
epi_vmin, epi_vmax = get_adaptive_bounds(epi, p_min=1.0, p_max=99.5)
tot_vmin, tot_vmax = get_adaptive_bounds(tot, p_min=1.0, p_max=99.5)

# C. Ratio Map Bounds (Tighter upper bound, e.g., 98th percentile, to ignore extreme vacuum spikes)
rat_vmin, rat_vmax = get_adaptive_bounds(fractional_uncertainty, p_min=2.0, p_max=98.0)

# ==========================================
# 4. ADVANCED MOSAIC LAYOUT
# ==========================================
mosaic = """
AAAABBBBCCCC
DDDEEEFFFGGG
"""

fig, axes = plt.subplot_mosaic(mosaic, figsize=(24, 12))
plt.suptitle("Super-Resolution Uncertainty Calibration: File 29", fontsize=24, y=0.95)

# --- TOP ROW: PHYSICAL MAPS ---
im_A = axes['A'].imshow(inp, norm=norm_phys, cmap='magma', origin='lower')
axes['A'].set_title("(A) Input (Low Res Planck)", fontsize=16)
plt.colorbar(im_A, ax=axes['A'], fraction=0.046, pad=0.04)

im_B = axes['B'].imshow(gt, norm=norm_phys, cmap='magma', origin='lower')
axes['B'].set_title("(B) Ground Truth (High Res ACT)", fontsize=16)
plt.colorbar(im_B, ax=axes['B'], fraction=0.046, pad=0.04)

im_C = axes['C'].imshow(pred, norm=norm_phys, cmap='magma', origin='lower')
axes['C'].set_title("(C) DSB Predicted Mean", fontsize=16)
plt.colorbar(im_C, ax=axes['C'], fraction=0.046, pad=0.04)

# --- BOTTOM ROW: UNCERTAINTY MAPS ---
im_D = axes['D'].imshow(ale, cmap='viridis', origin='lower', vmin=ale_vmin, vmax=ale_vmax)
axes['D'].set_title("(D) Aleatoric (Data Uncertainty)", fontsize=16)
plt.colorbar(im_D, ax=axes['D'], fraction=0.046, pad=0.04)

im_E = axes['E'].imshow(epi, cmap='plasma', origin='lower', vmin=epi_vmin, vmax=epi_vmax)
axes['E'].set_title("(E) Epistemic (Model Uncertainty)", fontsize=16)
plt.colorbar(im_E, ax=axes['E'], fraction=0.046, pad=0.04)

im_F = axes['F'].imshow(tot, cmap='inferno', origin='lower', vmin=tot_vmin, vmax=tot_vmax)
axes['F'].set_title("(F) Total Predictive Uncertainty", fontsize=16)
plt.colorbar(im_F, ax=axes['F'], fraction=0.046, pad=0.04)

# --- BOTTOM ROW: RATIO MAP ---
im_G = axes['G'].imshow(fractional_uncertainty, cmap='Reds', origin='lower', vmin=rat_vmin, vmax=rat_vmax)
axes['G'].set_title("(G) Fractional Uncertainty ($\sigma_{tot}$ / Pred)", fontsize=16)
plt.colorbar(im_G, ax=axes['G'], fraction=0.046, pad=0.04)

# Clean up axes
for ax in axes.values():
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.90) 

# Save and show
plt.savefig("Custom_Uncertainty_Plot_File_29.png", dpi=300, bbox_inches='tight')
# plt.show()