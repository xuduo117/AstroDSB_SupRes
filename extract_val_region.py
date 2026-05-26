import os
import numpy as np
import healpy as hp
from pixell import enmap

def extract_aligned_validation(act_fits_path, planck_fits_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read the global geometry
    print(f"1. Reading ACT Geometry from: {act_fits_path}")
    act_shape, act_wcs = enmap.read_map_geometry(act_fits_path)
    h, w = act_shape[-2:]

    # 2. Define the split threshold (Right 30%)
    split_x = int(w * 0.70)
    print(f"   -> Dataset split occurs at X = {split_x}. Validation is X > {split_x}.")

    # 3. Load the validation region
    print("2. Loading ACT validation region...")
    # Read the right 30% of the map. enmap preserves the WCS natively!
    act_val_region = enmap.read_map(act_fits_path, sel=(..., slice(None), slice(split_x, None)))
    
    # If the map has multiple channels (I, Q, U), grab Temperature (Intensity) only
    if act_val_region.ndim > 2:
        act_val_region = act_val_region[0]

    # 4. Find the active data bounds within this region to save memory
    print("3. Scanning for dense active pixels...")
    active_mask = np.isfinite(act_val_region) & (act_val_region != 0.0)
    valid_y, valid_x = np.where(active_mask)
    
    if len(valid_y) == 0:
        raise RuntimeError("No valid data found in the validation split!")

    y0, y1 = valid_y.min(), valid_y.max() + 1
    x0, x1 = valid_x.min(), valid_x.max() + 1

    # 5. Crop to the final, dense ACT patch
    # Slicing the enmap object automatically slices the underlying WCS too!
    act_val_patch = act_val_region[y0:y1, x0:x1]
    print(f"   -> Final Cropped Shape: {act_val_patch.shape}")

    # ==========================================
    # YOUR PROVEN HEALPIX INTERPOLATION LOGIC
    # ==========================================
    print("4. Loading Planck HEALPix Sphere...")
    planck_hp = hp.read_map(planck_fits_path, field=0)

    print("5. Extracting exact pixel coordinates and rotating C -> G...")
    # Get the RA/Dec in radians for our specific cropped patch
    coords = act_val_patch.posmap()
    dec_rad = coords[0]
    ra_rad = coords[1]
    
    original_shape = dec_rad.shape
    
    # Convert to Theta/Phi for Healpy
    theta_eq = (np.pi / 2.0 - dec_rad).flatten()
    phi_eq = ra_rad.flatten()
    
    # Rotate from Equatorial (Celestial) to Galactic
    rotator = hp.Rotator(coord=['C', 'G']) 
    theta_gal, phi_gal = rotator(theta_eq, phi_eq)
    
    print("6. Interpolating Planck data onto the ACT grid...")
    input_planck_K_flat = hp.get_interp_val(planck_hp, theta_gal, phi_gal)
    
    # Reshape back into the 2D image grid
    planck_val_patch_K = input_planck_K_flat.reshape(original_shape)
    
    print("   -> Converting Planck units from Kelvin to microKelvin...")
    planck_val_patch_uK = planck_val_patch_K * 1e6 
    
    # ==========================================
    # CLEANUP AND SAVE
    # ==========================================
    # Convert ACT map from enmap object to pure numpy array
    act_final = np.array(act_val_patch, dtype=np.float32)
    planck_final = np.array(planck_val_patch_uK, dtype=np.float32)

    # Clean up NaNs
    act_final[~np.isfinite(act_final)] = 0.0
    planck_final[~np.isfinite(planck_final)] = 0.0

    act_out_path = os.path.join(output_dir, "ACT_Val_GroundTruth.npy")
    planck_out_path = os.path.join(output_dir, "Planck_Val_Input.npy")
    
    np.save(act_out_path, act_final)
    np.save(planck_out_path, planck_final)
    
    print(f"7. ✅ Success! Perfectly aligned arrays saved to {output_dir}")

if __name__ == "__main__":
    # --- CONFIGURE THESE PATHS ---
    ACT_FITS = "/scratch/xuduo117/Real_Obs_Data/Planck_Data/act-planck_dr6.02_coadd_AA_night_f220_map.fits"
    PLANCK_FITS = "/scratch/xuduo117/Real_Obs_Data/Planck_Data/HFI_SkyMap_217_2048_R4.00_full.fits"
    OUTPUT_DIR = "./act_inference_data/Validation_Split/"
    
    extract_aligned_validation(ACT_FITS, PLANCK_FITS, OUTPUT_DIR)