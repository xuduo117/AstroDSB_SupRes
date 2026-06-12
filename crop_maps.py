import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

class GalaxyMapCropper:
    def __init__(self, patch_size=256, stride=64, max_chunk_gb=1.0):
        self.patch_size = patch_size
        self.stride = stride 
        self.max_chunk_bytes = max_chunk_gb * 1024**3

    def process_list(self, file_list, output_dir):
        """Processes a list of .npy files and chunks them into subfolders."""
        os.makedirs(output_dir, exist_ok=True)
        
        for file_path in file_list:
            if not os.path.exists(file_path):
                print(f"[Warning] File not found: {file_path}")
                continue
                
            # Extract target name (e.g., 'ngc6052_sdss' from 'ngc6052_sdss.npy')
            target_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"\n--- Processing Target: {target_name} ---")
            
            try:
                self._process_single_map(file_path, target_name, output_dir)
            except Exception as e:
                print(f"[Error] Failed to process {target_name}: {e}")

    def _process_single_map(self, file_path, target_name, out_root):
        # 1. Load and Sanitize Data
        raw_map = np.load(file_path)
        if raw_map.ndim > 2:
            raw_map = raw_map.squeeze()
            
        raw_map[~np.isfinite(raw_map)] = np.nan
        raw_map = np.nan_to_num(raw_map, nan=0.0)

        h_raw, w_raw = raw_map.shape

        # 2. Dynamic Padding (Ensures perfect sliding window fit)
        pad_h = 0
        if h_raw < self.patch_size:
            pad_h += self.patch_size - h_raw
        if (h_raw + pad_h - self.patch_size) % self.stride != 0:
            pad_h += self.stride - ((h_raw + pad_h - self.patch_size) % self.stride)

        pad_w = 0
        if w_raw < self.patch_size:
            pad_w += self.patch_size - w_raw
        if (w_raw + pad_w - self.patch_size) % self.stride != 0:
            pad_w += self.stride - ((w_raw + pad_w - self.patch_size) % self.stride)

        padded_map = np.pad(raw_map, ((0, pad_h), (0, pad_w)), mode='reflect')
        padded_h, padded_w = padded_map.shape

        # 3. Setup Coordinates and Save Paths
        y_coords = list(range(0, padded_h - self.patch_size + 1, self.stride))
        x_coords = list(range(0, padded_w - self.patch_size + 1, self.stride))
        
        save_path = os.path.join(out_root, target_name)
        os.makedirs(save_path, exist_ok=True)

        current_batch = []
        patch_coords = [] 
        chunk_count = 0
        
        # 4. Sliding Window Extraction
        total_patches = len(y_coords) * len(x_coords)
        with tqdm(total=total_patches, desc="  Extracting Patches") as pbar:
            for y in y_coords:
                for x in x_coords:
                    patch = padded_map[y:y+self.patch_size, x:x+self.patch_size]
                    current_batch.append(patch)
                    patch_coords.append((y, x))
                    
                    # Dump to disk if we hit the memory limit
                    if (len(current_batch) * patch.nbytes) >= self.max_chunk_bytes:
                        self._save_chunk(current_batch, save_path, chunk_count)
                        current_batch = []
                        chunk_count += 1
                        
                    pbar.update(1)
        
        # Dump remaining patches
        if current_batch:
            self._save_chunk(current_batch, save_path, chunk_count)
            chunk_count += 1

        # 5. Save Metadata (Crucial for the stitching script later)
        meta = {
            'target_name': target_name,
            'original_shape': (h_raw, w_raw),
            'padded_shape': (padded_h, padded_w),
            'patch_coords': patch_coords,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'pad_h': pad_h,
            'pad_w': pad_w
        }
        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
            
        print(f"  -> Saved {len(patch_coords)} patches across {chunk_count} chunks to {save_path}")

    def _save_chunk(self, batch, path, count):
        out_fn = os.path.join(path, f"chunk_{count:03d}.npy")
        # Save as float32 to preserve dynamic range for Arcsinh normalization!
        np.save(out_fn, np.array(batch, dtype=np.float32))

# =========================================================
# EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    # Define the galaxies you want to process
    galaxy_files = [
        "/scratch/xuduo117/Real_Obs_Data/Planck_Data/ACT_Resolution_Maps/Planck_217GHz_0.5arcmin_CAR.npy"
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/CleanLR_12CO_full.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/VeryLR_12CO_full.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_polaris_planck_545GHz_Input_4x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_545GHz_Input_4x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Ursa_Major_planck_545GHz_Input_4x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_polaris_planck_857GHz_Input.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/Nobeyama_12CO_Input.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/ngc6052_sdss_input.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/ngc6052_synthetic_input.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/Nobeyama_Input.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/CARMA_NRO_downsampled_5.npy",
        # "/scratch/xuduo117/Project_DSB/AstroDSB_SupRes_2/act_inference_data/Validation_Split/Planck_Val_Input.npy",
        # "/scratch/xuduo117/Project_DSB/AstroDSB_SupRes_2/act_inference_data/Full_Footprint_Inference/Planck_FullSky_Input.npy",  
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_545GHz_Input_1x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_545GHz_Input_2x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_545GHz_Input_3x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_545GHz_Input_4x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_545GHz_Input_5x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_857GHz_Input_1x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_857GHz_Input_2x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_857GHz_Input_3x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_857GHz_Input_4x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Polaris_planck_857GHz_Input_5x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_545GHz_Input_1x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_545GHz_Input_2x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_545GHz_Input_3x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_545GHz_Input_4x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_545GHz_Input_5x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_857GHz_Input_1x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_857GHz_Input_2x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_857GHz_Input_3x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_857GHz_Input_4x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/aligned_Spider_planck_857GHz_Input_5x.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/HST_M83_Centered_Ready.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/HST_Antennae_Centered_Ready.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/HST_M83_6xSR_Input.npy",
        # "/scratch/xuduo117/Real_Obs_Data/SuperRes/sdss/HST_Antennae_6xSR_Input.npy"

    ]
    
    # Set max_chunk_gb to a safe size for your GPU inference later (e.g., 0.5 GB)
    cropper = GalaxyMapCropper(patch_size=256, stride=128, max_chunk_gb=1.5)
    # cropper = GalaxyMapCropper(patch_size=256, stride=64, max_chunk_gb=1.5)
    # cropper = GalaxyMapCropper(patch_size=256, stride=16, max_chunk_gb=1.5)
    
    # output_directory = "./act_inference_data/cropped_inference_data_128/"
    output_directory = "./cropped_inference_data/"
    cropper.process_list(galaxy_files, output_directory)