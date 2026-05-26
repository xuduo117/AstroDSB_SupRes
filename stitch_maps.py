import os
import glob
import pickle
import numpy as np
import re
from tqdm import tqdm

def natural_sort_key(s):
    """Ensures chunk_10 comes after chunk_2 (Natural Integer Sorting)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class GalaxyMapReconstructor:
    def __init__(self, crop_meta_root, output_final_dir):
        # In our pipeline, the predictions are saved right next to the chunks
        self.crop_meta_root = crop_meta_root 
        self.output_final_dir = output_final_dir
        os.makedirs(output_final_dir, exist_ok=True)

    def check_completion(self, target_folder):
        meta_path = os.path.join(self.crop_meta_root, target_folder, "metadata.pkl")
        if not os.path.exists(meta_path):
            return None, False

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        expected_patches = len(meta['patch_coords'])
        pred_path = os.path.join(self.crop_meta_root, target_folder)
        pred_chunks = sorted(glob.glob(os.path.join(pred_path, "pred_chunk_*.npy")), key=natural_sort_key)
        
        total_predicted = 0
        for pc in pred_chunks:
            total_predicted += np.load(pc, mmap_mode='r').shape[0]
            
        return meta, (total_predicted == expected_patches)

    def _create_hann_window(self, window_size):
        """Creates a 2D bell curve to seamlessly blend overlapping patches."""
        window_1d = np.hanning(window_size)
        window_2d = window_1d[:, None] * window_1d[None, :]
        return window_2d.astype(np.float32)

    def process_target(self, target_folder):
        meta, is_complete = self.check_completion(target_folder)
        if meta is None or not is_complete:
            status = "Incomplete predictions" if meta else "Metadata Missing"
            print(f"  [Skip] {target_folder}: {status}")
            return

        print(f"\n--- [Smart Stitching] {target_folder} ---")
        
        patch_size = meta['patch_size']
        H, W = meta['original_shape']
        padded_h, padded_w = meta['padded_shape']
        
        # Accumulators (Canvas)
        output_canvas = np.zeros((padded_h, padded_w), dtype=np.float32)
        weight_canvas = np.zeros((padded_h, padded_w), dtype=np.float32)

        # Apply Windowing
        hann_window = self._create_hann_window(patch_size)

        pred_path = os.path.join(self.crop_meta_root, target_folder)
        chunk_files = sorted(glob.glob(os.path.join(pred_path, "pred_chunk_*.npy")), key=natural_sort_key)
        
        patch_idx = 0
        coords = meta['patch_coords']
        
        for cf in tqdm(chunk_files, desc="  Window-Blended Stitching", leave=False):
            # Use mmap_mode to prevent memory overflow on massive maps
            chunk_data = np.load(cf, mmap_mode='r').astype(np.float32)
            
            for patch in chunk_data:
                y, x = coords[patch_idx]
                
                # Blend the physical prediction into the canvas
                output_canvas[y:y+patch_size, x:x+patch_size] += patch * hann_window
                weight_canvas[y:y+patch_size, x:x+patch_size] += hann_window
                
                patch_idx += 1

        # Normalize by weights to get a seamless weighted average
        # Add 1e-8 to prevent divide-by-zero warnings on un-patched edge pixels
        final_stitched = output_canvas / (weight_canvas + 1e-8)

        # Crop off the reflection padding to return EXACTLY the original input size
        final_act_map = final_stitched[:H, :W]

        # Save the final NumPy array
        out_fn = os.path.join(self.output_final_dir, f"Recon_{target_folder}.npy")
        np.save(out_fn, final_act_map)
        print(f"  [Success] Saved Final High-Res Map: {out_fn}")

if __name__ == "__main__":
    # Point this to the directory containing all your target subfolders
    # data_directory = './act_inference_data/cropped_inference_data_128/'
    data_directory = './cropped_inference_data/'
    output_directory = './final_super_res_maps/'
    
    recon = GalaxyMapReconstructor(
        crop_meta_root=data_directory,
        output_final_dir=output_directory
    )
    
    # Find all subfolders (targets) and process them
    targets = sorted([d for d in os.listdir(recon.crop_meta_root) if os.path.isdir(os.path.join(recon.crop_meta_root, d))])
    
    for target in targets:
        recon.process_target(target)