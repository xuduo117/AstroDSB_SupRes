import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, rotate
import torchvision.transforms.functional as TF
import os
import healpy as hp
from pixell import enmap, utils as pix_utils

# =========================================================
# NEW: GLOBAL MEMORY CACHE
# Prevents loading massive files multiple times across Train/Val splits
# =========================================================
GLOBAL_MAP_CACHE = {
    'planck': None,
    'act': None
}


# =========================================================
# PHYSICAL CMB GRF GENERATOR
# =========================================================
def generate_cmb_grf(h, w, pixel_scale_arcmin):
    """Generates an analytical CMB Gaussian Random Field."""
    pixel_scale_rad = pixel_scale_arcmin * (np.pi / 180.0 / 60.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    
    ell = 2 * np.pi * f / pixel_scale_rad
    ell[0, 0] = 1.0  
    
    base_power = 1.0 / (ell**2 + 500.0)
    oscillation = np.cos(np.pi * (ell - 220.0) / 300.0)**2
    baryon_dip = 1.0 - 0.55 * np.exp(-0.5 * ((ell - 540.0) / 120.0)**2)
    damping = np.exp(-(ell / 1300.0)**1.5)
    
    c_ell = base_power * (1.0 + 2.5 * oscillation * baryon_dip) * damping
    
    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    cmb_fourier = white_noise * np.sqrt(c_ell)
    cmb_map = np.real(np.fft.ifft2(cmb_fourier))
    cmb_map = (cmb_map - np.mean(cmb_map)) / np.std(cmb_map)
    
    return cmb_map

# =========================================================
# 1. SYNTHETIC DATASET (Starforge + CMB)
# =========================================================
class PlanckACTDataset(Dataset):
    def __init__(self, cfg, file_list, mode='train'):
        self.cfg = cfg
        self.file_list = file_list
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)
        
        self.target_size = 256   
        self.planck_size = 51    
        self.min_crop = 512
        self.max_crop = 1536
        self.val_crop_scales = [512, 768, 1024, 1280, 1536]
        self.act_arcmin_per_pixel = 0.5
        self.total_fov_arcmin = self.target_size * self.act_arcmin_per_pixel
        self.base_noise_level = 0.02

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        raw_cube_hwc = np.load(file_path, mmap_mode='c')
        raw_cube = raw_cube_hwc.transpose(2, 0, 1) 
        num_views, h, w = raw_cube.shape
        max_possible_crop = min(h, w)
        actual_min_crop = min(self.min_crop, max_possible_crop)
        actual_max_crop = min(self.max_crop, max_possible_crop)
        
        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []

        for _ in range(self.num_patches):
            v_idx = random.randint(0, num_views - 1)
            
            if self.mode == 'train':
                crop_size = random.randint(actual_min_crop, actual_max_crop)
                y_off = random.randint(0, h - crop_size)
                x_off = random.randint(0, w - crop_size)
            else:
                valid_scales = [c for c in self.val_crop_scales if c <= max_possible_crop]
                crop_size = random.choice(valid_scales) if valid_scales else max_possible_crop
                y_off, x_off = (h - crop_size) // 2, (w - crop_size) // 2 
                
            cropped_view = raw_cube[v_idx, y_off:y_off+crop_size, x_off:x_off+crop_size].copy()
            cropped_view = np.nan_to_num(cropped_view, nan=0.0, posinf=0.0, neginf=0.0)

            # Synthetic Noise & CMB Background Injection
            if self.augment:
                h_crop, w_crop = cropped_view.shape
                effective_scale = max(cropped_view.max(), 1e-3)
                total_noise = np.random.normal(0, effective_scale * random.uniform(0.001, 0.015), (h_crop, w_crop))
                if random.random() < 0.8:
                    cmb_base = generate_cmb_grf(h_crop, w_crop, self.total_fov_arcmin / crop_size)
                    cmb_amp = effective_scale * (random.uniform(0.5, 2.0) if random.random() < 0.15 else random.uniform(0.01, 0.06))
                    total_noise += cmb_base * cmb_amp
                total_noise -= np.mean(total_noise)
                cropped_view += total_noise

            # DEGRADATION PIPELINE
            act_sigma = (3.0 * (crop_size / self.target_size)) / 2.355
            planck_sigma = (15.0 * (crop_size / self.target_size)) / 2.355
            act_blurred = gaussian_filter(cropped_view, sigma=act_sigma)
            planck_blurred = gaussian_filter(cropped_view, sigma=planck_sigma)
            
            t_act = torch.from_numpy(act_blurred).float().unsqueeze(0).unsqueeze(0)
            t_planck = torch.from_numpy(planck_blurred).float().unsqueeze(0).unsqueeze(0)

            act_linear = F.interpolate(t_act, size=(self.target_size, self.target_size), mode='area')
            planck_linear = F.interpolate(t_planck, size=(self.planck_size, self.planck_size), mode='area')
            planck_upsampled = F.interpolate(planck_linear, size=(self.target_size, self.target_size), mode='bicubic', align_corners=False)

            gt_raw = act_linear.squeeze().numpy()
            input_raw = planck_upsampled.squeeze().numpy()

            # UNIVERSAL ARCSINH HEADROOM NORMALIZER
            i_median = np.nanmedian(input_raw)
            i_std = np.nanstd(input_raw) + 1e-8
            i_min_bound = np.nanpercentile(input_raw, 0.1)
            i_max_bound = np.nanpercentile(input_raw, 99.9)
            
            in_tf = np.arcsinh((input_raw - i_median) / i_std)
            gt_tf = np.arcsinh((gt_raw - i_median) / i_std)
            
            tf_min = np.arcsinh((i_min_bound - i_median) / i_std)
            tf_max = np.arcsinh((i_max_bound - i_median) / i_std)
            N_scale = (tf_max - tf_min) + 1e-8
            
            input_norm = np.clip(0.1 + 0.8 * (in_tf - tf_min) / N_scale, 0.0, 1.0)
            gt_norm = np.clip(0.1 + 0.8 * (gt_tf - tf_min) / N_scale, 0.0, 1.0)
            
            patch_cond = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0)
            patch_y = torch.tensor(gt_norm, dtype=torch.float32).unsqueeze(0)
            patch_mask = torch.ones_like(patch_y)
            memory_tensor = torch.tensor([i_median, i_std, tf_min, N_scale], dtype=torch.float32)

            # AUGMENTATIONS
            combined = torch.stack([patch_y, patch_cond, patch_mask], dim=0) 
            if self.augment:
                if random.random() < 0.5: combined = TF.rotate(combined, random.uniform(0, 360), interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
                if random.random() < 0.5: combined = TF.hflip(combined)
                if random.random() < 0.5: combined = TF.vflip(combined)

            patch_y = combined[0]
            patch_cond = combined[1]
            patch_mask = combined[2]
            patch_prim = patch_cond.clone()
            
            x_prim_list.append(patch_prim)
            x_cond_list.append(patch_cond)
            y_list.append(patch_y)
            mask_list.append(patch_mask)
            memory_list.append(memory_tensor)

        return (torch.stack(x_prim_list, dim=0), torch.stack(x_cond_list, dim=0), 
                torch.stack(y_list, dim=0), torch.stack(mask_list, dim=0), 
                torch.stack(memory_list, dim=0), index)

# =========================================================
# 2. REAL PLANCK DATASET (Cached)
# =========================================================
class RealPlanckDegradationDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)
        
        self.target_size = 256
        self.reso_arcmin = 1.71 
        
        # --- GLOBAL CACHE IMPLEMENTATION ---
        if GLOBAL_MAP_CACHE['planck'] is None:
            print(f"[{self.mode.upper()}] Loading Global Planck Map from disk to RAM...")
            real_map_path = cfg.dataset.get('real_planck_path', '/scratch/xuduo117/Real_Obs_Data/Planck_Data/PSM_217GHz_sim.fits')
            GLOBAL_MAP_CACHE['planck'] = hp.read_map(real_map_path, field=0)
            
        self.real_map = GLOBAL_MAP_CACHE['planck']
        self.base_noise_level = 0.02
        
        if mode == 'train': self.lon_min, self.lon_max = 0.0, 270.0
        else: self.lon_min, self.lon_max = 270.0, 360.0

    def __len__(self):
        return 1000  

    def _extract_flat_patch(self, lon_center, lat_center):
        extent = self.target_size / 2.0
        x = np.linspace(-extent, extent - 1, self.target_size) * self.reso_arcmin / 60.0
        y = np.linspace(-extent, extent - 1, self.target_size) * self.reso_arcmin / 60.0
        X, Y = np.meshgrid(x, y)
        lon_grid = lon_center + (X / np.cos(np.radians(lat_center)))
        lat_grid = lat_center + Y
        theta = np.radians(90.0 - lat_grid)
        phi = np.radians(lon_grid % 360.0)
        return hp.get_interp_val(self.real_map, theta, phi)

    def __getitem__(self, index):
        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []
        
        for _ in range(self.num_patches):
            lon = random.uniform(self.lon_min, self.lon_max)
            lat = random.uniform(-15.0, 15.0) 
            patch_raw = self._extract_flat_patch(lon, lat)
            patch_raw = np.where(patch_raw < -1e20, np.nanmean(patch_raw[patch_raw > -1e20]), patch_raw)
            
            # DEGRADATION PIPELINE
            patch_tensor_raw = torch.tensor(patch_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            if self.augment:
                blur_sigma = random.uniform(1.5, 3.5)
                down_factor = random.uniform(3.0, 5.0)
            else:
                blur_sigma = 2.5
                down_factor = 5.0
                
            k_size = int(blur_sigma * 4) | 1 
            blurred = TF.gaussian_blur(patch_tensor_raw, kernel_size=[k_size, k_size], sigma=[blur_sigma, blur_sigma])
            low_res_size = int(self.target_size / down_factor)
            downsampled = F.interpolate(blurred, size=(low_res_size, low_res_size), mode='area')
            noise = torch.randn_like(downsampled) * self.base_noise_level
            upsampled = F.interpolate(downsampled + noise, size=(self.target_size, self.target_size), mode='bicubic', align_corners=False)
            
            gt_raw = patch_raw
            input_raw = upsampled.squeeze().numpy()

            # UNIVERSAL ARCSINH HEADROOM NORMALIZER
            i_median = np.nanmedian(input_raw)
            i_std = np.nanstd(input_raw) + 1e-8
            i_min_bound = np.nanpercentile(input_raw, 0.1)
            i_max_bound = np.nanpercentile(input_raw, 99.9)
            
            in_tf = np.arcsinh((input_raw - i_median) / i_std)
            gt_tf = np.arcsinh((gt_raw - i_median) / i_std)
            
            tf_min = np.arcsinh((i_min_bound - i_median) / i_std)
            tf_max = np.arcsinh((i_max_bound - i_median) / i_std)
            N_scale = (tf_max - tf_min) + 1e-8
            
            input_norm = np.clip(0.1 + 0.8 * (in_tf - tf_min) / N_scale, 0.0, 1.0)
            gt_norm = np.clip(0.1 + 0.8 * (gt_tf - tf_min) / N_scale, 0.0, 1.0)
            
            patch_cond = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0)
            patch_y = torch.tensor(gt_norm, dtype=torch.float32).unsqueeze(0)
            patch_mask = torch.ones_like(patch_y)
            memory_tensor = torch.tensor([i_median, i_std, tf_min, N_scale], dtype=torch.float32)

            # AUGMENTATIONS
            combined = torch.stack([patch_y, patch_cond, patch_mask], dim=0) 
            if self.augment:
                if random.random() < 0.5: combined = TF.hflip(combined)
                if random.random() < 0.5: combined = TF.vflip(combined)
                if random.random() < 0.5: combined = TF.rotate(combined, random.uniform(0, 360), interpolation=TF.InterpolationMode.NEAREST, fill=0.0)

            patch_y = combined[0]      
            patch_cond = combined[1]
            patch_mask = combined[2]
            patch_prim = patch_cond.clone()
            
            x_prim_list.append(patch_prim)
            x_cond_list.append(patch_cond)
            y_list.append(patch_y)
            mask_list.append(patch_mask)
            memory_list.append(memory_tensor)

        return (torch.stack(x_prim_list, dim=0), torch.stack(x_cond_list, dim=0), 
                torch.stack(y_list, dim=0), torch.stack(mask_list, dim=0), 
                torch.stack(memory_list, dim=0), index)

# =========================================================
# 3. REAL ACT DATASET (Cached + Fast Boolean Pooling)
# =========================================================
class RealACTDegradationDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)
        self.target_size = 256
        self.base_noise_level = 0.02
        
        act_map_path = cfg.dataset.get('real_act_path', '/scratch/xuduo117/Real_Obs_Data/Planck_Data/act-planck_dr6.02_coadd_AA_night_f220_map.fits')
        
        # =========================================================
        # THE HPC SILVER BULLET: ZERO-RAM MEMORY MAPPING
        # =========================================================
        cache_dir = os.path.dirname(act_map_path)
        npy_map_path = os.path.join(cache_dir, "act_dr6_fast_map.npy")
        pix_box_path = os.path.join(cache_dir, "act_test_pix_box.npy")
        coords_path = os.path.join(cache_dir, f"act_valid_coords_{mode}.npy")
        
        # 1. ONE-TIME FITS CONVERSION (Only runs once, ever!)
        if not os.path.exists(npy_map_path) or not os.path.exists(pix_box_path):
            if getattr(cfg, 'global_rank', 0) == 0:
                print("⏳ [ONE-TIME SETUP] Converting 6GB ACT FITS to Memory-Mapped NPY...")
                raw_enmap = enmap.read_map(act_map_path)[0]
                
                # Convert the massive array and save it
                act_np = np.asarray(raw_enmap, dtype=np.float32)
                act_np[act_np == 0.0] = np.nan
                np.save(npy_map_path, act_np)
                
                # Calculate the WCS quarantine coordinates so we never have to load headers again
                test_ra, test_dec, hw = 265.0, -28.0, 2.5
                dec_min, dec_max = (test_dec - hw) * pix_utils.degree, (test_dec + hw) * pix_utils.degree
                ra_min, ra_max = (test_ra - hw) * pix_utils.degree, (test_ra + hw) * pix_utils.degree
                pix_box = raw_enmap.sky2pix(np.array([[dec_min, ra_max], [dec_max, ra_min]]))
                np.save(pix_box_path, pix_box)
                print("✅ FITS Conversion complete! Future runs will bypass this instantly.")
            else:
                # Other GPUs sleep and wait for GPU 0 to finish writing the file
                import time
                while not os.path.exists(npy_map_path):
                    time.sleep(2)
        
        # 2. INSTANT ZERO-RAM LOAD
        # This prevents the PyTorch Dataloader from exploding your RAM!
        self.act_map = np.load(npy_map_path, mmap_mode='c')
        self.h, self.w = self.act_map.shape
        pix_box = np.load(pix_box_path)
        
        # 3. O(1) COORDINATE POOLING CACHE
        # Instead of taking 5 seconds to calculate the crop pool, we save it and load it in 0.001s
        if os.path.exists(coords_path):
            self.valid_coords = np.load(coords_path).tolist()
        else:
            if getattr(cfg, 'global_rank', 0) == 0:
                print(f"[{self.mode.upper()}] Building pool of valid ACT coordinates...")
                valid_mask = ~np.isnan(self.act_map)
                
                y0, y1 = max(0, int(min(pix_box[0]))), min(self.h, int(max(pix_box[0])))
                x0, x1 = max(0, int(min(pix_box[1]))), min(self.w, int(max(pix_box[1])))
                valid_mask[y0:y1, x0:x1] = False
                
                split_x = int(self.w * 0.70)
                if mode == 'train': valid_mask[:, split_x:] = False
                elif mode == 'val': valid_mask[:, :split_x] = False
                
                valid_y, valid_x = np.where(valid_mask)
                if len(valid_y) == 0: raise RuntimeError(f"No valid pixels found in {self.mode} split!")
                
                min_y, max_y = valid_y.min(), valid_y.max()
                min_x, max_x = valid_x.min(), valid_x.max()
                
                self.valid_coords = []
                stride = 128 
                max_invalid_allowed = 0.20 * (self.target_size ** 2)
                
                for y in range(min_y, max_y - self.target_size + 1, stride):
                    for x in range(min_x, max_x - self.target_size + 1, stride):
                        patch_invalid = (~valid_mask[y : y + self.target_size, x : x + self.target_size]).sum()
                        if patch_invalid <= max_invalid_allowed:
                            self.valid_coords.append((y, x))
                            
                np.save(coords_path, np.array(self.valid_coords))
            else:
                import time
                while not os.path.exists(coords_path):
                    time.sleep(1)
                self.valid_coords = np.load(coords_path).tolist()

    def __len__(self):
        return 1000  

    def __getitem__(self, index):
        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []
        
        for _ in range(self.num_patches):
            # Safe Jitter & Cropping
            y_off, x_off = random.choice(self.valid_coords)
            max_safe_y = max(0, min(127, self.h - self.target_size - y_off))
            max_safe_x = max(0, min(127, self.w - self.target_size - x_off))
            
            y_start = y_off + random.randint(0, max_safe_y)
            x_start = x_off + random.randint(0, max_safe_x)
            
            patch_raw = np.array(self.act_map[y_start : y_start + self.target_size, 
                                              x_start : x_start + self.target_size])
            
            # Masking & Padding NaNs
            nan_mask = np.isnan(patch_raw)
            patch_mask_np = (~nan_mask).astype(np.float32)
            
            with np.errstate(invalid='ignore'):
                valid_mean = np.nanmean(patch_raw)
                if np.isnan(valid_mean): valid_mean = 0.0
            patch_raw = np.where(nan_mask, valid_mean, patch_raw)
            
            # DEGRADATION PIPELINE
            patch_tensor_raw = torch.tensor(patch_raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            if self.augment:
                blur_sigma = random.uniform(2.0, 5.0) 
                down_factor = random.uniform(3.0, 5.0)
            else:
                blur_sigma = 3.5 
                down_factor = 5.0 
                
            k_size = int(blur_sigma * 4) | 1 
            blurred = TF.gaussian_blur(patch_tensor_raw, kernel_size=[k_size, k_size], sigma=[blur_sigma, blur_sigma])
            low_res_size = int(self.target_size / down_factor)
            downsampled = F.interpolate(blurred, size=(low_res_size, low_res_size), mode='area')
            noise = torch.randn_like(downsampled) * self.base_noise_level
            upsampled = F.interpolate(downsampled + noise, size=(self.target_size, self.target_size), mode='bicubic', align_corners=False)
            
            gt_raw = patch_raw
            input_raw = upsampled.squeeze().numpy()

            # UNIVERSAL ARCSINH HEADROOM NORMALIZER
            i_median = np.nanmedian(input_raw)
            i_std = np.nanstd(input_raw) + 1e-8
            i_min_bound = np.nanpercentile(input_raw, 0.1)
            i_max_bound = np.nanpercentile(input_raw, 99.9)
            
            in_tf = np.arcsinh((input_raw - i_median) / i_std)
            gt_tf = np.arcsinh((gt_raw - i_median) / i_std)
            
            tf_min = np.arcsinh((i_min_bound - i_median) / i_std)
            tf_max = np.arcsinh((i_max_bound - i_median) / i_std)
            N_scale = (tf_max - tf_min) + 1e-8
            
            input_norm = np.clip(0.1 + 0.8 * (in_tf - tf_min) / N_scale, 0.0, 1.0)
            gt_norm = np.clip(0.1 + 0.8 * (gt_tf - tf_min) / N_scale, 0.0, 1.0)
            
            patch_cond = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0)
            patch_y = torch.tensor(gt_norm, dtype=torch.float32).unsqueeze(0)
            patch_mask = torch.tensor(patch_mask_np, dtype=torch.float32).unsqueeze(0)
            memory_tensor = torch.tensor([i_median, i_std, tf_min, N_scale], dtype=torch.float32)

            # AUGMENTATIONS
            combined = torch.stack([patch_y, patch_cond, patch_mask], dim=0) 
            if self.augment:
                if random.random() < 0.5: combined = TF.hflip(combined)
                if random.random() < 0.5: combined = TF.vflip(combined)
                if random.random() < 0.5: combined = TF.rotate(combined, random.uniform(0, 360), interpolation=TF.InterpolationMode.NEAREST, fill=0.0)

            patch_y = combined[0]      
            patch_cond = combined[1]    
            patch_mask_final = combined[2]
            patch_prim = patch_cond.clone()
            
            x_prim_list.append(patch_prim)
            x_cond_list.append(patch_cond)
            y_list.append(patch_y)
            mask_list.append(patch_mask_final)
            memory_list.append(memory_tensor)

        return (torch.stack(x_prim_list, dim=0), torch.stack(x_cond_list, dim=0), 
                torch.stack(y_list, dim=0), torch.stack(mask_list, dim=0), 
                torch.stack(memory_list, dim=0), index)
        
# =========================================================
# 4. WRAPPER DATASET (Tri-Modal Loader)
# =========================================================
class CombinedTransferDataset(Dataset):
    def __init__(self, cfg, file_list, mode='train', real_planck_fraction=0.2, real_act_fraction=0.2):
        self.synthetic_dataset = PlanckACTDataset(cfg, file_list, mode=mode)
        self.planck_dataset = RealPlanckDegradationDataset(cfg, mode=mode)
        self.act_dataset = RealACTDegradationDataset(cfg, mode=mode)
        self.p_frac = real_planck_fraction
        self.a_frac = real_act_fraction

    def __len__(self):
        return len(self.synthetic_dataset)

    def __getitem__(self, index):
        p = random.random()
        if p < self.a_frac:
            return self.act_dataset[0]
        elif p < (self.a_frac + self.p_frac):
            return self.planck_dataset[0]
        else:
            return self.synthetic_dataset[index]