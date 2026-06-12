import torch
import numpy as np
import random
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as TF
from PIL import Image
import os
import healpy as hp
from pixell import enmap, utils as pix_utils

# =========================================================
# GLOBAL MEMORY CACHE
# =========================================================
GLOBAL_MAP_CACHE = {
    'planck': None,
    'act': None
}

# =========================================================
# ASYMMETRIC DILUTION-AWARE ARCSINH (SMART BOUNDS)
# =========================================================
def apply_patch_asymmetric_arcsinh(patch, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0):
    """
    Anchors the transform at physical 0.0.
    Uses Smart Bounds (10-sigma fallback) to protect CMB Gaussian tails and point sources 
    without letting single cosmic rays destroy the dynamic range.
    """
    c = 0.0  # Absolute physical anchor (Zero-mean background)
    
    s = np.nanstd(patch) + 1e-8
    alpha = s * transition_sigmas
    tf_patch = np.arcsinh(patch / alpha)
    
    # 1. Get raw statistics
    abs_min, abs_max = np.nanmin(patch), np.nanmax(patch)
    perc_min, perc_max = np.nanpercentile(patch, 0.1), np.nanpercentile(patch, 99.9)
    sigma_min, sigma_max = c - max_sigma * s, c + max_sigma * s
    
    # 2. SMART BOUNDS:
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
    
    memory_tensor = torch.tensor([c, alpha, t_min, t_max], dtype=torch.float32)
    return final_norm, memory_tensor

# =========================================================
# PHYSICAL CMB GRF GENERATOR
# =========================================================
def generate_cmb_grf(h, w, pixel_scale_arcmin):
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

def apply_advanced_telescope_noise(low_res_map, augment=True):
    """
    Simulates physical detector artifacts dynamically scaled to the patch SNR.
    UPDATED: Much heavier noise base to force robust learning on low-res data.
    """
    local_std = np.nanstd(low_res_map)
    if np.isnan(local_std) or local_std == 0:
        local_std = 1e-5 
        
    if not augment:
        val_noise = np.random.normal(0, local_std * 0.05, low_res_map.shape)
        return (low_res_map + val_noise).astype(np.float32)
        
    h, w = low_res_map.shape
    total_noise = np.zeros_like(low_res_map)
    
    # ---> CRANKED UP: 10% to 30% of signal standard deviation <---
    dynamic_noise_base = local_std * random.uniform(0.10, 0.30)
    
    # 1. BASE WHITE NOISE 
    white_scale = dynamic_noise_base * random.uniform(0.4, 0.8)
    total_noise += np.random.normal(0, white_scale, (h, w))
    
    # 2. 1/f PINK NOISE 
    if random.random() < 0.50:
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        f = np.sqrt(fx**2 + fy**2)
        f[0, 0] = 1.0  
        
        amp = 1.0 / np.sqrt(f)
        amp[0, 0] = 0.0
        
        complex_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
        pink_noise = np.real(np.fft.ifft2(complex_noise * amp))
        
        pink_noise = (pink_noise - np.mean(pink_noise)) / (np.std(pink_noise) + 1e-8)
        pink_scale = dynamic_noise_base * random.uniform(0.5, 1.5) 
        total_noise += pink_noise * pink_scale

    # 3. SIGNAL-CORRELATED NOISE (Poisson)
    if random.random() < 0.50:
        shifted_signal = low_res_map - np.min(low_res_map) + 1e-6
        noise_envelope = np.sqrt(shifted_signal)
        noise_envelope /= (np.mean(noise_envelope) + 1e-8)
        
        shot_scale = dynamic_noise_base * random.uniform(0.3, 1.0)
        shot_noise = np.random.normal(0, 1, (h, w)) * noise_envelope * shot_scale
        total_noise += shot_noise

    return (low_res_map + total_noise).astype(np.float32)

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

            # --- 1. PHYSICAL SKY & NOISE INJECTION ---
            if self.augment:
                h_crop, w_crop = cropped_view.shape
                effective_scale = max(np.nanpercentile(cropped_view,99.5), 1e-3)
                total_noise = np.random.normal(0, effective_scale * random.uniform(0.001, 0.015), (h_crop, w_crop))
                
                if random.random() < 0.8:
                    cmb_base = generate_cmb_grf(h_crop, w_crop, self.total_fov_arcmin / crop_size)
                    cmb_amp = effective_scale * (random.uniform(0.5, 2.0) if random.random() < 0.15 else random.uniform(0.01, 0.06))
                    total_noise += cmb_base * cmb_amp
                    
                total_noise -= np.mean(total_noise)
                cropped_view += total_noise

            # --- 2. FAST CPU TELESCOPE OPTICS (SciPy) ---
            act_fwhm = 3.0
            if self.augment:
                planck_fwhm = random.uniform(12.0, 18.0) 
            else:
                planck_fwhm = 15.0
                
            act_sigma = (act_fwhm * (crop_size / self.target_size)) / 2.355
            planck_sigma = (planck_fwhm * (crop_size / self.target_size)) / 2.355
            
            act_blurred = gaussian_filter(cropped_view, sigma=act_sigma, truncate=4.0)
            planck_blurred = gaussian_filter(cropped_view, sigma=planck_sigma, truncate=4.0)

            # --- 3. FAST CPU DETECTOR GRIDS (PIL C-Optimized) ---
            act_linear = np.array(Image.fromarray(act_blurred, mode='F').resize((self.target_size, self.target_size), resample=Image.BOX))
            planck_linear = np.array(Image.fromarray(planck_blurred, mode='F').resize((self.planck_size, self.planck_size), resample=Image.BOX))
            
            # Heavy noise added here to the low resolution data
            planck_noisy = apply_advanced_telescope_noise(planck_linear, augment=self.augment)

            planck_upsampled = np.array(Image.fromarray(planck_noisy, mode='F').resize((self.target_size, self.target_size), resample=Image.BICUBIC))

            # ---> POST-INTERPOLATION HIGH-FREQUENCY NOISE <---
            if self.augment:
                post_noise_scale = (np.nanstd(planck_upsampled) + 1e-8) * random.uniform(0.03, 0.1)
                high_freq_noise = np.random.normal(0, post_noise_scale, planck_upsampled.shape).astype(np.float32)
                planck_upsampled += high_freq_noise
                
            gt_raw_np = act_linear
            input_raw_np = planck_upsampled

            # --- 4. EXTRACT LOCAL MEDIAN FOR PADDING ---
            c_fill = float(np.nanmedian(input_raw_np))
            if np.isnan(c_fill): c_fill = 0.0

            # --- 5. PHYSICAL AUGMENTATIONS (Decoupled Math) ---
            t_in = torch.from_numpy(input_raw_np).float().unsqueeze(0)
            t_gt = torch.from_numpy(gt_raw_np).float().unsqueeze(0)
            t_mask = torch.ones_like(t_gt)

            if self.augment:
                if random.random() < 0.5:
                    t_in = TF.hflip(t_in)
                    t_gt = TF.hflip(t_gt)
                    t_mask = TF.hflip(t_mask)
                if random.random() < 0.5:
                    t_in = TF.vflip(t_in)
                    t_gt = TF.vflip(t_gt)
                    t_mask = TF.vflip(t_mask)
                if random.random() < 0.5: 
                    angle = random.uniform(0.0, 360.0)
                    t_in = TF.rotate(t_in, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=c_fill)
                    t_gt = TF.rotate(t_gt, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=c_fill)
                    t_mask = TF.rotate(t_mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0.0)

            in_aug = t_in[0].numpy()
            gt_aug = t_gt[0].numpy()
            patch_mask = t_mask[0]

            # --- 6. SMART BOUNDS ARCSINH NORMALIZATION ---
            _, memory_tensor = apply_patch_asymmetric_arcsinh(
                input_raw_np, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0
            )
            
            alpha = memory_tensor[1].item()
            t_min = memory_tensor[2].item()
            t_max = memory_tensor[3].item()
            
            in_norm_np = np.clip((np.arcsinh(in_aug / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)
            gt_final_norm = np.clip((np.arcsinh(gt_aug / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)

            patch_cond = torch.tensor(in_norm_np, dtype=torch.float32).unsqueeze(0)
            patch_y = torch.tensor(gt_final_norm, dtype=torch.float32).unsqueeze(0)
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
        
        if GLOBAL_MAP_CACHE['planck'] is None:
            print(f"[{self.mode.upper()}] Loading Global Planck Map from disk to RAM...")
            real_map_path = cfg.dataset.get('real_planck_path', '/scratch/xuduo117/Real_Obs_Data/Planck_Data/PSM_217GHz_sim.fits')
            GLOBAL_MAP_CACHE['planck'] = hp.read_map(real_map_path, field=0)
            
        self.real_map = GLOBAL_MAP_CACHE['planck']
        
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
            valid_patch_found = False
            for attempt in range(50):
                lon = random.uniform(self.lon_min, self.lon_max)
                lat = random.uniform(-15.0, 15.0) 
                
                patch_raw = self._extract_flat_patch(lon, lat).astype(np.float32)
                
                patch_raw[~np.isfinite(patch_raw)] = np.nan
                patch_raw[np.abs(patch_raw) > 1e10] = np.nan
                
                valid_mask = ~np.isnan(patch_raw)
                
                if valid_mask.mean() > 0.80:
                    fill_val = np.nanmedian(patch_raw)
                    if np.isnan(fill_val): fill_val = 0.0
                    patch_raw = np.where(valid_mask, patch_raw, fill_val)
                    valid_patch_found = True
                    break
            
            if not valid_patch_found:
                patch_raw = np.zeros((self.target_size, self.target_size), dtype=np.float32)

            patch_raw = gaussian_filter(patch_raw, sigma=0.8, truncate=4.0)
            
            # --- 2. FAST CPU DEGRADATION ---
            if self.augment:
                blur_sigma = random.uniform(1.5, 3.5)
                down_factor = random.uniform(3.0, 5.0)
            else:
                blur_sigma = 2.5
                down_factor = 5.0
                
            blurred = gaussian_filter(patch_raw, sigma=blur_sigma, truncate=4.0)
            low_res_size = int(self.target_size / down_factor)
            downsampled = np.array(Image.fromarray(blurred, mode='F').resize((low_res_size, low_res_size), resample=Image.BOX))
            
            # ---> NEW: DYNAMIC HEAVY NOISE ON LOW-RES REAL DATA <---
            if self.augment:
                noise_std = (np.nanstd(downsampled) + 1e-8) * random.uniform(0.10, 0.30)
                noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.float32)
            else:
                noise_std = (np.nanstd(downsampled) + 1e-8) * 0.05
                noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.float32)
            
            upsampled = np.array(Image.fromarray(downsampled + noise, mode='F').resize((self.target_size, self.target_size), resample=Image.BICUBIC))
            
            if self.augment:
                post_noise_scale = (np.nanstd(upsampled) + 1e-8) * random.uniform(0.03, 0.1)
                high_freq_noise = np.random.normal(0, post_noise_scale, upsampled.shape).astype(np.float32)
                upsampled += high_freq_noise
            
            gt_raw_np = patch_raw
            input_raw_np = upsampled

            c_fill = float(np.nanmedian(input_raw_np))
            if np.isnan(c_fill): c_fill = 0.0

            t_in = torch.from_numpy(input_raw_np).float().unsqueeze(0)
            t_gt = torch.from_numpy(gt_raw_np).float().unsqueeze(0)
            t_mask = torch.ones_like(t_gt)

            if self.augment:
                if random.random() < 0.5:
                    t_in = TF.hflip(t_in)
                    t_gt = TF.hflip(t_gt)
                    t_mask = TF.hflip(t_mask)
                if random.random() < 0.5:
                    t_in = TF.vflip(t_in)
                    t_gt = TF.vflip(t_gt)
                    t_mask = TF.vflip(t_mask)
                if random.random() < 0.5: 
                    angle = random.uniform(0.0, 360.0)
                    t_in = TF.rotate(t_in, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=c_fill)
                    t_gt = TF.rotate(t_gt, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=c_fill)
                    t_mask = TF.rotate(t_mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0.0)

            gt_aug = t_gt[0].numpy()
            in_aug = t_in[0].numpy()
            patch_mask = t_mask[0]

            _, memory_tensor = apply_patch_asymmetric_arcsinh(
                input_raw_np, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0
            )
            
            alpha = memory_tensor[1].item()
            t_min = memory_tensor[2].item()
            t_max = memory_tensor[3].item()
            
            in_norm_np = np.clip((np.arcsinh(in_aug / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)
            gt_final_norm = np.clip((np.arcsinh(gt_aug / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)

            patch_cond = torch.tensor(in_norm_np, dtype=torch.float32).unsqueeze(0)
            patch_y = torch.tensor(gt_final_norm, dtype=torch.float32).unsqueeze(0)
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
        
        act_map_path = cfg.dataset.get('real_act_path', '/scratch/xuduo117/Real_Obs_Data/Planck_Data/act-planck_dr6.02_coadd_AA_night_f220_map.fits')
        
        cache_dir = os.path.dirname(act_map_path)
        npy_map_path = os.path.join(cache_dir, "act_dr6_fast_map.npy")
        pix_box_path = os.path.join(cache_dir, "act_test_pix_box.npy")
        coords_path = os.path.join(cache_dir, f"act_valid_coords_{mode}.npy")
        
        if not os.path.exists(npy_map_path) or not os.path.exists(pix_box_path):
            if getattr(cfg, 'global_rank', 0) == 0:
                print("⏳ [ONE-TIME SETUP] Converting 6GB ACT FITS to Memory-Mapped NPY...")
                raw_enmap = enmap.read_map(act_map_path)[0]
                act_np = np.asarray(raw_enmap, dtype=np.float32)
                act_np[act_np == 0.0] = np.nan
                np.save(npy_map_path, act_np)
                
                test_ra, test_dec, hw = 265.0, -28.0, 2.5
                dec_min, dec_max = (test_dec - hw) * pix_utils.degree, (test_dec + hw) * pix_utils.degree
                ra_min, ra_max = (test_ra - hw) * pix_utils.degree, (test_ra + hw) * pix_utils.degree
                pix_box = raw_enmap.sky2pix(np.array([[dec_min, ra_max], [dec_max, ra_min]]))
                np.save(pix_box_path, pix_box)
                print("✅ FITS Conversion complete!")
            else:
                import time
                while not os.path.exists(npy_map_path): time.sleep(2)
        
        self.act_map = np.load(npy_map_path, mmap_mode='c')
        self.h, self.w = self.act_map.shape
        pix_box = np.load(pix_box_path)
        
        if os.path.exists(coords_path):
            self.valid_coords = np.load(coords_path).tolist()
        else:
            if getattr(cfg, 'global_rank', 0) == 0:
                valid_mask = ~np.isnan(self.act_map)
                y0, y1 = max(0, int(min(pix_box[0]))), min(self.h, int(max(pix_box[0])))
                x0, x1 = max(0, int(min(pix_box[1]))), min(self.w, int(max(pix_box[1])))
                valid_mask[y0:y1, x0:x1] = False
                split_x = int(self.w * 0.70)
                if mode == 'train': valid_mask[:, split_x:] = False
                elif mode == 'val': valid_mask[:, :split_x] = False
                
                valid_y, valid_x = np.where(valid_mask)
                min_y, max_y = valid_y.min(), valid_y.max()
                min_x, max_x = valid_x.min(), valid_x.max()
                
                self.valid_coords = []
                stride = 128 
                max_invalid_allowed = 0.20 * (self.target_size ** 2)
                for y in range(min_y, max_y - self.target_size + 1, stride):
                    for x in range(min_x, max_x - self.target_size + 1, stride):
                        if (~valid_mask[y:y+self.target_size, x:x+self.target_size]).sum() <= max_invalid_allowed:
                            self.valid_coords.append((y, x))
                np.save(coords_path, np.array(self.valid_coords))
            else:
                import time
                while not os.path.exists(coords_path): time.sleep(1)
                self.valid_coords = np.load(coords_path).tolist()

    def __len__(self):
        return 1000  

    def __getitem__(self, index):
        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []
        
        for _ in range(self.num_patches):
            y_off, x_off = random.choice(self.valid_coords)
            max_safe_y = max(0, min(127, self.h - self.target_size - y_off))
            max_safe_x = max(0, min(127, self.w - self.target_size - x_off))
            
            y_start = y_off + random.randint(0, max_safe_y)
            x_start = x_off + random.randint(0, max_safe_x)
            
            patch_raw = np.array(self.act_map[y_start:y_start+self.target_size, x_start:x_start+self.target_size])
            nan_mask = np.isnan(patch_raw)
            patch_mask_np = (~nan_mask).astype(np.float32)
            
            with np.errstate(invalid='ignore'):
                valid_mean = np.nanmean(patch_raw)
                if np.isnan(valid_mean): valid_mean = 0.0
            patch_raw = np.where(nan_mask, valid_mean, patch_raw)
            
            if self.augment:
                blur_sigma = random.uniform(2.0, 5.0) 
                down_factor = random.uniform(3.0, 5.0)
            else:
                blur_sigma = 3.5 
                down_factor = 5.0 
                
            blurred = gaussian_filter(patch_raw, sigma=blur_sigma, truncate=4.0)
            low_res_size = int(self.target_size / down_factor)
            downsampled = np.array(Image.fromarray(blurred, mode='F').resize((low_res_size, low_res_size), resample=Image.BOX))
            
            # ---> NEW: DYNAMIC HEAVY NOISE ON LOW-RES REAL DATA <---
            if self.augment:
                noise_std = (np.nanstd(downsampled) + 1e-8) * random.uniform(0.02, 0.10)
                noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.float32)
            else:
                noise_std = (np.nanstd(downsampled) + 1e-8) * 0.02
                noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.float32)
            
            upsampled = np.array(Image.fromarray(downsampled + noise, mode='F').resize((self.target_size, self.target_size), resample=Image.BICUBIC))
            
            if self.augment:
                post_noise_scale = (np.nanstd(upsampled) + 1e-8) * random.uniform(0.03, 0.1)
                high_freq_noise = np.random.normal(0, post_noise_scale, upsampled.shape).astype(np.float32)
                upsampled += high_freq_noise
            
            gt_raw_np = patch_raw
            input_raw_np = upsampled

            c_fill = float(np.nanmedian(input_raw_np))
            if np.isnan(c_fill): c_fill = 0.0

            t_in = torch.from_numpy(input_raw_np).float().unsqueeze(0)
            t_gt = torch.from_numpy(gt_raw_np).float().unsqueeze(0)
            t_mask = torch.tensor(patch_mask_np, dtype=torch.float32).unsqueeze(0)

            if self.augment:
                if random.random() < 0.5:
                    t_in = TF.hflip(t_in)
                    t_gt = TF.hflip(t_gt)
                    t_mask = TF.hflip(t_mask)
                if random.random() < 0.5:
                    t_in = TF.vflip(t_in)
                    t_gt = TF.vflip(t_gt)
                    t_mask = TF.vflip(t_mask)
                if random.random() < 0.5: 
                    angle = random.uniform(0.0, 360.0)
                    t_in = TF.rotate(t_in, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=c_fill)
                    t_gt = TF.rotate(t_gt, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=c_fill)
                    t_mask = TF.rotate(t_mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0.0)

            gt_aug = t_gt[0].numpy()
            in_aug = t_in[0].numpy()
            patch_mask = t_mask[0]

            _, memory_tensor = apply_patch_asymmetric_arcsinh(
                input_raw_np, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0
            )
            
            alpha = memory_tensor[1].item()
            t_min = memory_tensor[2].item()
            t_max = memory_tensor[3].item()
            
            in_norm_np = np.clip((np.arcsinh(in_aug / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)
            gt_final_norm = np.clip((np.arcsinh(gt_aug / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)

            patch_cond = torch.tensor(in_norm_np, dtype=torch.float32).unsqueeze(0)
            patch_y = torch.tensor(gt_final_norm, dtype=torch.float32).unsqueeze(0)
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