import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, shift  
import torchvision.transforms.functional as TF

# =========================================================
# STEP-BY-STEP DEBUGGER & HELPERS
# =========================================================
def check_finite(name, arr):
    if not np.isfinite(arr).all():
        nans = np.isnan(arr).sum()
        infs = np.isinf(arr).sum()
        raise ValueError(f"🚨 FAILED AT [{name}]! NaNs: {nans}, Infs: {infs}")
    return arr

def apply_patch_asymmetric_arcsinh(patch, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0):
    c = 0.0  
    s = np.nanstd(patch) + 1e-8
    alpha = s * transition_sigmas
    tf_patch = np.arcsinh(patch / alpha)
    
    abs_min, abs_max = np.nanmin(patch), np.nanmax(patch)
    perc_min, perc_max = np.nanpercentile(patch, 0.1), np.nanpercentile(patch, 99.9)
    sigma_min, sigma_max = c - max_sigma * s, c + max_sigma * s
    
    p_min = max(abs_min, min(perc_min, sigma_min))
    p_max = min(abs_max, max(perc_max, sigma_max))
    
    max_phys = headroom_multiplier * max(0, p_max)
    min_phys = headroom_multiplier * min(0, p_min) 
    
    t_min = np.arcsinh(min_phys / alpha)
    t_max = np.arcsinh(max_phys / alpha)
    if t_max - t_min < 1e-6: t_max = t_min + 1.0
        
    final_norm = np.clip((tf_patch - t_min) / (t_max - t_min), 0.0, 1.0)
    return final_norm, torch.tensor([c, alpha, t_min, t_max], dtype=torch.float32)

def safe_resize(img_np, size, mode='bicubic', antialias=True):
    """
    Replaces PIL.Image.resize to strictly enforce anti-aliasing.
    Prevents high-frequency grid artifacts during downsampling.
    """
    t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    # PyTorch natively applies the correct low-pass filter when antialias=True
    t = F.interpolate(t, size=(size, size), mode=mode, antialias=antialias)
    return t.squeeze(0).squeeze(0).numpy()

def inject_point_sources(base_signal, max_sources=15, pixel_scale_arcmin=0.5):
    # 30% chance to have no point sources at all
    if random.random() < 0.3: return base_signal
    
    h, w = base_signal.shape
    signal_max = np.nanmax(base_signal)
    if np.isnan(signal_max) or signal_max <= 0: signal_max = 1e-3
    
    ps_map = np.zeros_like(base_signal)
    
    # 50/50 split to enhance diversity between faint and extreme point sources
    if random.random() < 0.5:
        # MODE 1: Subtle & Sparse (Max 4 sources, 0.5x to 2.0x amplitude)
        # Prevents overwhelming the loss, adds faint background diversity
        num_sources = random.randint(1, 4)
        for _ in range(num_sources):
            y, x = random.randint(0, h-1), random.randint(0, w-1)
            ps_map[y, x] = signal_max * random.uniform(0.5, 2.0)
            
    else:
        # MODE 2: Extreme & Dense (Max 15 sources, 2.0x to 8.0x amplitude)
        # Teaches the network to handle extreme dynamic range (quasars/cores)
        num_sources = random.randint(1, max_sources)
        for _ in range(num_sources):
            y, x = random.randint(0, h-1), random.randint(0, w-1)
            ps_map[y, x] = signal_max * random.uniform(1.0, 10.0)
            
        ps_max = np.nanmax(ps_map)
        if ps_max > 0:
            ps_map = ps_map * (signal_max * random.uniform(2.0, 8.0) / ps_max)

    # Injected as pure single pixels (Dirac deltas). 
    # The downstream act_blurred = gaussian_filter(...) will handle the exact telescope beam physics.
    return base_signal + ps_map


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
    cmb_map = np.real(np.fft.ifft2(white_noise * np.sqrt(c_ell)))
    return (cmb_map - np.nanmean(cmb_map)) / (np.nanstd(cmb_map) + 1e-8)

def generate_diffuse_cirrus(h, w, pixel_scale_arcmin):
    """
    Generates faint, extended fractal dust clouds (Galactic Cirrus).
    Includes a high-frequency shelf for sub-beam fractal clumping.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0
    
    # ISM turbulent dust power spectrum
    power_spectrum = 1.0 / (f ** 2.7)
    
    # ---> NEW: High-frequency fractal shelf <---
    # This forces the cloud to have a "grainy" sub-structure
    power_spectrum += 0.05 
    
    power_spectrum[0, 0] = 0.0
    
    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    cirrus = np.real(np.fft.ifft2(white_noise * np.sqrt(power_spectrum)))
    cirrus = (cirrus - np.nanmean(cirrus)) / (np.nanstd(cirrus) + 1e-8)
    
    cirrus = np.exp(cirrus * 0.8)
    cirrus = (cirrus - np.nanmin(cirrus)) / (np.nanmax(cirrus) - np.nanmin(cirrus) + 1e-8)
    return cirrus.astype(np.float32)


def generate_clumpy_pepper_noise(h, w, pixel_scale_arcmin):
    """
    Generates a correlated noise field that mimics the 'pepper-like' 
    clumpy texture seen in real ACT backgrounds.
    """
    pixel_scale_rad = pixel_scale_arcmin * (np.pi / 180.0 / 60.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    
    ell = 2 * np.pi * f / pixel_scale_rad
    ell[0, 0] = 1.0  
    
    # Create a power spectrum that peaks at intermediate/small scales
    # This creates the "clumpy" look, avoiding massive blobs or pure static
    ell_peak = 6000.0  # Adjust this to change the typical size of the "pepper" clumps
    width = 3000.0
    
    # A log-normal bump in the power spectrum creates distinct clumps
    power_spectrum = np.exp(-((np.log(ell) - np.log(ell_peak))**2) / (2 * (width/ell_peak)**2))
    
    # Add a slight white noise floor
    power_spectrum += 0.05
    
    # Anti-aliasing taper
    taper = np.exp(-(ell / 15000.0)**3)
    power_spectrum *= taper
    
    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    noise_fourier = white_noise * np.sqrt(power_spectrum)
    
    noise_map = np.real(np.fft.ifft2(noise_fourier))
    noise_map = (noise_map - np.nanmean(noise_map)) / (np.nanstd(noise_map) + 1e-8)
    
    return noise_map

def generate_magnetic_striations(h, w):
    """
    Generates highly anisotropic, straight, needle-like fibers.
    Simulates PAH/dust alignment with magnetic fields or shock fronts (e.g., WISE 12um).
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    
    # Combine 1 to 3 distinct angles to simulate crossing shocks/fields
    num_layers = random.randint(1, 3)
    final_striations = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(num_layers):
        angle = random.uniform(0, np.pi)
        
        # Rotate Fourier coordinates
        kx = fx * np.cos(angle) + fy * np.sin(angle)
        ky = -fx * np.sin(angle) + fy * np.cos(angle)
        
        # Anisotropic parameters: Very long length (kx), very sharp width (ky)
        length_scale = random.uniform(50.0, 200.0)
        width_scale = random.uniform(1.0, 5.0)
        
        k_sq = (kx * length_scale)**2 + (ky * width_scale)**2
        k_sq[0, 0] = 1.0  
        
        # Steep power law to make the fibers highly coherent
        power_spectrum = 1.0 / (k_sq ** random.uniform(1.2, 1.8))
        power_spectrum[0, 0] = 0.0
        
        white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
        layer = np.real(np.fft.ifft2(white_noise * np.sqrt(power_spectrum)))
        
        # Exponentiate to create sharp ridges rather than soft sine waves
        layer = (layer - np.nanmean(layer)) / (np.nanstd(layer) + 1e-8)
        layer = np.exp(layer * random.uniform(1.5, 3.0))
        
        final_striations += layer.astype(np.float32)
        
    # Modulate with a large-scale blob so the lines aren't uniform everywhere
    env_f = np.sqrt(fx**2 + fy**2)
    env_f[0, 0] = 1.0
    env_p = 1.0 / (env_f ** 3.0)
    env_p[0, 0] = 0.0
    env = np.real(np.fft.ifft2((np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))) * np.sqrt(env_p)))
    env = (env - np.nanmin(env)) / (np.nanmax(env) - np.nanmin(env) + 1e-8)
    
    final_striations = final_striations * env
    final_striations = (final_striations - np.nanmin(final_striations)) / (np.nanmax(final_striations) - np.nanmin(final_striations) + 1e-8)
    
    return final_striations


def generate_cib_field(h, w, local_signal_max, pixel_scale_arcmin=0.5, dense_mode=False):
    """
    Simulates the Cosmic Infrared Background (CIB) Confusion Limit.
    """
    cib_map = np.zeros((h, w), dtype=np.float32)
    
    # Massive source density to hit the confusion limit
    num_sources = random.randint(30000, 80000) if dense_mode else random.randint(10000, 30000)
    
    # ---> BUFFED: Increased the flux scale slightly so the model doesn't ignore it <---
    flux_scale = local_signal_max * (random.uniform(0.02, 0.08) if dense_mode else 0.015)
    fluxes = np.random.exponential(scale=flux_scale, size=num_sources)
    
    y_coords = np.random.randint(0, h, num_sources)
    x_coords = np.random.randint(0, w, num_sources)
    np.add.at(cib_map, (y_coords, x_coords), fluxes)
    
    # Left as pure Dirac deltas so they become sharp texture in the GT
    return cib_map


def apply_filament_fragmentation(patch):
    """
    Breaks continuous continuous gas filaments into patchy, disconnected clumps 
    by applying a low-frequency, high-contrast modulation mask.
    """
    h, w = patch.shape
    noise = np.random.normal(0, 1, (h, w))
    
    # Create large, smooth blobs (size scaled to the image)
    blur_sigma = random.uniform(h/12, h/4)
    mask = gaussian_filter(noise, sigma=blur_sigma)
    
    # Normalize the mask to exactly [0, 1]
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-8)
    
    # Push the contrast aggressively. Squaring/Cubing it makes the "valleys" 
    # much wider and deeper, slicing cleanly through the filaments.
    mask = mask ** random.uniform(2.0, 4.0)
    
    # Add a small floor so the carved-out valleys don't hit absolute zero
    floor = random.uniform(0.05, 0.25)
    mask = floor + (1.0 - floor) * mask
    
    return patch * mask

def generate_synthetic_act_noise(h, w, pixel_scale_arcmin):
    pixel_scale_rad = pixel_scale_arcmin * (np.pi / 180.0 / 60.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    
    ell = 2 * np.pi * f / pixel_scale_rad
    ell[0, 0] = 1.0  
    
    ell_knee = 3000.0
    alpha = 3.0
    
    n_ell = 1.0 + (ell_knee / ell)**alpha
    taper = np.exp(-(ell / 15000.0)**3)
    
    power_spectrum = n_ell * taper
    
    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    noise_fourier = white_noise * np.sqrt(power_spectrum)
    
    noise_map = np.real(np.fft.ifft2(noise_fourier))
    noise_map = (noise_map - np.nanmean(noise_map)) / (np.nanstd(noise_map) + 1e-8)
    return noise_map


def apply_scanning_stripes(patch, intensity=0.15):
    h, w = patch.shape
    local_std = np.nanstd(patch) + 1e-8
    f = np.fft.fftfreq(h)
    f[0] = 1.0
    amp = 1.0 / np.sqrt(np.abs(f))
    amp[0] = 0.0
    complex_noise = np.random.normal(0, 1, h) + 1j * np.random.normal(0, 1, h)
    stripe_1d = np.real(np.fft.ifft(complex_noise * amp))
    stripe_1d = (stripe_1d - np.nanmean(stripe_1d)) / (np.nanstd(stripe_1d) + 1e-8)
    return patch + (np.tile(stripe_1d[:, None], (1, w)) * local_std * intensity)

def apply_advanced_telescope_noise(low_res_map, augment=True):
    local_std = np.nanstd(low_res_map) + 1e-8
    if not augment: return (low_res_map + np.random.normal(0, local_std * 0.02, low_res_map.shape)).astype(np.float32)
        
    h, w = low_res_map.shape
    total_noise = np.zeros_like(low_res_map)
    dynamic_noise_base = local_std * random.uniform(0.10, 0.30)
    total_noise += np.random.normal(0, dynamic_noise_base * random.uniform(0.4, 0.8), (h, w))

    if random.random() < 0.50:
        shifted_signal = np.clip(low_res_map - np.nanmin(low_res_map) + 1e-6, 0.0, None)
        noise_envelope = np.sqrt(shifted_signal)
        noise_envelope /= (np.nanmean(noise_envelope) + 1e-8)
        total_noise += np.random.normal(0, 1, (h, w)) * noise_envelope * (dynamic_noise_base * random.uniform(0.3, 1.0))
    return (low_res_map + total_noise).astype(np.float32)

# =========================================================
# CORE PROCESSING PIPELINE (Shared across datasets)
# =========================================================
def process_noise_to_clean_patch(cropped_view, augment, target_size, planck_size, total_fov_arcmin, crop_size, add_point_sources=True):
    cropped_view = np.nan_to_num(cropped_view, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    local_max = np.nanmax(np.abs(cropped_view))
    if local_max == 0 or np.isnan(local_max): local_max = 1.0
    cropped_view = (cropped_view / local_max).astype(np.float32)
    cropped_view = check_finite("Base Crop", cropped_view)

    base_signal = cropped_view.copy()
    pixel_scale_arcmin = total_fov_arcmin / crop_size
    
    is_null_patch = False
    heavy_cib_mode = False
    
    if augment:
        rand_roll = random.random()
        h_crop, w_crop = base_signal.shape
        
        # ---> 1. DRAMATICALLY REDUCED VACUUM PRIOR <---
        # Only 5% chance of pure vacuum (instead of 30%)
        if rand_roll < 0.05:
            base_signal = np.zeros_like(base_signal)
            is_null_patch = True
            heavy_cib_mode = True
            
        # Only 10% chance to heavily suppress the main filaments
        elif rand_roll < 0.15:
            base_signal = base_signal * random.uniform(0.05, 0.2)
            heavy_cib_mode = True
            
        # Softened fragmentation (Doesn't carve as deeply to zero)
        elif random.random() < 0.3:  
            base_signal = apply_filament_fragmentation(base_signal)
            # Raise the floor so we don't destroy the diffuse background
            base_signal += random.uniform(0.05, 0.15) * np.nanmax(base_signal)
            
        if is_null_patch:
            effective_scale = random.uniform(0.1, 0.3) 
        else:
            effective_scale = max(np.nanpercentile(cropped_view, 99.5), 1e-3)
        
        # ---> 2. INJECT DIFFUSE GALACTIC CIRRUS PRIOR <---
        # 75% of the time, inject a faint, structured dust background
        if not is_null_patch and random.random() < 0.75:
            cirrus_bg = generate_diffuse_cirrus(h_crop, w_crop, pixel_scale_arcmin)
            # Make it 5% to 25% the brightness of the main filaments
            cirrus_amp = effective_scale * random.uniform(0.05, 0.25)
            base_signal += (cirrus_bg * cirrus_amp)
            
        if not is_null_patch and random.random() < 0.50:
            striations = generate_magnetic_striations(h_crop, w_crop)
            striation_amp = effective_scale * random.uniform(0.1, 0.4)
            base_signal += (striations * striation_amp)

        # Add CMB (Slightly fainter than the Cirrus)
        if random.random() < 0.9:
            cmb_base = generate_cmb_grf(h_crop, w_crop, pixel_scale_arcmin)
            if is_null_patch or heavy_cib_mode:
                cmb_amp = effective_scale * random.uniform(0.15, 0.4)
            else:
                cmb_amp = effective_scale * random.uniform(0.01, 0.05)
            base_signal += (cmb_base * cmb_amp)
            
        # Add Dense CIB Background (The Salt-and-Pepper)
        # ---> NEW: Trigger 95% of the time, because the CIB is everywhere in the universe <---
        if heavy_cib_mode or random.random() < 0.95:  
            cib_base = generate_cib_field(h_crop, w_crop, effective_scale, pixel_scale_arcmin, dense_mode=heavy_cib_mode)
            base_signal += cib_base
            
        # Add Bright Point Sources 
        if add_point_sources and not is_null_patch:
            base_signal = inject_point_sources(base_signal, max_sources=15, pixel_scale_arcmin=pixel_scale_arcmin)
            
    base_signal = check_finite("Base Signal", base_signal)
    

    if augment:
        act_fwhm = random.uniform(0.4, 3.5)
        planck_fwhm = act_fwhm * random.uniform(3.0, 6.0)
    else:
        act_fwhm, planck_fwhm = 3.0, 15.0
    
    # Clean Target
    act_sigma = (act_fwhm * (crop_size / target_size)) / 2.355
    act_blurred = check_finite("ACT Blur", gaussian_filter(base_signal, sigma=act_sigma, truncate=4.0).astype(np.float32))
    
    gt_raw_np = safe_resize(act_blurred, target_size, mode='bicubic', antialias=True)
    
    # if augment and random.random() < 0.8:
    #     act_noise_base = generate_synthetic_act_noise(target_size, target_size, total_fov_arcmin / target_size)
    #     pepper_noise = generate_clumpy_pepper_noise(target_size, target_size, total_fov_arcmin / target_size)
        
    #     combined_noise = (act_noise_base * 0.5) + (pepper_noise * 0.5)
    #     combined_noise = (combined_noise - np.mean(combined_noise)) / (np.std(combined_noise) + 1e-8)
        
    #     signal_std = np.nanstd(gt_raw_np) + 1e-8
    #     noise_amplitude = signal_std * random.uniform(0.02, 0.08) 
    #     gt_raw_np += (combined_noise * noise_amplitude).astype(np.float32)

    gt_raw_np = check_finite("GT Target", gt_raw_np)
    
    # Degraded Input
    planck_sigma = (planck_fwhm * (crop_size / target_size)) / 2.355
    planck_blurred = check_finite("Planck Blur", gaussian_filter(base_signal, sigma=planck_sigma, truncate=4.0).astype(np.float32))
    
    planck_linear = safe_resize(planck_blurred, planck_size, mode='bicubic', antialias=True)
    planck_linear = check_finite("Planck Linear", planck_linear)
    
    planck_noisy = apply_advanced_telescope_noise(planck_linear, augment=augment)
    if augment and random.random() < 0.5: 
        planck_noisy = apply_scanning_stripes(planck_noisy, intensity=random.uniform(0.1, 0.3))
    planck_noisy = check_finite("Planck Noise", planck_noisy.astype(np.float32))
    
    planck_upsampled = safe_resize(planck_noisy, target_size, mode='bicubic', antialias=False)
    
    if augment: 
        planck_upsampled += np.random.normal(0, (np.nanstd(planck_upsampled) + 1e-8) * random.uniform(0.01, 0.03), planck_upsampled.shape).astype(np.float32)
        
        # Only shift if there are actually structures to shift (avoids uniform edge artifacts on zero maps)
        if random.random() < 0.5 and not is_null_patch:
            shift_y = random.uniform(-0.5, 0.5)
            shift_x = random.uniform(-0.5, 0.5)
            planck_upsampled = shift(planck_upsampled, shift=(shift_y, shift_x), order=1, mode='nearest').astype(np.float32)
            
    input_raw_np = check_finite("Input Final", planck_upsampled)

    c_fill = float(np.nanmedian(input_raw_np))
    if np.isnan(c_fill): c_fill = 0.0

    t_in, t_gt = torch.from_numpy(input_raw_np).unsqueeze(0), torch.from_numpy(gt_raw_np).unsqueeze(0)
    t_mask = torch.ones_like(t_gt)

    if augment:
        if random.random() < 0.5: t_in, t_gt = TF.hflip(t_in), TF.hflip(t_gt)
        if random.random() < 0.5: t_in, t_gt = TF.vflip(t_in), TF.vflip(t_gt)
        if random.random() < 0.5: 
            angle = random.uniform(0.0, 360.0)
            t_in, t_gt = TF.rotate(t_in, angle, fill=c_fill), TF.rotate(t_gt, angle, fill=c_fill)

    _, memory_tensor = apply_patch_asymmetric_arcsinh(input_raw_np, headroom_multiplier=2.5)
    
    alpha, t_min, t_max = memory_tensor[1].item(), memory_tensor[2].item(), memory_tensor[3].item()
    in_norm_np = np.clip((np.arcsinh(t_in[0].numpy() / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)
    gt_final_norm = np.clip((np.arcsinh(t_gt[0].numpy() / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)

    memory_tensor[0] *= local_max  
    memory_tensor[1] *= local_max  

    patch_cond = torch.tensor(in_norm_np, dtype=torch.float32).unsqueeze(0)
    patch_y = torch.tensor(gt_final_norm, dtype=torch.float32).unsqueeze(0)
    
    return patch_cond.clone(), patch_cond, patch_y, t_mask[0], memory_tensor


# =========================================================
# DATASET 1: RAW DUST MAPS (H, W, Views)
# =========================================================
class SyntheticAstroDataset(Dataset):
    def __init__(self, cfg, file_list, mode='train'):
        self.cfg = cfg
        self.file_list = file_list
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)
        
        self.target_size = 256   
        self.planck_size = 51    
        self.min_crop, self.max_crop = 512, 1536
        self.val_crop_scales = [512, 768, 1024, 1280, 1536]
        self.total_fov_arcmin = self.target_size * 0.5

    def __len__(self): return len(self.file_list)

    def __getitem__(self, index):
        raw_cube = np.load(self.file_list[index], mmap_mode='c')
        h, w, num_views = raw_cube.shape 
        max_c = min(h, w)
        act_min, act_max = min(self.min_crop, max_c), min(self.max_crop, max_c)
        
        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []

        for _ in range(self.num_patches):
            v_idx = random.randint(0, num_views - 1)
            if self.mode == 'train':
                c_size = random.randint(act_min, act_max)
                y_off, x_off = random.randint(0, h - c_size), random.randint(0, w - c_size)
            else:
                valid = [c for c in self.val_crop_scales if c <= max_c]
                c_size = random.choice(valid) if valid else max_c
                y_off, x_off = (h - c_size) // 2, (w - c_size) // 2 
                
            crop = raw_cube[y_off:y_off+c_size, x_off:x_off+c_size, v_idx].copy()
            # ---> Raw Dust gets point sources <---
            p1, p2, py, pm, pmem = process_noise_to_clean_patch(crop, self.augment, self.target_size, self.planck_size, self.total_fov_arcmin, c_size, add_point_sources=True)
            
            x_prim_list.append(p1); x_cond_list.append(p2); y_list.append(py); mask_list.append(pm); memory_list.append(pmem)

        return (torch.stack(x_prim_list), torch.stack(x_cond_list), torch.stack(y_list), torch.stack(mask_list), torch.stack(memory_list), index)


# =========================================================
# DATASET 2: SKIRT MULTIBAND (Views, Bands, H, W)
# =========================================================
class SkirtAstroDataset(Dataset):
    def __init__(self, cfg, file_list, mode='train'):
        self.cfg = cfg
        self.file_list = file_list
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)
        
        self.target_size = 256   
        self.planck_size = 51    
        self.min_crop, self.max_crop = 512, 1536
        self.val_crop_scales = [512, 768, 1024, 1280, 1536]
        self.total_fov_arcmin = self.target_size * 0.5

    def __len__(self): return len(self.file_list)

    def __getitem__(self, index):
        raw_cube = np.load(self.file_list[index], mmap_mode='c')
        num_views, num_bands, h, w = raw_cube.shape 
        max_c = min(h, w)
        act_min, act_max = min(self.min_crop, max_c), min(self.max_crop, max_c)
        
        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []

        for _ in range(self.num_patches):
            v_idx = random.randint(0, num_views - 1)
            b_idx = random.randint(0, num_bands - 1)
            
            if self.mode == 'train':
                c_size = random.randint(act_min, act_max)
                y_off, x_off = random.randint(0, h - c_size), random.randint(0, w - c_size)
            else:
                valid = [c for c in self.val_crop_scales if c <= max_c]
                c_size = random.choice(valid) if valid else max_c
                y_off, x_off = (h - c_size) // 2, (w - c_size) // 2 
                
            crop = raw_cube[v_idx, b_idx, y_off:y_off+c_size, x_off:x_off+c_size].copy()
            # ---> SKIRT does NOT get artificial point sources <---
            p1, p2, py, pm, pmem = process_noise_to_clean_patch(crop, self.augment, self.target_size, self.planck_size, self.total_fov_arcmin, c_size, add_point_sources=False)
            
            x_prim_list.append(p1); x_cond_list.append(p2); y_list.append(py); mask_list.append(pm); memory_list.append(pmem)

        return (torch.stack(x_prim_list), torch.stack(x_cond_list), torch.stack(y_list), torch.stack(mask_list), torch.stack(memory_list), index)


# =========================================================
# WRAPPER: COMBINED MIXER (75% Dust / 25% SKIRT)
# =========================================================
class CombinedSyntheticDataset(Dataset):
    def __init__(self, dust_dataset, skirt_dataset, skirt_fraction=0.25):
        self.dust_dataset = dust_dataset
        self.skirt_dataset = skirt_dataset
        self.skirt_fraction = skirt_fraction

    def __len__(self):
        return len(self.dust_dataset) + len(self.skirt_dataset)

    def __getitem__(self, index):
        if len(self.skirt_dataset) > 0 and random.random() < self.skirt_fraction:
            idx = random.randint(0, len(self.skirt_dataset) - 1)
            return self.skirt_dataset[idx]
        else:
            idx = random.randint(0, len(self.dust_dataset) - 1)
            return self.dust_dataset[idx]