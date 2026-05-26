import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from utils.util import unsqueeze_xdim 

def compute_gaussian_product_coef(sigma1, sigma2):
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class AstroDiffusionBridge:
    def __init__(self, cfg, device):
        self.device = device
        self.num_steps = cfg.model.intervals
        
        # Linear beta schedule for the Schrödinger Bridge
        betas = np.linspace(0, cfg.model.beta_max, self.num_steps)
        
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        self.mu_x0 = to_torch(mu_x0)
        self.mu_x1 = to_torch(mu_x1)
        self.std_sb = to_torch(std_sb)
        self.std_fwd = to_torch(std_fwd)
        self.std_bwd = to_torch(std_bwd)

    def q_sample(self, step, x0, x1):
        """Forward sampling: interpolates between target (x0) and primary obs (x1)."""
        mu_x0 = unsqueeze_xdim(self.mu_x0[step], x0.shape[1:])
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], x0.shape[1:])
        std_sb = unsqueeze_xdim(self.std_sb[step], x0.shape[1:])
        
        noise = torch.randn_like(x0)
        return mu_x0 * x0 + mu_x1 * x1 + std_sb * noise

    def p_posterior(self, prev_step, step, xt, pred_x0, ot_ode=False):
        """Backward step: x_{t} -> x_{t-1} using predicted x_0 (B-field)."""
        mu_x0_t = unsqueeze_xdim(self.mu_x0[step], xt.shape[1:])
        mu_x1_t = unsqueeze_xdim(self.mu_x1[step], xt.shape[1:])
        
        mu_x0_prev = unsqueeze_xdim(self.mu_x0[prev_step], xt.shape[1:])
        mu_x1_prev = unsqueeze_xdim(self.mu_x1[prev_step], xt.shape[1:])
        std_sb_prev = unsqueeze_xdim(self.std_sb[prev_step], xt.shape[1:])
        std_sb_t = unsqueeze_xdim(self.std_sb[step], xt.shape[1:])

        # Solve for x1 in bridge equation: pred_x1 = (xt - mu0*pred_x0) / mu1
        pred_x1 = (xt - mu_x0_t * pred_x0) / (mu_x1_t + 1e-8)
        
        xt_prev_mean = mu_x0_prev * pred_x0 + mu_x1_prev * pred_x1
        
        if ot_ode:
            return xt_prev_mean
        
        var = (std_sb_prev**2) * (1 - (std_sb_prev/std_sb_t)**2)
        noise = torch.randn_like(xt)
        return xt_prev_mean + torch.sqrt(var.clamp(min=0)) * noise

    @torch.no_grad()
    def ddpm_sampling(self, model, x_start, cond_img=None, cond_spec=None, cond_text=None, 
                      verbose=True, skip_steps=20):
        """
        Multimodal accelerated inference loop.
        x_start: Projected primary observable (start of bridge).
        """
        xt = x_start.clone()
        
        # 1. Create strided schedule
        all_steps = np.arange(self.num_steps)
        steps = all_steps[::skip_steps][::-1] 
        if steps[-1] != 0:
            steps = np.append(steps, 0)
            
        pbar = tqdm(zip(steps[:-1], steps[1:]), 
                    desc=f"Bridge Sampling (Stride {skip_steps})", 
                    total=len(steps)-1) if verbose else zip(steps[:-1], steps[1:])
        
        for step, prev_step in pbar:
            t_batch = torch.full((x_start.shape[0],), step, device=self.device, dtype=torch.long)
            
            pred_y = model(
                xt, t_batch, 
                cond_img=cond_img, 
                cond_spec=cond_spec, 
                cond_text=cond_text
            )
            
            # 2. Uplift Prediction to Latent Bridge Space [B, 4, H, W]
            model_engine = model.module if hasattr(model, 'module') else model
            pred_x0_latent = model_engine.apply_target_uplift(pred_y)
            
            # 3. Step Backward in 4-channel space
            xt = self.p_posterior(prev_step, step, xt, pred_x0_latent)
            
        final_prediction = model(xt, torch.zeros((xt.shape[0],), device=self.device, dtype=torch.long), 
                                 cond_img=cond_img, cond_spec=cond_spec, cond_text=cond_text)
        
        return final_prediction # This is now [B, 1, H, W] and constrained by Sigmoid
    
    
def _to_diffusion_domain(X):
    """Pass-through: Data is already scaled to [-1, 1] by the Trainer/Inference scripts."""
    return X

def _to_physical_domain(X):
    """Strictly clamp predictions to the [-1, 1] diffusion bounds before returning."""
    return torch.clamp(X, -1.0, 1.0)


class DDPMHandler:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.num_timesteps = cfg.model.intervals
        
        schedule_type = cfg.model.get('beta_schedule', 'linear')
    
        if schedule_type == 'cosine':
            # 1. Cosine Beta Schedule (Nichol & Dhariwal 2021)
            def cosine_beta_schedule(timesteps, s=0.008):
                steps = timesteps + 1
                x = torch.linspace(0, timesteps, steps)
                # alpha_bar logic
                alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                # Derive betas from alphas_cumprod
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                return torch.clip(betas, 0.0001, 0.02) # Standard clipping for stability

            self.betas = cosine_beta_schedule(self.num_timesteps).to(device)
        else:
            # Fallback to your existing linear logic
            beta_start = cfg.model.get('beta_start', 0.0001)
            beta_end = cfg.model.get('beta_end', 0.02)
            self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps).to(device)
        
        # 2. Derive Alpha constants
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        
        # 3. Calculation constants for posterior mean / q_sample
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _extract(self, a, t, x_shape):
        """Extracts values from a 1D tensor at indices t and reshapes for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion (Noising process):
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, cond_img=None, cond_spec=None, cond_text=None):
        """
        Reverse sampling step: Predicts x_{t-1} from x_t
        """
        # Model predicts the noise (epsilon) that was added to x_0
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # 1. Predict Noise
        model_output = model(x, t, cond_img=cond_img, cond_spec=cond_spec, cond_text=cond_text)
        
        # 2. Compute Mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            # Add Langevin dynamics noise (sigma * z)
            # Posterior variance setup
            posterior_variance = betas_t * (1.0 - self._extract(self.alphas_cumprod_prev, t, x.shape)) / (1.0 - self._extract(self.alphas_cumprod, t, x.shape))
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond_img=None, cond_spec=None, cond_text=None, 
                    eta=0.0, skip_steps=50):
        """
        Accelerated DDIM sampling loop.
        eta=0.0: Deterministic (DDIM)
        skip_steps=50: Subsamples the 1000-step process into 50 steps.
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # 1. Start from pure noise x_T ~ N(0, 1)
        img = torch.randn(shape, device=device)
        
        if cond_img is not None:
            cond_img = _to_diffusion_domain(cond_img)
            
        # 2. Define the accelerated sequence of timesteps
        times = torch.linspace(0, self.num_timesteps - 1, skip_steps).long().tolist()
        times = list(reversed(times))  
        time_pairs = list(zip(times[:-1], times[1:]))

        for i, j in tqdm(time_pairs, desc=f"DDIM Sampling (eta={eta})", leave=False):
            t = torch.full((b,), i, device=device, dtype=torch.long)
                
            # 3. Predict noise (epsilon) using the backbone
            eps = model(img, t, cond_img=cond_img, cond_spec=cond_spec, cond_text=cond_text)
            
            # 4. DDIM Math
            alpha_t = self.alphas_cumprod[i]
            alpha_prev = self.alphas_cumprod[j]
            
            # Calculate sigma for stochasticity (if eta > 0)
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            
            # Predict x0 (the "clean" estimate at this step)
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            
            # --- FIX: Clamp to [-1.0, 1.0] Gaussian bounds! ---
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev - sigma_t**2) * eps
            
            # Add random noise only if eta > 0
            noise = torch.randn(shape, device=device) if sigma_t > 0 else 0
            
            # Compute x_{t-1}
            img = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise

        return _to_physical_domain(pred_x0)
    
    @torch.no_grad()
    def solve_ode(self, model, shape, cond_img, skip_steps=50):
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        # 1. Shift condition
        if cond_img is not None: cond_img = _to_diffusion_domain(cond_img)
        
        times = torch.linspace(self.num_timesteps - 1, 0, skip_steps).long().to(device)
        time_pairs = list(zip(times[:-1], times[1:]))

        for i, j in tqdm(time_pairs, desc="ODE Solver (Deterministic)"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            eps = model(img, t, cond_img=cond_img)
            
            alpha_i = self.alphas_cumprod[i]
            alpha_j = self.alphas_cumprod[j]
            
            pred_x0 = (img - torch.sqrt(1 - alpha_i) * eps) / torch.sqrt(alpha_i)
            
            # --- FIX: Clamp to [-1.0, 1.0] Gaussian bounds! ---
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            img = torch.sqrt(alpha_j) * pred_x0 + torch.sqrt(1 - alpha_j) * eps
            
        # 2. Return physical data
        return _to_physical_domain(pred_x0)

    @torch.no_grad()
    def solve_sde(self, model, shape, cond_img, skip_steps=50, eta=0.8):
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        # 1. Shift condition
        if cond_img is not None: cond_img = _to_diffusion_domain(cond_img)
        
        times = torch.linspace(self.num_timesteps - 1, 0, skip_steps).long().to(device)
        time_pairs = list(zip(times[:-1], times[1:]))

        for i, j in tqdm(time_pairs, desc=f"SDE Solver (eta={eta})"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            eps = model(img, t, cond_img=cond_img)
            
            alpha_i = self.alphas_cumprod[i]
            alpha_j = self.alphas_cumprod[j]
            sigma = eta * torch.sqrt((1 - alpha_j) / (1 - alpha_i) * (1 - alpha_i / alpha_j))
            
            pred_x0 = (img - torch.sqrt(1 - alpha_i) * eps) / torch.sqrt(alpha_i)
            
            # --- FIX: Clamp to [-1.0, 1.0] Gaussian bounds! ---
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            dir_xt = torch.sqrt(1 - alpha_j - sigma**2) * eps
            noise = torch.randn_like(img)
            img = torch.sqrt(alpha_j) * pred_x0 + dir_xt + sigma * noise
            
        # 2. Return physical data
        return _to_physical_domain(pred_x0)
    
    @torch.no_grad()
    def solve_ancestral(self, model, shape, cond_img=None):
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        # 1. Shift condition
        if cond_img is not None: cond_img = _to_diffusion_domain(cond_img)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, t_index=i, cond_img=cond_img)
            
            # --- FIX: Clamp to [-1.0, 1.0] Gaussian bounds! ---
            img = torch.clamp(img, -1.0, 1.0)
            
        # 2. Return physical data
        return _to_physical_domain(img)