# core/network.py
import torch
import torch.nn as nn
from .unet import UNetModel  # This links the two files
from utils.util import count_parameters

import torch
import torch.nn as nn
from .unet import UNetModel

import torch
import torch.nn as nn
from .unet import UNetModel
from utils.util import count_parameters

import torch
import torch.nn as nn
from .unet import UNetModel

class AstroMultimodalUNet(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, noise_levels):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("noise_levels", noise_levels)
        
        # 1. LATENT BRIDGE SETUP (Primary High-Bandwidth Info)
        # Match latent_dim to primary_indices (e.g., 4: Density, Q, U, Vel)
        self.latent_dim = in_channels 
        self.bridge_dim = self.latent_dim
        self.primary_proj = nn.Identity()

        # 2. DYNAMIC CONDITIONING (Secondary Guidance)
        self.current_cond_channels = 0
        
        # A. Image-based Guidance (Secondary maps not in the bridge)
        cond_in_img = len(cfg.dataset.get('cond_indices', []))
        if cond_in_img > 0:
            # d_img = cfg.model.get('cond_dim_img', 4) 
            # self.img_cond_encoder = nn.Sequential(
            #     nn.Conv2d(cond_in_img, 64, 1), nn.SiLU(),
            #     nn.Conv2d(64, d_img, 1)
            # )
            # self.current_cond_channels += d_img
            self.current_cond_channels += cond_in_img
            
        # B. Spectral Guidance
        self.use_spectrum = cfg.model.get('use_spectrum', False)
        if self.use_spectrum:
            d_spec = cfg.model.get('cond_dim_spec', 2)
            self.spec_encoder = SpectralEncoder(cfg.model.get('spectrum_in_channels', 64), d_spec)
            self.current_cond_channels += d_spec
            
        # C. Text Guidance
        self.use_text = cfg.model.get('use_text', False)
        if self.use_text:
            d_text = cfg.model.get('cond_dim_text', 1)
            self.text_encoder = TextEncoder(cfg.model.get('text_in_dim', 768), d_text)
            self.current_cond_channels += d_text
            
        # 3. MAIN U-NET CONFIGURATION
        # Total channels = Latent Bridge (4) + Image Guide + Spec Guide + Text Guide
        total_unet_in = self.bridge_dim + self.current_cond_channels
        model_channels = cfg.model.base_channels

        self.diffusion_model = UNetModel(
            image_size=cfg.dataset.patch_size,
            in_channels=total_unet_in,
            model_channels=model_channels,
            out_channels=self.latent_dim, # Output must match Latent Bridge width
            num_res_blocks=cfg.model.num_res_blocks,
            attention_resolutions=tuple(cfg.model.attention_resolutions),
            dropout=cfg.model.get('dropout', 0.1),
            channel_mult=tuple(cfg.model.channel_mult),
            num_heads=cfg.model.get('num_heads', 1),
            use_scale_shift_norm=True,
            resblock_updown=True
        )

        # 4. PROJECTION HEADS
        # Reconstruct physical 1-channel target from 4-channel latent space
        self.projection_head = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.latent_dim * 2, out_channels, kernel_size=1),
            nn.Sigmoid() 
        )

        # Map 1-channel target to 4-channel latent space for DSB math
        self.target_uplift = nn.Conv2d(out_channels, self.latent_dim, kernel_size=1)
        nn.init.constant_(self.target_uplift.weight, 1.0)
        nn.init.constant_(self.target_uplift.bias, 0.0)

    def forward(self, xt, steps, cond_img=None, cond_spec=None, cond_text=None):
        t = self.noise_levels[steps].detach()
        h, w = xt.shape[2:]
        
        # 1. PRIMARY ENHANCEMENT (Replicates your old 'use_cond' logic)
        # Instead of using img_cond_encoder, we concatenate raw observations.
        # This prevents the information bottleneck for Q, U, and Density.
        if cond_img is not None:
            # Result: [B, 8, H, W] if xt is 4ch and cond_img is 4ch
            unet_input = torch.cat([xt, cond_img], dim=1) 
        else:
            unet_input = xt
            
        # 2. MULTIMODAL GUIDANCE (For non-image or secondary data)
        cond_list = [unet_input]
        if self.use_spectrum and cond_spec is not None:
            cond_list.append(self.spec_encoder(cond_spec))
        if self.use_text and cond_text is not None:
            cond_list.append(self.text_encoder(cond_text, h, w))
            
        final_input = torch.cat(cond_list, dim=1) if len(cond_list) > 1 else unet_input
        
        # 3. DENOISE AND PROJECT
        # diffusion_model outputs latent_dim (4 channels)
        latent_out = self.diffusion_model(final_input, t)
        
        # projection_head maps 4 channels -> 1 channel + Sigmoid [0, 1]
        return self.projection_head(latent_out)
    
    def apply_target_uplift(self, y):
        return self.target_uplift(y)

# core/network.py

def get_model(cfg, train_loader, log):
    """
    Multimodal Factory function.
    Updated to unpack 5 items from the new UniversalAstroDataset.
    """
    # Use a dummy variable (_) for the mask and index
    sample_x_prim, sample_x_cond, sample_y, _, _ = next(iter(train_loader))
    
    in_channels = sample_x_prim.shape[2] 
    out_channels = sample_y.shape[2]
    
    cfg.model.out_channels = out_channels 
    num_steps = cfg.model.get('intervals', 1000)
    noise_levels = torch.linspace(0.001, 1.0, num_steps)
    
    model = AstroMultimodalUNet(cfg, in_channels, out_channels, noise_levels)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[Net] Multimodal AstroUNet Initialized")
    log.info(f"      Bridge In: {in_channels} | Target Out: {out_channels}")
    log.info(f"      Total Trainable Params: {total_params:,}")
    
    return model


class SpectralEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Compresses 64 channels -> bottleneck of 1 or 2 channels
        # Keeps spatial resolution identical to Dust/Pol data
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, out_channels, kernel_size=1) 
        )
    def forward(self, x):
        return self.net(x)

class TextEncoder(nn.Module):
    def __init__(self, in_dim, out_channels):
        super().__init__()
        # Distills 768-dim global context into 1 or 2 spatial-ready channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_channels)
        )
    def forward(self, x, h, w):
        feat = self.net(x) # [B, out_channels]
        return feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)