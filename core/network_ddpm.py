import math
import torch
import torch.nn as nn
from .encoders import SpectrumEncoder, TextEncoder

# ---------------------------------------------------------
# 1. DATA DOMAIN TRANSFORMS (The Fix for DDPM Snow)
# ---------------------------------------------------------
def data_transform(X):
    """Converts [0, 1] data to [-1, 1] to match the N(0,1) diffusion prior."""
    return 2.0 * X - 1.0

def inverse_data_transform(X):
    """Converts [-1, 1] DDPM predictions back to physical [0, 1] space."""
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# ---------------------------------------------------------
# 2. TIMESTEP & UTILITY FUNCTIONS
# ---------------------------------------------------------

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# ---------------------------------------------------------
# 3. BUILDING BLOCKS (Unchanged)
# ---------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k) * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h*w)
        h_ = torch.bmm(v, w_.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h_)

# ---------------------------------------------------------
# 4. THE BACKBONE (DDPM UNET)
# ---------------------------------------------------------

class DDPMBackbone(nn.Module):
    def __init__(self, cfg, in_channels, out_ch):
        super().__init__()
        self.ch = cfg.model.ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(cfg.model.ch_mult)
        self.num_res_blocks = cfg.model.num_res_blocks
        self.resolution = cfg.dataset.patch_size
        
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)

        in_ch_mult = (1,) + tuple(cfg.model.ch_mult)
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * cfg.model.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=cfg.model.dropout))
                block_in = block_out
                if self.resolution // (2**i_level) in cfg.model.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block, down.attn = block, attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, cfg.model.resamp_with_conv)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=cfg.model.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=cfg.model.dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_out = self.ch * cfg.model.ch_mult[i_level]
            skip_in = self.ch * cfg.model.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=cfg.model.dropout))
                block_in = block_out
                if self.resolution // (2**i_level) in cfg.model.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block, up.attn = block, attn
            if i_level != 0:
                up.upsample = Upsample(block_in, cfg.model.resamp_with_conv)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)

    def forward(self, x, t):
        temb = self.temb.dense[1](nonlinearity(self.temb.dense[0](get_timestep_embedding(t, self.ch))))
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h, temb)), temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))

# ---------------------------------------------------------
# 5. MULTIMODAL WRAPPPER (Astro DDPM)
# ---------------------------------------------------------

class AstroMultimodalDDPM(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super().__init__()
        self.cfg = cfg
        # Input to UNet = Observation (in_channels) + Noisy Latent (out_channels)
        self.unet_in_channels = in_channels + out_channels
        self.backbone = DDPMBackbone(cfg, self.unet_in_channels, out_channels)
        
        self.use_spectrum = cfg.model.get('use_spectrum', False)
        self.use_text = cfg.model.get('use_text', False)
        if self.use_spectrum: self.spec_encoder = SpectrumEncoder(cfg)
        if self.use_text: self.text_encoder = TextEncoder(cfg)

    def forward(self, xt, steps, cond_img=None, cond_spec=None, cond_text=None):
        unet_input = torch.cat([xt, cond_img], dim=1) if cond_img is not None else xt
        
        cond_list = [unet_input]
        if self.use_spectrum and cond_spec is not None:
            cond_list.append(self.spec_encoder(cond_spec))
        if self.use_text and cond_text is not None:
            cond_list.append(self.text_encoder(cond_text, xt.shape[2], xt.shape[3]))
            
        final_input = torch.cat(cond_list, dim=1) if len(cond_list) > 1 else unet_input
        return self.backbone(final_input, steps)