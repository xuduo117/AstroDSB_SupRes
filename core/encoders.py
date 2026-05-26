import torch
import torch.nn as nn

class SpectrumEncoder(nn.Module):
    """Encodes spectral line data (1D or 3D cubes) into a latent representation."""
    def __init__(self, cfg):
        super().__init__()
        # Example: Simple MLP or 1D-CNN to compress spectrum
        in_dim = cfg.model.get('cond_dim_spec', 1)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        # Shape: [B, C, H, W] or [B, C]
        return self.encoder(x)

class TextEncoder(nn.Module):
    """Encodes global context strings (e.g., 'High metallicity nebula')."""
    def __init__(self, cfg):
        super().__init__()
        # Placeholder for a small transformer or frozen CLIP/BERT
        self.embed_dim = cfg.model.get('cond_dim_text', 768)
        self.dummy_param = nn.Parameter(torch.zeros(1)) 

    def forward(self, text_input, h, w):
        # Returns a spatial map [B, embed_dim, H, W] by broadcasting global context
        batch_size = text_input.shape[0]
        device = text_input.device
        # For now, return a zero-filled context map
        return torch.zeros(batch_size, self.embed_dim, h, w, device=device)