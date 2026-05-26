import torch
from engine.trainer_dsb import AstroDSBTrainer
from engine.trainer_ddpm import AstroDDPMTrainer

import torch

def get_model(cfg, train_loader, log):
    """Factory function to select and initialize the neural backbone."""
    model_type = cfg.model.get('type', 'dsb').lower()
    
    # 1. Authoritative Source: YAML Keywords
    in_channels = cfg.model.get('in_channels', 3)
    out_channels = cfg.model.get('out_channels', 2)
    
    if model_type == 'ddpm':
        from core.network_ddpm import AstroMultimodalDDPM
        if log: log.info(f"[Factory] Initializing DDPM Backbone (Noise Prediction)")
        # If YAML says 3 and 2, this class will internally create a 5-channel UNet
        model = AstroMultimodalDDPM(cfg, in_channels, out_channels)
    else:
        from core.network_dsb import AstroMultimodalUNet
        if log: log.info(f"[Factory] Initializing DSB Backbone (Bridge Target Prediction)")
        num_steps = cfg.model.get('intervals', 1000)
        noise_levels = torch.linspace(0.001, 1.0, num_steps)
        model = AstroMultimodalUNet(cfg, in_channels, out_channels, noise_levels)
    
    if log:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"      YAML Channels: In {in_channels} | Out {out_channels}")
        log.info(f"      Backbone Input: {in_channels + out_channels if model_type=='ddpm' else in_channels}")
        log.info(f"      Total Trainable Params: {total_params:,}")
        
    return model


def get_trainer(cfg, model, diffusion, train_loader, val_loader, log):
    """Factory function to select the correct training loop."""
    model_type = cfg.model.get('type', 'dsb').lower()
    
    if model_type == 'ddpm':
        return AstroDDPMTrainer(cfg, model, diffusion, train_loader, val_loader, log)
    else:
        return AstroDSBTrainer(cfg, model, diffusion, train_loader, val_loader, log)