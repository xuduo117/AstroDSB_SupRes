import torch
import numpy as np
from tqdm import tqdm

class AstroPredictor:
    def __init__(self, cfg, model, diffusion, device):
        self.cfg = cfg
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.model.eval()
        
        # Detect model type for branching logic
        self.model_type = cfg.model.get('type', 'dsb').lower()

    @torch.no_grad()
    def predict_batch(self, x_batch, cond_batch=None, cond_spec=None, cond_text=None, **kwargs):
        """
        Multimodal Batch inference. Branches logic based on DSB vs DDPM.
        """
        # 1. Flatten patches if [B, N, C, H, W] -> [B*N, C, H, W]
        if len(x_batch.shape) == 5:
            x_batch = x_batch.flatten(0, 1)
            if cond_batch is not None: cond_batch = cond_batch.flatten(0, 1)
            if cond_spec is not None: cond_spec = cond_spec.flatten(0, 1)
            if cond_text is not None and len(cond_text.shape) == 3: 
                cond_text = cond_text.flatten(0, 1)

        x_batch = x_batch.to(self.device)
        c_img = cond_batch.to(self.device) if cond_batch is not None else None
        c_spec = cond_spec.to(self.device) if cond_spec is not None else None
        c_text = cond_text.to(self.device) if cond_text is not None else None

        # 2. Backbone-Specific Inference
        if self.model_type == 'ddpm':
            # --- DDPM PATHWAY ---
            # Initial state is pure Gaussian Noise
            shape = (x_batch.shape[0], self.cfg.model.out_ch, 
                     self.cfg.dataset.patch_size, self.cfg.dataset.patch_size)
            
            # Use the kwargs (steps, eta) passed from predict.py
            prediction = self.diffusion.p_sample_loop(
                self.model,
                shape=shape,
                cond_img=x_batch, # Observations are passed as conditions
                cond_spec=c_spec,
                cond_text=c_text
            )
        else:
            # --- DSB PATHWAY ---
            # Project Primary Observable to Bridge Dimension (Target Space)
            model_engine = self.model.module if hasattr(self.model, 'module') else self.model
            x_start = model_engine.primary_proj(x_batch)

            # Retrieve sampling speed (skip_steps)
            s_steps = kwargs.get('skip_steps') or \
                      self.cfg.get('inference', {}).get('sampling_skip_steps', 1)
            
            prediction = self.diffusion.ddpm_sampling(
                self.model, 
                x_start,          # DSB starts from the data
                cond_img=c_img,   # Ancillary conditioning
                cond_spec=c_spec,
                cond_text=c_text,
                verbose=False,
                skip_steps=s_steps
            )
            
        return prediction

    def predict_large_snapshot(self, single_img, patch_size=128, stride=32, 
                               spec_cube=None, text_emb=None, **kwargs):
        """Processes one large image using a sliding window approach."""
        c_in, h, w = single_img.shape
        c_out = self.cfg.model.get('out_ch', 1) 
        
        merged_out = torch.zeros((c_out, h, w))
        count_map = torch.zeros((1, h, w))
        
        # Primary Observables (used as bridge start in DSB or priors in DDPM)
        p_idx = self.cfg.dataset.get('primary_indices', [0, 1, 2, 3])
        # Ancillary Conditioning (used in forward pass of the model)
        c_idx = self.cfg.dataset.get('cond_indices', [])
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                p_patch = single_img[p_idx, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                
                c_img_patch = None
                if len(c_idx) > 0:
                    c_img_patch = single_img[c_idx, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                
                c_spec_patch = None
                if spec_cube is not None:
                    c_spec_patch = spec_cube[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                
                pred_patch = self.predict_batch(
                    p_patch, 
                    cond_batch=c_img_patch, 
                    cond_spec=c_spec_patch, 
                    cond_text=text_emb, 
                    **kwargs
                )
                
                merged_out[:, y:y+patch_size, x:x+patch_size] += pred_patch.squeeze(0).cpu()
                count_map[:, y:y+patch_size, x:x+patch_size] += 1.0
        
        return merged_out / count_map.clamp(min=1.0)

    def predict_stacked_snapshots(self, stack, patch_size=128, stride=32, 
                                  spec_stack=None, text_stack=None, **kwargs):
        """Processes a stack of observations (e.g. from multiple simulation snapshots)."""
        n_samples = stack.shape[0]
        results = []
        
        for i in range(n_samples):
            s_cube = spec_stack[i] if spec_stack is not None else None
            t_emb = text_stack[i] if text_stack is not None else None
            
            sample_pred = self.predict_large_snapshot(
                stack[i], 
                patch_size=patch_size, 
                stride=stride, 
                spec_cube=s_cube,
                text_emb=t_emb,
                **kwargs
            )
            results.append(sample_pred)
            
        return torch.stack(results)