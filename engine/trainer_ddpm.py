import os
import torch
import torch.nn.functional as F
import shutil
import glob
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from utils.visualizer import save_snapshot

# =========================================================
# ASYMMETRIC ARCSINH PHYSICAL UNIT RECOVERY
# =========================================================
def unnormalize_prediction(pred_tensor, memory_tensor):
    """Reverses the [0, 1] Asymmetric Arcsinh prediction back into physical units."""
    c = memory_tensor[:, 0].view(-1, 1, 1, 1)
    alpha = memory_tensor[:, 1].view(-1, 1, 1, 1)
    t_min = memory_tensor[:, 2].view(-1, 1, 1, 1)
    t_max = memory_tensor[:, 3].view(-1, 1, 1, 1)
    
    # 1. Reverse the [0, 1] squeeze back to the [t_min, t_max] arcsinh space
    tf_data = pred_tensor * (t_max - t_min) + t_min
    
    # 2. Reverse Arcsinh to true physical units
    pred_physical = torch.sinh(tf_data) * alpha + c
    return pred_physical

class AstroDDPMTrainer:
    def __init__(self, cfg, model, diffusion, train_loader, val_loader, log):
        self.cfg = cfg
        self.log = log
        
        local_rank = getattr(cfg, 'local_rank', 0)
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        
        if getattr(cfg, 'global_size', 1) > 1:
            self.model = DDP(
                self.model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                gradient_as_bucket_view=True 
            )
        
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = AdamW(self.model.parameters(), lr=float(cfg.model.learning_rate))
        
        total_epochs = cfg.training.get('epochs', 1000)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_epochs, 
            eta_min=float(cfg.model.learning_rate) * 0.01
        )
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=cfg.model.ema_decay)
        self.start_epoch = 0
        self.best_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        obs_ch = self.cfg.model.get('in_channels', 1)
        target_ch = self.cfg.model.get('out_channels', 1)
        
        snr_threshold = self.cfg.model.get('snr_threshold', 5.0)
        bg_weight = self.cfg.model.get('background_loss_weight', 0.02)

        art_cfg = self.cfg.dataset.get('artifacts', {})
        use_mask_now = art_cfg.get('mask_loss', False) and (epoch >= art_cfg.get('activation_epoch', 0))
        
        data_iter = iter(self.train_loader)
        is_main_process = (getattr(self.cfg, 'global_rank', 0) == 0)
        pbar = tqdm(data_iter, desc=f"Epoch {epoch}") if is_main_process else data_iter
        
        for i, (x_prim_raw, x_cond_raw, y_raw, mask_raw, memory_raw, _) in enumerate(pbar):            
            
            if x_cond_raw.dim() == 5:
                x_obs = x_cond_raw[:, :, :obs_ch, ...].flatten(0, 1).to(self.device, non_blocking=True)   
                x_target = y_raw[:, :, :target_ch, ...].flatten(0, 1).to(self.device, non_blocking=True)    
                mask_data = mask_raw[:, :, :target_ch, ...].flatten(0, 1).to(self.device, non_blocking=True)      
            else:
                x_obs = x_cond_raw[:, :obs_ch, ...].to(self.device, non_blocking=True)   
                x_target = y_raw[:, :target_ch, ...].to(self.device, non_blocking=True)    
                mask_data = mask_raw[:, :target_ch, ...].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Operating directly in [0, 1] space as requested
            t = torch.randint(0, self.cfg.model.intervals, (x_target.shape[0],), device=self.device)
            noise = torch.randn_like(x_target)
            xt = self.diffusion.q_sample(x_target, t, noise)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                model_pred = self.model(xt, t, cond_img=x_obs)

                use_min_snr = self.cfg.model.get('use_min_snr', True)
                if use_min_snr:
                    alphas_cumprod_t = self.diffusion.alphas_cumprod.to(self.device)[t]
                    snr = (alphas_cumprod_t / (1 - alphas_cumprod_t)).clamp(min=1e-8)
                    snr_weight = torch.clamp(snr, max=snr_threshold) / snr
                    snr_weight = snr_weight[:, None, None, None]
                else:
                    snr_weight = 1.0

                objective = self.cfg.model.get('objective', 'eps')
                if objective == 'x0': target = x_target
                else: target = noise

                # ==========================================
                # INTENSITY-AWARE SPATIAL WEIGHTING
                # ==========================================
                # x_target is natively in [0, 1]. Square it to emphasize bright structures.
                # Background noise gets a base weight of 0.1 so the model doesn't completely ignore it.
                spatial_weight = 0.1 + 0.9 * (x_target ** 2.0)
                
                # Combine temporal SNR weighting with the physical spatial heatmap
                combined_weight = snr_weight * spatial_weight

               
                # Standard DDPM MSE Loss + Intensity-Aware Weighting
                raw_mse = F.mse_loss(model_pred, target, reduction='none') * combined_weight
                
                if use_mask_now:
                    inv_mask = 1.0 - mask_data
                    loss_internal = (raw_mse * mask_data).sum() / (mask_data.sum() + 1e-8)
                    loss_background = (raw_mse * inv_mask).sum() / (inv_mask.sum() + 1e-8)
                    loss = loss_internal + bg_weight * loss_background
                else:
                    loss = raw_mse.mean()

            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()
            
            total_loss += loss.item()
            
            if is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                iter_freq = self.cfg.training.get('snapshot_iter_freq', 5000)
                if i > 0 and i % iter_freq == 0:
                    self.validate(epoch, iteration=i)
                    self.save_checkpoint(epoch, total_loss / (i+1), iteration=i)

        self.scheduler.step()
        
        if is_main_process:
            pbar.close()
            avg_loss = total_loss / max(1, len(self.train_loader))
            self.validate(epoch)
            self.save_checkpoint(epoch, avg_loss)
            
            archive_int = self.cfg.training.get('archive_interval', 100)
            if epoch > 0 and epoch % archive_int == 0:
                self._archive_checkpoint(epoch)
            self._cleanup_snapshots(epoch)
            
            lr = self.scheduler.get_last_lr()[0]
            self.log.info(f"Summary Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")

        del pbar, data_iter
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self, epoch, iteration=None):
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        self.model.eval()
        
        x_prim_raw, x_cond_raw, y_raw, mask_raw, memory_raw, _ = next(iter(self.val_loader))
        
        if x_cond_raw.dim() == 5:
            x_obs = x_cond_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
            y_gt = y_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
            mask_v = mask_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
            memory = memory_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
        else:
            x_obs = x_cond_raw[0:1].to(self.device, non_blocking=True)
            y_gt = y_raw[0:1].to(self.device, non_blocking=True)
            mask_v = mask_raw[0:1].to(self.device, non_blocking=True)
            memory = memory_raw[0:1].to(self.device, non_blocking=True)
        
        out_ch = self.cfg.model.get('out_channels', self.cfg.model.get('out_ch', 1))
        shape = (x_obs.shape[0], out_ch, x_obs.shape[2], x_obs.shape[3])
        
        # Validation remains perfectly in [0, 1] space
        if iteration is not None:
            prediction = self.diffusion.solve_sde(
                self.model, shape=shape, cond_img=x_obs, 
                skip_steps=50, 
                eta=0.0  # <--- DETERMINISTIC SAMPLING
            )
            prefix = f"val_iter_{iteration}"
        else:
            prediction = self.diffusion.solve_ancestral(
                self.model, shape=shape, cond_img=x_obs
            )
            prefix = "val_final"
            
        if getattr(self.cfg, 'global_rank', 0) == 0:
            pred_phys = unnormalize_prediction(prediction, memory)
            y_gt_phys = unnormalize_prediction(y_gt, memory)
            x_obs_phys = unnormalize_prediction(x_obs, memory)
            
            save_snapshot(epoch, x_obs_phys, pred_phys, y_gt_phys, mask_v, self.cfg.log_dir, prefix=prefix)
            
        self.ema.restore(self.model.parameters())

    def _archive_checkpoint(self, epoch):
        src = os.path.join(self.cfg.log_dir, "latest.pt")
        dst = os.path.join(self.cfg.log_dir, f"checkpoint_milestone_ep{epoch}.pt")
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    def _cleanup_snapshots(self, epoch):
        snapshot_dir = os.path.join(self.cfg.log_dir, "snapshots")
        pattern = os.path.join(snapshot_dir, f"val_iter_*_epoch_{epoch:03d}.png")
        
        intermediate_files = glob.glob(pattern)
        for f in intermediate_files:
            try: os.remove(f)
            except Exception: pass

    def save_checkpoint(self, epoch, loss, iteration=None):
        if getattr(self.cfg, 'global_rank', 0) == 0:
            state = {
                'epoch': epoch,
                'iteration': iteration,
                'model_state': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'ema_state': self.ema.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'scaler_state': self.scaler.state_dict(), 
                'loss': loss,
                'cfg': self.cfg
            }
            torch.save(state, os.path.join(self.cfg.log_dir, "latest.pt"))
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(state, os.path.join(self.cfg.log_dir, "best_model.pt"))

    def load_checkpoint(self):
        ckpt_path = self.cfg.experiment.get('checkpoint_path')
        if self.cfg.experiment.get('resume') and ckpt_path and os.path.exists(ckpt_path):
            self.log.info(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            m = self.model.module if hasattr(self.model, 'module') else self.model
            m.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.ema.load_state_dict(ckpt['ema_state'])
            
            if 'scheduler_state' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
            if 'scaler_state' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler_state'])
                
            self.start_epoch = ckpt['epoch'] + 1