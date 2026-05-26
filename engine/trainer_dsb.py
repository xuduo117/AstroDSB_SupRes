import os
import gc
import torch
import shutil
import glob
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from utils.visualizer import save_snapshot

# =========================================================
# NEW: PHYSICAL UNIT RECOVERY
# =========================================================
# def unnormalize_prediction(pred_tensor, memory_tensor):
#     i_median = memory_tensor[:, 0].view(-1, 1, 1, 1)
#     i_std = memory_tensor[:, 1].view(-1, 1, 1, 1)
#     tf_min = memory_tensor[:, 2].view(-1, 1, 1, 1)
#     N_scale = memory_tensor[:, 3].view(-1, 1, 1, 1)
#     tf_data = ((pred_tensor - 0.1) / 0.8) * N_scale + tf_min
#     return torch.sinh(tf_data) * i_std + i_median

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

class AstroDSBTrainer:
    def __init__(self, cfg, model, diffusion, train_loader, val_loader, log):
        self.cfg = cfg
        self.log = log
        
        local_rank = getattr(cfg, 'local_rank', 0)
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        if cfg.training.get('distributed', False):
            self.model = DDP(
                self.model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True 
            )
        
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = AdamW(self.model.parameters(), lr=float(cfg.model.learning_rate))
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        
        total_epochs = cfg.training.get('epochs', 2000)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_epochs, 
            eta_min=float(cfg.model.learning_rate) * 0.01
        )
        
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=cfg.model.ema_decay)
        self.start_epoch = 0
        self.best_loss = float('inf')

    def train_epoch(self, epoch):
        torch.cuda.empty_cache()
        gc.collect()
        
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        is_main_process = (getattr(self.cfg, 'global_rank', 0) == 0)
        data_iter = iter(self.train_loader)
        pbar = tqdm(data_iter, desc=f"Epoch {epoch}") if is_main_process else data_iter
        
        art_cfg = self.cfg.dataset.get('artifacts', {})
        activation_limit = art_cfg.get('activation_epoch', 0)
        
        if hasattr(self.train_loader.dataset, 'artifacts_active'):
            self.train_loader.dataset.artifacts_active = (epoch >= activation_limit)
            
        use_mask_now = art_cfg.get('mask_loss', False) and (epoch >= activation_limit)

        # ==========================================
        # FIX 1: Catch memory_tensor in the 6-element tuple
        # ==========================================
        for i, (x_prim_raw, x_cond_raw, y_raw, mask_raw, memory_raw, _) in enumerate(pbar):            
            
            if x_prim_raw.dim() == 5:
                x_prim = x_prim_raw.flatten(0, 1).to(self.device, non_blocking=True)
                y = y_raw.flatten(0, 1).to(self.device, non_blocking=True)
                mask_data = mask_raw.flatten(0, 1).to(self.device, non_blocking=True)
                x_cond = x_cond_raw.flatten(0, 1).to(self.device, non_blocking=True) if x_cond_raw.shape[2] > 0 else None
            else:
                x_prim = x_prim_raw.to(self.device, non_blocking=True)
                y = y_raw.to(self.device, non_blocking=True)
                mask_data = mask_raw.to(self.device, non_blocking=True)
                x_cond = x_cond_raw.to(self.device, non_blocking=True) if x_cond_raw is not None else None
            
            self.optimizer.zero_grad(set_to_none=True)
            model_engine = self.model.module if hasattr(self.model, 'module') else self.model
            
            with torch.no_grad():
                y_latent = model_engine.apply_target_uplift(y) if hasattr(model_engine, 'apply_target_uplift') else y
                x_start = x_prim 

            steps = torch.randint(0, self.cfg.model.intervals, (y.shape[0],), device=self.device)
            xt = self.diffusion.q_sample(steps, y_latent, x_start)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                pred = self.model(xt, steps, cond_img=x_cond)

                if use_mask_now:
                    loss_internal = F.mse_loss(pred * mask_data, y * mask_data, reduction='sum')
                    loss_internal = loss_internal / (mask_data.sum() + 1e-8) 

                    inv_mask = 1.0 - mask_data
                    loss_background = F.mse_loss(pred * inv_mask, y * inv_mask, reduction='sum')
                    loss_background = loss_background / (inv_mask.sum() + 1e-8)

                    loss = loss_internal + 0.1 * loss_background
                else:
                    loss = F.mse_loss(pred, y)

            loss_is_good = torch.tensor(1.0 if torch.isfinite(loss) else 0.0, device=self.device)
            if getattr(self.cfg.training, 'distributed', False) and torch.distributed.is_initialized():
                torch.distributed.all_reduce(loss_is_good, op=torch.distributed.ReduceOp.MIN)
            
            if loss_is_good.item() < 0.5:
                self.nan_strikes += 1
                if is_main_process:
                    self.log.info(f"⚠️ Warning: NaN loss detected (Strike {self.nan_strikes}/10). Skipping globally!")
                
                self.optimizer.zero_grad(set_to_none=True)
                del pred, loss, xt 
                torch.cuda.empty_cache()
                if self.nan_strikes >= 10: raise RuntimeError("❌ 10 consecutive NaNs detected! Stopping training.")
                continue
            
            self.nan_strikes = 0
            valid_batches += 1

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            grads_are_finite = True
            for param in self.model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    grads_are_finite = False
                    break
            
            if not grads_are_finite:
                if is_main_process:
                    self.log.info(f"⚠️ Warning: NaN/Inf GRADIENT detected at Epoch {epoch}, Batch {i}. Skipping step.")
            else:
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

            del x_prim, y, mask_data, xt, pred, loss
            if x_cond is not None: del x_cond
            if i % 100 == 0: torch.cuda.empty_cache()

        self.scheduler.step()

        if is_main_process:
            pbar.close()
            avg_loss = total_loss / max(1, valid_batches)
            
            self.validate(epoch)
            self.save_checkpoint(epoch, avg_loss)
            
            archive_int = self.cfg.training.get('archive_interval', 100)
            if epoch > 0 and epoch % archive_int == 0:
                self._archive_checkpoint(epoch)
            
            self._cleanup_snapshots(epoch)
            lr = self.scheduler.get_last_lr()[0]
            self.log.info(f"Summary Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")

        del pbar, data_iter
        torch.cuda.empty_cache() 
        return total_loss / max(1, valid_batches)

    @torch.no_grad()
    def validate(self, epoch, iteration=None):
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        self.model.eval()
        
        # ==========================================
        # FIX 2: Catch memory_tensor in Validation
        # ==========================================
        x_prim_raw, x_cond_raw, y_raw, mask_raw, memory_raw, _ = next(iter(self.val_loader))
        
        if x_prim_raw.dim() == 5:
            x_prim_in = x_prim_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
            y_gt = y_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
            mask_v = mask_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
            memory = memory_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True)
            x_cond_in = x_cond_raw[0:1].flatten(0, 1).to(self.device, non_blocking=True) if x_cond_raw.shape[2] > 0 else None
        else:
            x_prim_in = x_prim_raw[0:1].to(self.device, non_blocking=True)
            y_gt = y_raw[0:1].to(self.device, non_blocking=True)
            mask_v = mask_raw[0:1].to(self.device, non_blocking=True)
            memory = memory_raw[0:1].to(self.device, non_blocking=True)
            x_cond_in = x_cond_raw[0:1].to(self.device, non_blocking=True) if x_cond_raw is not None else None
        
        prediction = self.diffusion.ddpm_sampling(
            self.model, x_prim_in, cond_img=x_cond_in, verbose=False,
            skip_steps=self.cfg.training.get('snapshot_skip_steps', 50)
        )
        
        # ==========================================
        # FIX 3: Convert predictions to physical units
        # ==========================================
        if getattr(self.cfg, 'global_rank', 0) == 0:
            pred_phys = unnormalize_prediction(prediction, memory)
            y_gt_phys = unnormalize_prediction(y_gt, memory)
            x_prim_phys = unnormalize_prediction(x_prim_in, memory)
            
            prefix = "val_final" if iteration is None else f"val_iter_{iteration}"
            save_snapshot(epoch, x_prim_phys, pred_phys, y_gt_phys, mask_v, self.cfg.log_dir, prefix=prefix)

        self.ema.restore(self.model.parameters())
        del x_prim_in, y_gt, mask_v, x_cond_in, prediction, memory
        torch.cuda.empty_cache()

    def _archive_checkpoint(self, epoch):
        src = os.path.join(self.cfg.log_dir, "latest.pt")
        dst = os.path.join(self.cfg.log_dir, f"checkpoint_milestone_ep{epoch}.pt")
        if os.path.exists(src): shutil.copyfile(src, dst)

    def _cleanup_snapshots(self, epoch):
        snapshot_dir = os.path.join(self.cfg.log_dir, "snapshots")
        pattern = os.path.join(snapshot_dir, f"val_iter_*_epoch_{epoch:03d}.png")
        for f in glob.glob(pattern):
            try: os.remove(f)
            except: pass

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