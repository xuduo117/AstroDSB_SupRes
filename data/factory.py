import os
import glob
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import datasets
from .datasets import SyntheticAstroDataset, SkirtAstroDataset, CombinedSyntheticDataset

# =====================================================================
# THE FIX: Kill NumPy internal multithreading to prevent CPU Thrashing
# =====================================================================
def worker_init_fn(worker_id):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Ensure true randomness across workers
    import numpy as np
    np.random.seed((torch.initial_seed() + worker_id) % (2**32 - 1))
    random.seed((torch.initial_seed() + worker_id) % (2**32 - 1))



def scan_and_split_files(root_dirs, pattern="**/*850um_tensor.npy", split_mode='physical_ood', ood_keywords=None, test_ratio=0.01, seed=42):
    if ood_keywords is None:
        ood_keywords = ['M2e5'] 
        
    file_groups = {}
    
    for d in root_dirs:
        files = glob.glob(os.path.join(d, pattern), recursive=True)
        
        for f in files:
            # We combine the folder name AND the filename to guarantee we catch OOD keywords 
            # even if the SKIRT files are all pooled in one 'Observational_Galleries' folder.
            unique_id = f"{os.path.basename(os.path.dirname(f))}_{os.path.basename(f)}"

            if unique_id not in file_groups:
                file_groups[unique_id] = []
            
            file_groups[unique_id].append(f)

    train_files, test_files = [], []
    
    if split_mode == 'physical_ood':
        print(f"[Split] Mode: PHYSICAL OOD. Holding out keywords: {ood_keywords}")
        for unique_id, paths in file_groups.items():
            if any(k in unique_id for k in ood_keywords):
                test_files.extend(paths)
            else:
                train_files.extend(paths)
    else:
        print(f"[Split] Mode: RANDOM SNAPSHOT")
        all_ids = sorted(list(file_groups.keys()))
        random.seed(seed)
        random.shuffle(all_ids)
        n_test = max(1, int(len(all_ids) * test_ratio)) if test_ratio > 0 else 0
        
        for snap in all_ids[n_test:]: train_files.extend(file_groups[snap])
        for snap in all_ids[:n_test]: test_files.extend(file_groups[snap])

    print(f"  Train: {len(train_files)} files | Test: {len(test_files)} files")
    return train_files, test_files

def get_dataloaders(cfg, log):
    split_mode = cfg.dataset.get('split_mode', 'physical_ood')
    ood_keywords = cfg.dataset.get('ood_keywords', ['I100'])
    seed = cfg.experiment.seed
    skirt_fraction = cfg.dataset.get('skirt_fraction', 0.25)

    # 1. Load Raw Dust Data (850um)
    print("\n--- Scanning Raw Dust Data ---")
    dust_train, dust_val = scan_and_split_files(
        cfg.dataset.roots, pattern="**/*850um_tensor.npy", split_mode=split_mode, ood_keywords=ood_keywords, seed=seed
    )
    dust_train_ds = SyntheticAstroDataset(cfg, dust_train, mode='train')
    dust_val_ds = SyntheticAstroDataset(cfg, dust_val, mode='val')

    # 2. Load SKIRT Multiband Data
    skirt_roots = cfg.dataset.get('skirt_roots', ['/scratch/xuduo117/project_SKIRT/Observational_Galleries/'])
    print("\n--- Scanning SKIRT Synthetic Data ---")
    skirt_train, skirt_val = scan_and_split_files(
        skirt_roots, pattern="**/*_gallery.npy", split_mode=split_mode, ood_keywords=ood_keywords, seed=seed
    )
    skirt_train_ds = SkirtAstroDataset(cfg, skirt_train, mode='train')
    skirt_val_ds = SkirtAstroDataset(cfg, skirt_val, mode='val')

    # 3. Combine Them (75% / 25%)
    if getattr(cfg, 'global_rank', 0) == 0:
        log.info(f"🧪 MIXED SYNTHETIC MODE: Combining {1-skirt_fraction:.0%} Dust and {skirt_fraction:.0%} SKIRT.")

    train_dataset = CombinedSyntheticDataset(dust_train_ds, skirt_train_ds, skirt_fraction=skirt_fraction)
    val_dataset = CombinedSyntheticDataset(dust_val_ds, skirt_val_ds, skirt_fraction=skirt_fraction)

    is_dist = cfg.training.get('distributed', False) and (int(os.environ.get("WORLD_SIZE", 1)) > 1)

    if is_dist:
        train_sampler = DistributedSampler(train_dataset, num_replicas=cfg.global_size, rank=cfg.global_rank, shuffle=True)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    # ---> ADDED PREFETCH_FACTOR AND WORKER_INIT_FN <---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=train_shuffle, 
        sampler=train_sampler, 
        num_workers=cfg.dataset.get('num_workers', 16), 
        pin_memory=True, 
        drop_last=True, 
        persistent_workers=cfg.dataset.get('persistent_workers', True),
        prefetch_factor=cfg.dataset.get('prefetch_factor', 4),
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=False, 
        num_workers=cfg.dataset.get('num_workers', 4), 
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader