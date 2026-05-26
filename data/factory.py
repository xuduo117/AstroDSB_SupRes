import os
import glob
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# --- UPDATED: Import the new wrapper dataset ---
from .datasets import PlanckACTDataset, CombinedTransferDataset 

def scan_and_split_files(root_dirs, split_mode='physical_ood', ood_keywords=None, test_ratio=0.01, seed=42):
    if ood_keywords is None:
        # Default fallback keyword, adjust in your YAML config!
        ood_keywords = ['M2e5'] 
        
    file_groups = {}
    
    for d in root_dirs:
        # 1. Search for the new analytical 850um dust maps
        files = glob.glob(os.path.join(d, "**/*850um_tensor.npy"), recursive=True)
        
        for f in files:
            # 2. Extract simulation physics from the folder name 
            # (e.g., starforge_v1.2_M2e4_R10_Z1_S0_A2_B0.01_I1_Res271_n2_sol0.5_42)
            sim_name = os.path.basename(os.path.dirname(f)) 
            
            # Extract snapshot info
            filename = os.path.basename(f)
            
            # Group by physics + snapshot
            unique_id = f"{sim_name}_{filename}"

            if unique_id not in file_groups:
                file_groups[unique_id] = []
            
            file_groups[unique_id].append(f)

    train_files = []
    test_files = []
    
    if split_mode == 'physical_ood':
        print(f"[Split] Mode: PHYSICAL OOD. Holding out keywords: {ood_keywords}")
        for unique_id, paths in file_groups.items():
            # If ANY of the OOD keywords are in the simulation folder name, it goes to test
            if any(k in unique_id for k in ood_keywords):
                test_files.extend(paths)
            else:
                train_files.extend(paths)
    else:
        # Standard random split fallback
        print(f"[Split] Mode: RANDOM SNAPSHOT")
        all_ids = sorted(list(file_groups.keys()))
        random.seed(seed)
        random.shuffle(all_ids)
        n_test = max(1, int(len(all_ids) * test_ratio)) if test_ratio > 0 else 0
        
        for snap in all_ids[n_test:]: train_files.extend(file_groups[snap])
        for snap in all_ids[:n_test]: test_files.extend(file_groups[snap])

    print(f"[Split] Summary:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Test:  {len(test_files)} files (OOD Samples)")
    
    return train_files, test_files

def get_dataloaders(cfg, log):
    # Ensure cfg.dataset.roots is a list of directories in your YAML
    root_dirs = cfg.dataset.roots 
    split_mode = cfg.dataset.get('split_mode', 'physical_ood')
    
    # You can pass things like ['B0.01'] or ['M2e4'] in your config to test physical generalization!
    ood_keywords = cfg.dataset.get('ood_keywords', ['I100'])

    train_files, val_files = scan_and_split_files(
        root_dirs, 
        split_mode=split_mode,
        ood_keywords=ood_keywords,
        seed=cfg.experiment.seed
    )

    # =========================================================
    # THE TRANSFER LEARNING TOGGLE
    # =========================================================
    # Fetch the fraction from config, default to 0.0 (pure synthetic)
    # real_fraction = cfg.dataset.get('real_fraction', 0.0)
    real_planck_fraction = cfg.dataset.get('real_planck_fraction', 0.0)
    real_act_fraction = cfg.dataset.get('real_act_fraction', 0.0)
    
    if real_planck_fraction > 0.0 or real_act_fraction > 0:
        if getattr(cfg, 'global_rank', 0) == 0:
            log.info(f"🔄 TRANSFER LEARNING ENABLED: Routing {real_planck_fraction*100}% of batches to Real Planck Data!")
            log.info(f"🔄 TRANSFER LEARNING ENABLED: Routing {real_act_fraction*100}% of batches to Real ACT Data!")
        
        train_dataset = CombinedTransferDataset(cfg, train_files, mode='train', real_planck_fraction=real_planck_fraction, real_act_fraction=real_act_fraction)
        val_dataset = CombinedTransferDataset(cfg, val_files, mode='val', real_planck_fraction=real_planck_fraction, real_act_fraction=real_act_fraction)
    else:
        if getattr(cfg, 'global_rank', 0) == 0:
            log.info("🧪 SYNTHETIC BASELINE MODE: Using 100% Starforge Simulations.")
            
        train_dataset = PlanckACTDataset(cfg, train_files, mode='train')
        val_dataset = PlanckACTDataset(cfg, val_files, mode='val')
    # =========================================================

    is_dist = cfg.training.get('distributed', False) and (int(os.environ.get("WORLD_SIZE", 1)) > 1)

    if is_dist:
        train_sampler = DistributedSampler(train_dataset, num_replicas=cfg.global_size, rank=cfg.global_rank, shuffle=True)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=train_shuffle, 
                              sampler=train_sampler, num_workers=cfg.dataset.get('num_workers', 8), 
                              pin_memory=True, drop_last=True, persistent_workers=True)

    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, 
                            num_workers=cfg.dataset.get('num_workers', 4), pin_memory=True)

    return train_loader, val_loader