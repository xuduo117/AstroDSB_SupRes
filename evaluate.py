import os
import yaml
import argparse
import random
import torch
import numpy as np
import h5py
from tqdm import tqdm
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage

# Modular Components
from data.factory import get_dataloaders
from core.factory import get_model
from core.diffusion import AstroDiffusionBridge, DDPMHandler
from utils.util import set_seed
from logger import Logger

def clean_state_dict(state_dict):
    """Removes the 'module.' prefix if the model was saved via DDP."""
    clean_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        clean_dict[new_key] = v
    return clean_dict

# ==========================================
# NEW: PHYSICAL UNIT RECOVERY
# ==========================================
def unnormalize_prediction(pred_tensor, memory_tensor):
    """
    Rebuilds true physical units from the U-Net prediction using universal arcsinh inversion.
    """
    i_median = memory_tensor[:, 0].view(-1, 1, 1, 1)
    i_std = memory_tensor[:, 1].view(-1, 1, 1, 1)
    tf_min = memory_tensor[:, 2].view(-1, 1, 1, 1)
    N_scale = memory_tensor[:, 3].view(-1, 1, 1, 1)

    # 1. Remove the 10% Headroom
    tf_data = ((pred_tensor - 0.1) / 0.8) * N_scale + tf_min

    # 2. Invert the arcsinh transform
    raw_physical_data = torch.sinh(tf_data) * i_std + i_median
    return raw_physical_data


# ==========================================================================
# DETERMINISTIC EVALUATION SET
#
# CombinedSyntheticDataset is a training SAMPLER, not an indexable dataset:
# its __getitem__ throws the requested index away and returns
# ds[random.randint(...)] from a randomly chosen component. Iterating it for
# evaluation draws len(dataset) samples WITH REPLACEMENT, so some held-out
# files are scored several times and others never, the run is not
# reproducible, and no index-to-file mapping can be written for it. It also
# has no .file_list, which is the AttributeError that exposed all of this.
#
# For evaluation we therefore unwrap it and visit every held-out file exactly
# once, in a fixed order.
# ==========================================================================
class DeterministicEvalDataset(Dataset):
    """Every held-out file of every component, once each, in a fixed order."""

    def __init__(self, parts):
        # parts: [(name, component_dataset), ...]
        self.parts = [(n, ds) for n, ds in parts
                      if ds is not None and len(ds) > 0]
        self.map = []        # global index -> (part index, local index)
        self.file_list = []  # global index -> filepath
        self.sources = []    # global index -> component name
        for pi, (name, ds) in enumerate(self.parts):
            for li in range(len(ds)):
                self.map.append((pi, li))
                self.file_list.append(ds.file_list[li])
                self.sources.append(name)

    def __len__(self):
        return len(self.map)

    def __getitem__(self, g):
        pi, li = self.map[g]
        item = list(self.parts[pi][1][li])
        item[-1] = g            # replace the component-local index with the
        return tuple(item)      # global one, so the CSV maps one to one


def unwrap_components(dataset):
    """Recover the component datasets behind whatever wrapper is in use."""
    # current CombinedSyntheticDataset: parallel .names / .datasets lists
    if hasattr(dataset, 'datasets') and hasattr(dataset, 'names'):
        return list(zip(dataset.names, dataset.datasets))
    # older two-component version
    if hasattr(dataset, 'dust_dataset') and hasattr(dataset, 'skirt_dataset'):
        return [('dust', dataset.dust_dataset), ('skirt', dataset.skirt_dataset)]
    if hasattr(dataset, 'synthetic_dataset'):
        return [('synthetic', dataset.synthetic_dataset)]
    if hasattr(dataset, 'file_list'):
        return [('all', dataset)]
    raise AttributeError(
        f'{type(dataset).__name__} exposes neither component datasets nor a '
        f'file_list; evaluate.py cannot build a deterministic evaluation set '
        f'from it.')


def eval_worker_init(worker_id):
    """Like the training worker_init_fn, but seeded reproducibly.

    data/factory.py seeds its workers from torch.initial_seed(), which changes
    between runs, so two evaluations of the same checkpoint would not agree
    wherever the val path still draws a random number.
    """
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    np.random.seed(20240910 + worker_id)
    random.seed(20240910 + worker_id)
    torch.manual_seed(20240910 + worker_id)


def evaluate(cfg):
    # 1. Setup Single-GPU Environment for Evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg.training.distributed = False
    cfg.local_rank = 0
    cfg.global_rank = 0
    cfg.global_size = 1

    log = Logger(None)
    log.info(f"Starting Full Dataset Evaluation on {device}")
    set_seed(cfg.experiment.seed)

    # 2. Setup Data, then rebuild the validation loader deterministically
    _, val_loader = get_dataloaders(cfg, log)

    components = unwrap_components(val_loader.dataset)
    eval_dataset = DeterministicEvalDataset(components)
    if len(eval_dataset) == 0:
        raise RuntimeError('The held-out set is empty. Check ood_keywords.')

    counts = {}
    for s in eval_dataset.sources:
        counts[s] = counts.get(s, 0) + 1
    log.info("Deterministic evaluation set: "
             + ", ".join(f"{n} {c} files" for n, c in counts.items())
             + f"  (total {len(eval_dataset)})")
    log.info(f"Replaced the {type(val_loader.dataset).__name__} sampler, which "
             f"would have drawn {len(val_loader.dataset)} samples at random "
             f"with replacement.")
    if hasattr(val_loader.dataset, 'names'):
        dropped = {'dust', 'skirt', 'fire'} - set(val_loader.dataset.names)
        if dropped:
            log.info(f"NOTE: {sorted(dropped)} were dropped by the mixture "
                     f"weights and are therefore absent from this evaluation. "
                     f"Give them a nonzero dataset_fractions entry if their "
                     f"held-out files should be scored.")

    val_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.dataset.get('eval_batch_size', cfg.dataset.batch_size),
        shuffle=False,
        num_workers=cfg.dataset.get('num_workers', 4),
        pin_memory=True,
        worker_init_fn=eval_worker_init,
    )

    # 3. Setup Model & Diffusion
    model = get_model(cfg, None, log).to(device)
    model.eval()

    model_type = cfg.model.get('type', 'dsb').lower()
    if model_type == 'ddpm':
        diffusion = DDPMHandler(cfg, device=device)
    else:
        diffusion = AstroDiffusionBridge(cfg, device=device)

    # Take the sampler resolution from the config instead of hard-coding it.
    # inference.sampling_skip_steps is 10, i.e. np.arange(1000)[::10] = 100
    # network evaluations; the 50 that used to be hard-coded here is 20, so
    # the I100 figure was being made at a fifth of the ACT figure's sampling.
    skip_steps = int(cfg.get('inference', {}).get('sampling_skip_steps', 10))
    log.info(f"Sampling with skip_steps={skip_steps} "
             f"({len(np.arange(1000)[::skip_steps])} network evaluations per patch)")

    # 4. Load Checkpoint and EMA Weights
    ckpt_path = os.path.join(cfg.experiment.base_log_dir, cfg.experiment.name, "latest.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

    log.info(f"Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'epoch' in ckpt:
        log.info(f"Checkpoint epoch: {ckpt['epoch']}")

    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(clean_state_dict(state_dict))

    if 'ema_state' in ckpt:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.get('ema_decay', 0.999))
        ema.load_state_dict(ckpt['ema_state'])
        ema.copy_to(model.parameters())
        log.info("Successfully applied EMA weights.")

    # 5. Output Directory Setup
    eval_dir = os.path.join(cfg.experiment.base_log_dir, cfg.experiment.name, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    obs_ch = cfg.model.get('cond_channels', cfg.model.get('in_channels', 1))
    target_ch = cfg.model.get('out_channels', 1)

    h5_path = os.path.join(eval_dir, f"{model_type}_predictions_full.h5")
    csv_path = os.path.join(eval_dir, f"{model_type}_file_mapping.csv")

    log.info(f"Saving file mapping to: {csv_path}")
    with open(csv_path, 'w') as f:
        f.write("File_Index,Source,Filepath\n")
        for idx, (src, path) in enumerate(zip(eval_dataset.sources,
                                              eval_dataset.file_list)):
            f.write(f"{idx},{src},{path}\n")

    src_codes = {n: i for i, n in enumerate(dict.fromkeys(eval_dataset.sources))}

    # 6. Evaluation Loop (Streaming to Disk)
    with h5py.File(h5_path, 'w') as h5f:
        dset_in = h5f.create_dataset('inputs', shape=(0, obs_ch, 256, 256), maxshape=(None, obs_ch, 256, 256), dtype='float32')
        dset_gt = h5f.create_dataset('ground_truth', shape=(0, target_ch, 256, 256), maxshape=(None, target_ch, 256, 256), dtype='float32')
        dset_pred = h5f.create_dataset('prediction', shape=(0, target_ch, 256, 256), maxshape=(None, target_ch, 256, 256), dtype='float32')
        dset_idx = h5f.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int64')
        # which patch of its file each row is, and which component it came
        # from, so dust and SKIRT can be separated downstream
        dset_patch = h5f.create_dataset('patch', shape=(0,), maxshape=(None,), dtype='int64')
        dset_src = h5f.create_dataset('source', shape=(0,), maxshape=(None,), dtype='int64')
        dset_src.attrs['codes'] = ",".join(f"{i}={n}" for n, i in src_codes.items())
        h5f.attrs['skip_steps'] = skip_steps
        h5f.attrs['checkpoint'] = ckpt_path

        ptr = 0

        with torch.no_grad():
            for i, (x_prim_raw, x_cond_raw, y_raw, mask_raw, memory_raw, index_raw) in enumerate(tqdm(val_loader, desc=f"Evaluating {model_type.upper()}")):

                if x_prim_raw.dim() == 5:
                    num_patches = x_prim_raw.shape[1]

                    x_prim = x_prim_raw.flatten(0, 1).to(device)
                    x_cond = x_cond_raw.flatten(0, 1).to(device)
                    y_gt = y_raw.flatten(0, 1).to(device)
                    memory = memory_raw.flatten(0, 1).to(device)

                    index = index_raw.repeat_interleave(num_patches).to(device)
                    patch = torch.arange(num_patches).repeat(index_raw.shape[0])
                else:
                    x_prim = x_prim_raw.to(device)
                    x_cond = x_cond_raw.to(device)
                    y_gt = y_raw.to(device)
                    memory = memory_raw.to(device)
                    index = index_raw.to(device)
                    patch = torch.zeros_like(index_raw)

                shape = (x_prim.shape[0], target_ch, x_prim.shape[2], x_prim.shape[3])

                # 1. Run the Model Sampler
                if model_type == 'ddpm':
                    prediction = diffusion.solve_sde(model, shape=shape, cond_img=x_cond, skip_steps=skip_steps)
                else:
                    prediction = diffusion.ddpm_sampling(model, x_prim, cond_img=x_cond, skip_steps=skip_steps, verbose=False)

                # ==========================================
                # NEW: Convert everything back to True Physics!
                # ==========================================
                x_cond_phys = unnormalize_prediction(x_cond, memory)
                y_gt_phys = unnormalize_prediction(y_gt, memory)
                pred_phys = unnormalize_prediction(prediction, memory)

                x_cond_np = x_cond_phys.cpu().numpy()
                y_gt_np = y_gt_phys.cpu().numpy()
                pred_np = pred_phys.cpu().numpy()

                batch_size = x_cond_np.shape[0]
                idx_np = index.cpu().numpy()
                src_np = np.array([src_codes[eval_dataset.sources[j]] for j in idx_np],
                                  dtype='int64')

                # Dynamically resize the H5 datasets
                for dset in (dset_in, dset_gt, dset_pred, dset_idx,
                             dset_patch, dset_src):
                    dset.resize(ptr + batch_size, axis=0)

                # Write data
                dset_in[ptr:ptr+batch_size] = x_cond_np
                dset_gt[ptr:ptr+batch_size] = y_gt_np
                dset_pred[ptr:ptr+batch_size] = pred_np
                dset_idx[ptr:ptr+batch_size] = idx_np
                dset_patch[ptr:ptr+batch_size] = patch.cpu().numpy()
                dset_src[ptr:ptr+batch_size] = src_np

                ptr += batch_size

    log.info(f"Evaluation complete! {ptr} physical patches from "
             f"{len(eval_dataset)} files packed into {h5_path}. "
             f"Mapping saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Astro-DSB Bulk Inference & Save")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = edict(yaml.safe_load(f))

    evaluate(cfg)

if __name__ == "__main__":
    main()