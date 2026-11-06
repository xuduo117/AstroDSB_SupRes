#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_npy_files_comprehensive_v2.py

Arbitrary-shape .npy inference for the AstroDSB / DDPM patch models.

The network only ever sees `patch_size x patch_size` (256x256) tiles, but the
files on disk can be any shape: (H, W), (C, H, W), or (..., H, W).  All of the
geometry is handled on the fly, in RAM, so nothing is written to disk except
the final map.

    * H or W smaller than the patch size
        -> the image is centred in a 256x256 canvas, the border is filled
           (zeros by default, see --pad_mode), and only the central region is
           read back out.

    * H or W larger than the patch size
        -> a sliding window with --stride covers the map.  The last tile of
           each row / column is flushed against the far edge, so the map is
           fully covered without any reflection padding, and the predictions
           are recombined with a Hann-weighted average.

    * Leading axes are treated as independent images (channels, frames, or the
      256x256 patch stacks the old pipeline produced) and are batched together
      on the GPU.

Notes on fidelity to v1:
    * The arcsinh normalisation maths is unchanged.  The only difference is
      that alpha / t_min / t_max / c_median are now computed from *real*
      pixels only (a validity mask), so padded borders and NaNs cannot bias
      the stretch.  For a fully-valid 256x256 patch the numbers are identical.
    * A file that is exactly 256x256 (or a stack of them) produces one tile
      per image, the blending window is flat in that case, and the result is
      the same as v1 - except that non-finite pixels are now filled before
      being handed to the network instead of propagating NaNs into the model.

Output shape always equals input shape.  Saved as float32 to
Recon_<name>.npy via a temp file + atomic rename, so a killed job never
leaves a half-written array behind.

Examples
--------
# whole directory, default 256/128 tiling
python inference_npy_files_comprehensive_v2.py \
    --config configs/astrodsb_supres.yaml \
    --data_dir /scratch/xuduo117/Real_Obs_Data/SuperRes/sdss \
    --batch_size 32

# one file, heavy overlap, write elsewhere
python inference_npy_files_comprehensive_v2.py \
    --config configs/astrodsb_supres.yaml \
    --files /path/HST_M83_9xSR_Input.npy \
    --stride 64 --out_dir ./final_super_res_maps/

# just report the tiling cost, no GPU work
python inference_npy_files_comprehensive_v2.py \
    --config configs/astrodsb_supres.yaml --data_dir ./maps --plan_only
"""

import argparse
import glob
import os

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

# Modular Components (Ensure these point to your actual codebase)
from core.factory import get_model
from core.diffusion import AstroDiffusionBridge, DDPMHandler


# =========================================================
# ASYMMETRIC DILUTION-AWARE ARCSINH [0, 1] NATIVE
# =========================================================
def apply_patch_asymmetric_arcsinh(patch_centered, valid_mask=None,
                                   transition_sigmas=3.0,
                                   headroom_multiplier=2.5,
                                   max_sigma=10.0):
    """Identical to v1, except that `valid_mask` restricts the statistics to
    real pixels.  Pass valid_mask=None (or an all-True mask) to reproduce v1."""
    c_math = 0.0
    stats = patch_centered if valid_mask is None else patch_centered[valid_mask]

    s = np.nanstd(stats) + 1e-8
    alpha = s * transition_sigmas
    tf_patch = np.arcsinh(patch_centered / alpha)

    abs_min, abs_max = np.nanmin(stats), np.nanmax(stats)
    perc_min, perc_max = np.nanpercentile(stats, 0.1), np.nanpercentile(stats, 99.9)
    sigma_min, sigma_max = c_math - max_sigma * s, c_math + max_sigma * s

    p_min = max(abs_min, min(perc_min, sigma_min))
    p_max = min(abs_max, max(perc_max, sigma_max))

    max_phys = headroom_multiplier * max(0, p_max)
    min_phys = headroom_multiplier * min(0, p_min)

    t_min = np.arcsinh(min_phys / alpha)
    t_max = np.arcsinh(max_phys / alpha)
    if t_max - t_min < 1e-6:
        t_max = t_min + 1.0

    final_norm = (tf_patch - t_min) / (t_max - t_min)
    final_norm = np.clip(final_norm, 0.0, 1.0)

    return final_norm, alpha, t_min, t_max


def reverse_patch_asymmetric_arcsinh(pred_norm, c_median, alpha, t_min, t_max):
    tf_data = pred_norm * (t_max - t_min) + t_min
    pred_centered = np.sinh(tf_data) * alpha
    return pred_centered + c_median


# =========================================================
# TILING GEOMETRY
# =========================================================
def overlap_slices(y0, x0, patch, H, W):
    """Slices of the map and of the tile for the part of the tile that
    actually falls inside the (H, W) image."""
    ys, ye = max(0, y0), min(H, y0 + patch)
    xs, xe = max(0, x0), min(W, x0 + patch)
    map_sl = (slice(ys, ye), slice(xs, xe))
    tile_sl = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
    return map_sl, tile_sl


class PatchTiler:
    """Turns an image of any size into a list of `patch_size` tiles and back."""

    def __init__(self, patch_size=256, stride=128, pad_mode="zero",
                 window_floor=1e-2):
        if stride < 1 or stride > patch_size:
            raise ValueError("stride must satisfy 1 <= stride <= patch_size")
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.pad_mode = pad_mode
        self.window_floor = float(window_floor)

    # ---------- where do the tiles go ----------
    def _starts(self, length):
        """Tile start indices along one axis.  Negative when the image is
        smaller than a tile (the image is then centred inside the tile)."""
        p, s = self.patch_size, self.stride
        if length <= p:
            return [-((p - length) // 2)]            # centred, zero/edge padded
        starts = list(range(0, length - p + 1, s))
        if starts[-1] != length - p:
            starts.append(length - p)                # flush the last tile
        return starts

    def plan(self, H, W):
        ys, xs = self._starts(H), self._starts(W)
        coords = [(y, x) for y in ys for x in xs]
        return coords, len(ys), len(xs)

    # ---------- blending ----------
    def window(self, n_tiles):
        """Hann taper for overlap-add.  A single tile gets a flat window so
        that small / exactly-256 images pass through untouched."""
        p = self.patch_size
        if n_tiles <= 1:
            return np.ones((p, p), dtype=np.float32)
        w = np.hanning(p).astype(np.float32)
        w = np.clip(w, self.window_floor, None)      # never exactly zero
        return np.outer(w, w).astype(np.float32)

    def weight_canvas(self, H, W, coords, window):
        """Sum of the blending weights landing on every pixel of the map."""
        wc = np.zeros((H, W), dtype=np.float32)
        for (y0, x0) in coords:
            msl, tsl = overlap_slices(y0, x0, self.patch_size, H, W)
            wc[msl] += window[tsl]
        np.maximum(wc, 1e-12, out=wc)
        return wc

    # ---------- extraction ----------
    def extract(self, img, y0, x0):
        """Return (tile, valid_mask).  `tile` is finite everywhere; padded and
        non-finite pixels are filled according to pad_mode.  `valid_mask`
        flags the pixels that came from real, finite data."""
        p = self.patch_size
        H, W = img.shape
        mask = np.zeros((p, p), dtype=bool)

        ys, ye = max(0, y0), min(H, y0 + p)
        xs, xe = max(0, x0), min(W, x0 + p)
        if ye <= ys or xe <= xs:                     # tile entirely outside
            return np.zeros((p, p), dtype=np.float32), mask

        top, left = ys - y0, xs - x0
        bottom, right = (y0 + p) - ye, (x0 + p) - xe

        sub = np.asarray(img[ys:ye, xs:xe], dtype=np.float32)
        sub_h, sub_w = sub.shape
        finite = np.isfinite(sub)
        mask[top:top + sub_h, left:left + sub_w] = finite

        if finite.all():
            interior, interior_med = sub, None
        elif finite.any():
            interior_med = float(np.median(sub[finite]))
            interior = np.where(finite, sub, interior_med).astype(np.float32)
        else:
            interior_med = 0.0
            interior = np.zeros_like(sub)

        if not (top or bottom or left or right):
            return np.ascontiguousarray(interior), mask

        pads = ((top, bottom), (left, right))
        mode = self.pad_mode
        if mode in ("reflect", "symmetric") and (
                max(top, bottom) >= sub_h or max(left, right) >= sub_w):
            mode = "edge"                            # too few pixels to reflect

        if mode == "median":
            if interior_med is None:
                interior_med = float(np.median(interior))
            tile = np.pad(interior, pads, mode="constant",
                          constant_values=interior_med)
        elif mode in ("reflect", "symmetric", "edge"):
            tile = np.pad(interior, pads, mode=mode)
        else:                                        # "zero"
            tile = np.pad(interior, pads, mode="constant", constant_values=0.0)

        return tile.astype(np.float32, copy=False), mask


# =========================================================
# INFERENCE ENGINE
# =========================================================
@torch.no_grad()
def predict_array(cfg, arr, model, diffusion, device, tiler,
                  batch_size=16, skip_steps=50, progress=True, desc="  Tiles"):
    """arr: (..., H, W) -> float32 prediction of exactly the same shape."""
    if arr.ndim < 2:
        raise ValueError(f"need at least 2 spatial axes, got shape {arr.shape}")

    lead = arr.shape[:-2]
    H, W = arr.shape[-2:]
    stack = np.asarray(arr, dtype=np.float32).reshape(-1, H, W)
    n_img = stack.shape[0]
    p = tiler.patch_size

    coords, _, _ = tiler.plan(H, W)
    window = tiler.window(len(coords))
    # Geometry is identical for every image in the stack, so one weight canvas
    # is enough.  It is built analytically, which keeps it consistent even
    # when an all-blank tile skips the network below.
    weights = tiler.weight_canvas(H, W, coords, window)

    out = np.zeros((n_img, H, W), dtype=np.float32)
    tasks = [(i, y, x) for i in range(n_img) for (y, x) in coords]

    is_ddpm = str(cfg.model.get('type', 'dsb')).lower() == 'ddpm'
    batches = range(0, len(tasks), batch_size)
    if progress:
        batches = tqdm(batches, total=(len(tasks) + batch_size - 1) // batch_size,
                       desc=desc, leave=False)

    for start in batches:
        chunk = tasks[start:start + batch_size]

        norm_tiles, memory, keep = [], [], []
        for (i, y0, x0) in chunk:
            tile, mask = tiler.extract(stack[i], y0, x0)
            if not mask.any():
                continue          # nothing real here; weight canvas already knows
            c_median = float(np.median(tile[mask]))
            if not np.isfinite(c_median):
                c_median = 0.0
            centered = tile - c_median
            t_norm, alpha, t_min, t_max = apply_patch_asymmetric_arcsinh(
                centered, valid_mask=mask)
            norm_tiles.append(t_norm)
            memory.append((c_median, alpha, t_min, t_max))
            keep.append((i, y0, x0))

        if not norm_tiles:
            continue

        t_cond = torch.tensor(np.asarray(norm_tiles, dtype=np.float32),
                              dtype=torch.float32).unsqueeze(1).to(device)

        if is_ddpm:
            pred_tensor = diffusion.solve_sde(model, shape=t_cond.shape,
                                              cond_img=t_cond,
                                              skip_steps=skip_steps, eta=0.0)
        else:
            pred_tensor = diffusion.ddpm_sampling(model, x_start=t_cond,
                                                  cond_img=t_cond,
                                                  skip_steps=skip_steps,
                                                  verbose=False)

        pred_numpy = pred_tensor.squeeze(1).cpu().numpy()

        for k, (i, y0, x0) in enumerate(keep):
            c, alpha, t_min, t_max = memory[k]
            phys = reverse_patch_asymmetric_arcsinh(pred_numpy[k], c, alpha,
                                                    t_min, t_max)
            msl, tsl = overlap_slices(y0, x0, p, H, W)
            out[i][msl] += (phys[tsl] * window[tsl]).astype(np.float32)

    out /= weights                                   # broadcast over the stack
    return out.reshape(lead + (H, W))


def process_files(cfg, target_files, model, diffusion, device, tiler,
                  out_dir=None, batch_size=16, skip_steps=50,
                  overwrite=False, preserve_invalid=False, plan_only=False):
    for file_path in target_files:
        filename = os.path.basename(file_path)
        dest_dir = out_dir if out_dir else os.path.dirname(os.path.abspath(file_path))
        os.makedirs(dest_dir, exist_ok=True)
        save_path = os.path.join(dest_dir, f"Recon_{filename}")

        already_done = os.path.exists(save_path) and not overwrite
        if already_done and not plan_only:
            print(f"Skipping {filename} - prediction already exists.")
            continue

        try:
            arr = np.load(file_path, mmap_mode='r')
        except Exception as exc:
            print(f"  [Error] Could not read {filename}: {exc}")
            continue

        if arr.ndim < 2:
            print(f"  [Error] Unsupported shape {arr.shape} for {filename}. "
                  f"Need at least 2 spatial axes.")
            continue

        H, W = arr.shape[-2:]
        coords, ny, nx = tiler.plan(H, W)
        n_img = int(np.prod(arr.shape[:-2]))
        n_tiles = len(coords) * n_img
        geom = ("centred + padded" if (H <= tiler.patch_size and W <= tiler.patch_size)
                else f"sliding window {ny}x{nx}")
        print(f"\n{filename}: shape {tuple(arr.shape)} -> {geom}, "
              f"{n_tiles} tiles ({n_img} image(s) x {len(coords)})")

        if plan_only:
            if already_done:
                print("    (would be skipped: Recon_ file already exists)")
            continue

        arr = np.asarray(arr)                        # realise the mmap
        pred = predict_array(cfg, arr, model, diffusion, device, tiler,
                             batch_size=batch_size, skip_steps=skip_steps,
                             desc=f"  {filename}")

        if preserve_invalid:
            pred[~np.isfinite(arr)] = np.nan

        finite = pred[np.isfinite(pred)]
        rng = (f"[{finite.min():.4g}, {finite.max():.4g}]" if finite.size else "empty")
        tmp_path = save_path + ".tmp.npy"
        np.save(tmp_path, pred.astype(np.float32, copy=False))
        os.replace(tmp_path, save_path)              # atomic
        print(f"  -> Saved Recon_{filename}  shape {pred.shape}  range {rng}")


# =========================================================
# CLI
# =========================================================
def resolve_targets(args):
    if args.files:
        return [f for f in args.files]
    if not args.data_dir:
        raise SystemExit("give either --files or --data_dir")
    found = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    return [f for f in found if not os.path.basename(f).startswith("Recon_")]


def main():
    parser = argparse.ArgumentParser(
        description="Tiled diffusion inference for .npy maps of any shape.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory of target .npy files (Recon_* skipped).")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                        help="Explicit file list, overrides --data_dir.")
    parser.add_argument("--pattern", type=str, default="*.npy")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write Recon_*.npy (default: next to input).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint override (default: <log_dir>/<name>/latest.pt).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Tiles per GPU pass.")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Tile size the network was trained on.")
    parser.add_argument("--stride", type=int, default=128,
                        help="Sliding-window stride. Smaller = more overlap, "
                             "smoother seams, quadratically more compute.")
    parser.add_argument("--pad_mode", type=str, default="zero",
                        choices=["zero", "median", "reflect", "symmetric", "edge"],
                        help="Border fill for images smaller than a tile.")
    parser.add_argument("--skip_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the sampler for reproducible maps.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preserve_invalid", action="store_true",
                        help="Write NaN back where the input was not finite.")
    parser.add_argument("--plan_only", action="store_true",
                        help="Print the tiling plan and exit (no model needed).")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = edict(yaml.safe_load(f))

    tiler = PatchTiler(patch_size=args.patch_size, stride=args.stride,
                       pad_mode=args.pad_mode)
    target_files = resolve_targets(args)
    if not target_files:
        print("[Warning] No valid target .npy files found.")
        return
    print(f"Found {len(target_files)} file(s) to process.")

    if args.plan_only:
        process_files(cfg, target_files, None, None, None, tiler, plan_only=True)
        return

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Init Model
    model = get_model(cfg, None, None).to(device)
    model.eval()

    if str(cfg.model.get('type', 'dsb')).lower() == 'ddpm':
        diffusion = DDPMHandler(cfg, device=device)
    else:
        diffusion = AstroDiffusionBridge(cfg, device=device)

    # Load Weights
    ckpt_path = args.ckpt or os.path.join(cfg.experiment.base_log_dir,
                                          cfg.experiment.name, "latest.pt")
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)

    process_files(cfg, target_files, model, diffusion, device, tiler,
                  out_dir=args.out_dir, batch_size=args.batch_size,
                  skip_steps=args.skip_steps, overwrite=args.overwrite,
                  preserve_invalid=args.preserve_invalid)


if __name__ == "__main__":
    main()