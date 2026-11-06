#!/usr/bin/env python3
"""
dsb_ensemble_infer.py  --  posterior sampling and interpolation-kernel test
                           for Astro-DSB.  RUNS ON TRILLIUM (needs the GPU).

It answers three questions in one pass, writing a single .npz you download and
feed to dsb_ensemble_analyze.py:

  1. UNCERTAINTY.  The reverse bridge is stochastic: p_posterior() injects
     torch.randn_like(xt) at every step except the last, so two runs on the
     same input give two different maps.  This script draws N_SAMPLES of them
     per patch and keeps the whole stack.

  2. P(k) WITH ERROR BARS.  Per-sample power spectra from that stack give a
     median and a 16/84 band.  The analysis script also computes P(k) of the
     ENSEMBLE MEAN, which matters: the production sliding window at stride 32
     averages ~28 effectively independent posterior draws per pixel, so the
     delivered map is close to the mean rather than a sample.  Comparing the
     two curves measures how much sub-beam power that averaging costs.

  3. INTERPOLATION KERNEL.  The same low-resolution map is lifted onto the
     model grid with each requested scipy order and pushed through the model
     separately, so the effect of the kernel is isolated with everything else
     held fixed.  The upsampling reproduces dsb_upsample_pipeline.upsample()
     exactly, including grid_mode=True and the NaN handling, so the test
     reflects what deployment actually does.

WHAT THE MODEL SEES.  Inference here mirrors engine/predictor.py exactly:
primary_proj is nn.Identity, so the bridge starts from the normalised
condition, and the same array is passed as cond_img.  Weights are the EMA
weights, as in the trainer's validation path.

TYPICAL USE

  # posterior + kernel test on one region, 30 draws per patch
  python dsb_ensemble_infer.py \
      --code-dir .   (the repository root) \
      --config   /path/to/Astro_DSB/code/configs/SuperRes_dsb_0910.yaml \
      --ckpt     /path/to/results/astro_supres_dsb_0910/latest.pt \
      --lowres   planck217_native.npy \
      --gt       act220_modelgrid.npy \
      --factor 5 --orders 1,3 --n-samples 30 --n-patches 8 \
      --out ensemble_act_0910.npz

  # if your input is already on the model grid, skip --lowres/--factor:
  python dsb_ensemble_infer.py ... --input planck_x5.npy --gt act.npy

A NOTE ON --skip-steps.  bridge.ddpm_sampling builds its schedule as
np.arange(1000)[::skip_steps], so the number of network evaluations is
1000/skip_steps, NOT skip_steps.  The config's inference.sampling_skip_steps of
10 therefore means 100 evaluations.

USE 10, NOT 100.  engine/predictor.py takes skip_steps straight from
inference.sampling_skip_steps, so 10 is what produced the published maps.
--skip-steps 100 gives only TEN evaluations, and because ddpm_sampling starts
from xt = x_start.clone(), which is the condition itself, ten steps are not
enough to travel away from it: the prediction comes back sitting on top of the
input in P(k), the draws barely differ, and the interpolation variants become
indistinguishable.  All four symptoms at once.  If a run looks like that, check
this first.
"""
import argparse
import contextlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch


# ----------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(
        description="Draw posterior samples from Astro-DSB and test the "
                    "interpolation kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("model")
    g.add_argument('--code-dir', required=True,
                   help="Astro_DSB/code on Trillium (contains core/, data/, engine/)")
    g.add_argument('--config', required=True, help="the run's yaml")
    g.add_argument('--ckpt', required=True, help="latest.pt or an archive checkpoint")
    g.add_argument('--no-ema', action='store_true',
                   help="use the raw weights instead of the EMA weights "
                        "(the trainer validates with EMA, so leave this off "
                        "to match the published inference)")

    g = p.add_argument_group("input")
    g.add_argument('--lowres',
                   help="low-resolution map on its NATIVE grid, .npy or .fits. "
                        "The script lifts it onto the model grid at each "
                        "--orders value. Use this for the kernel test.")
    g.add_argument('--input',
                   help="alternative: a map ALREADY on the model grid. "
                        "The kernel test is then unavailable (one order only).")
    g.add_argument('--region',
                   help="alternative: the .npz written by crop_region.py. Uses "
                        "'planck' as the input, 'act' as the truth, and every "
                        "'var_*' entry as a further named variant, so the "
                        "kernel comparison runs on identical patches.")
    g.add_argument('--gt', help="high-resolution truth on the model grid, optional")
    g.add_argument('--factor', type=int, default=5,
                   help="upsampling factor applied to --lowres")
    g.add_argument('--orders', default='1,3',
                   help="comma-separated scipy interpolation orders to test")

    g = p.add_argument_group("sampling")
    g.add_argument('--n-samples', type=int, default=30,
                   help="posterior draws per patch")
    g.add_argument('--n-patches', type=int, default=8,
                   help="patches to evaluate (0 = every valid patch)")
    g.add_argument('--patch-size', type=int, default=256)
    g.add_argument('--patch-stride', type=int, default=0,
                   help="0 means non-overlapping (stride = patch size)")
    g.add_argument('--skip-steps', type=int, default=0,
                   help="0 takes inference.sampling_skip_steps from the config. "
                        "Network evaluations = 1000/skip_steps; see the header.")
    g.add_argument('--batch', type=int, default=10,
                   help="posterior draws evaluated together on the GPU")
    g.add_argument('--ot-ode', action='store_true',
                   help="deterministic probability-flow variant, for comparison. "
                        "All draws then collapse to one map.")
    g.add_argument('--max-nan-frac', type=float, default=0.02,
                   help="reject a patch with more than this fraction of blank pixels")

    g = p.add_argument_group("mosaic (build the whole region, not loose patches)")
    g.add_argument('--mosaic', action='store_true',
                   help="tile the model across the whole crop and blend, so each "
                        "draw is one full-size map. Needed for a 2048 image and "
                        "for P(k) at large angular scales, which a 256 patch "
                        "cannot reach (its fundamental mode is 128 arcmin).")
    g.add_argument('--mosaic-stride', type=int, default=128,
                   help="tile stride. Smaller is smoother and much slower; the "
                        "tile count goes as stride^-2.")
    g.add_argument('--noise-mode', default='coherent',
                   choices=['coherent', 'independent'],
                   help="coherent: one noise field per draw, shared by every "
                        "overlapping tile, so blending does NOT average the "
                        "generative component away. independent: fresh noise "
                        "per tile, which is what predict_large_snapshot does "
                        "today and collapses the draws toward the mean.")
    g.add_argument('--global-norm', action='store_true',
                   help="one arcsinh normalisation for the whole crop instead "
                        "of per tile. Production normalises per tile; use this "
                        "if the spread map shows tile seams.")
    g.add_argument('--seed', type=int, default=1234)

    g = p.add_argument_group("MC dropout (epistemic term, mosaic mode only)")
    g.add_argument('--mc-dropout', type=int, default=0,
                   help="extra draws with the Dropout layers left ON and the "
                        "reverse-SDE noise held FIXED, so the only thing varying "
                        "is the dropout mask. 0 disables. This is the epistemic "
                        "term in the ISRF interpretability script, made "
                        "separable here rather than mixed with the SDE spread.")
    g.add_argument('--mc-dropout-variant', default=None,
                   help="which variant to run MC dropout on. Default: the "
                        "primary input. Running it on one variant is enough, "
                        "since epistemic uncertainty is a property of the "
                        "weights, not of the interpolation.")

    g = p.add_argument_group("output")
    g.add_argument('--out', required=True, help="output .npz")
    g.add_argument('--save-dtype', default='float32', choices=['float32', 'float16'],
                   help="float16 halves the download, but its largest finite "
                        "value is 65504 and these maps reach 1e6 in uK_CMB, so "
                        "it is refused when the data would overflow")
    g.add_argument('--device', default='cuda',
                   help="'cuda', 'cuda:N' for a specific GPU, or 'cpu'")
    g.add_argument('--allow-cpu', action='store_true',
                   help="proceed on CPU when no usable GPU is found instead of "
                        "stopping. A mosaic run on CPU takes days, so this is "
                        "opt-in rather than a silent fallback.")
    return p.parse_args()


# ------------------------------------------------------------------- helpers
def load_map(path):
    """Read a .npy or a .fits into a 2D float32 array."""
    if path.lower().endswith(('.fits', '.fit', '.fz')):
        from astropy.io import fits
        with fits.open(path) as hdul:
            for h in hdul:
                if getattr(h, 'data', None) is not None and h.data.ndim >= 2:
                    return np.squeeze(np.asarray(h.data, dtype=np.float32))
        raise ValueError(f"no image HDU in {path}")
    a = np.load(path)
    return np.squeeze(np.asarray(a, dtype=np.float32))


def upsample(data, factor, order):
    """Lift `data` onto a grid `factor` times finer.

    This is dsb_upsample_pipeline.upsample() reproduced verbatim, so that the
    kernel test measures the deployed behaviour rather than an approximation
    of it.  NaNs are block-replicated rather than interpolated, which keeps a
    survey boundary exactly where it was instead of eroding or growing it.
    """
    f = int(factor)
    if order == 0:
        return np.repeat(np.repeat(data, f, axis=0), f, axis=1)

    from scipy.ndimage import zoom as _ndzoom
    bad = ~np.isfinite(data)
    filled = data
    if bad.any():
        fill = np.nanmedian(data) if np.isfinite(data).any() else 0.0
        filled = np.where(bad, fill, data)
    # grid_mode=True treats pixels as areas, the convention the CRPIX transform
    # assumes; without it the result is offset by (f-1)/(2f) of a coarse pixel.
    out = _ndzoom(filled, f, order=order, grid_mode=True, mode='nearest')
    if bad.any():
        out[np.repeat(np.repeat(bad, f, axis=0), f, axis=1)] = np.nan
    return out.astype(np.float32)


def load_model(args, device):
    """Build the network and bridge from the config and checkpoint."""
    sys.path.insert(0, os.path.abspath(args.code_dir))
    import yaml
    from easydict import EasyDict as edict
    from core.factory import get_model
    from core.diffusion import AstroDiffusionBridge

    cfg = edict(yaml.safe_load(open(args.config)))
    cfg.local_rank, cfg.global_rank, cfg.global_size = 0, 0, 1

    model = get_model(cfg, None, None)          # get_model ignores loader/log
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state'])

    used_ema = False
    if not args.no_ema and 'ema_state' in ck:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(),
                                       decay=float(cfg.model.ema_decay))
        ema.load_state_dict(ck['ema_state'])
        ema.copy_to(model.parameters())         # same as the trainer's val path
        used_ema = True

    model.to(device).eval()
    bridge = AstroDiffusionBridge(cfg, device=device)

    # the pipeline's own arcsinh, imported so normalisation cannot drift
    from data.datasets import apply_patch_asymmetric_arcsinh

    meta = dict(epoch=int(ck.get('epoch', -1)),
                loss=float(ck.get('loss', np.nan)),
                used_ema=used_ema,
                config=os.path.abspath(args.config),
                ckpt=os.path.abspath(args.ckpt))
    return cfg, model, bridge, apply_patch_asymmetric_arcsinh, meta


def normalise(patch, aaa):
    """Adaptive asymmetric arcsinh, exactly as the training pipeline applies it.

    The transform is scale-equivariant (alpha = 3*sigma scales with the data and
    the bounds are arcsinh of ratios), so it does not matter whether the patch
    was pre-divided by its peak.  Returns the [0,1] map and the three constants
    needed to invert it.
    """
    _, mem = aaa(np.asarray(patch, dtype=np.float64), headroom_multiplier=2.5)
    alpha, t_min, t_max = float(mem[1]), float(mem[2]), float(mem[3])
    y = (np.arcsinh(patch / alpha) - t_min) / (t_max - t_min)
    return np.clip(y, 0.0, 1.0).astype(np.float32), (alpha, t_min, t_max)


def denormalise(y, par):
    alpha, t_min, t_max = par
    return (alpha * np.sinh(np.asarray(y, dtype=np.float64)
                            * (t_max - t_min) + t_min)).astype(np.float32)


@torch.no_grad()
def draw_samples(model, bridge, y_norm, n, skip_steps, batch, device,
                 seed, ot_ode=False):
    """n posterior draws for one normalised patch.

    Draws are batched: p_posterior calls torch.randn_like on the whole batch,
    so every element of the batch is an independent realisation.  That is the
    same mechanism by which overlapping sliding-window patches differ.
    """
    out = np.empty((n, y_norm.shape[0], y_norm.shape[1]), dtype=np.float32)
    base = torch.from_numpy(y_norm)[None, None].to(device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        torch.manual_seed(seed + done)          # reproducible per draw index
        x = base.repeat(b, 1, 1, 1)
        if ot_ode:
            # deterministic variant: temporarily force the ODE branch
            orig = bridge.p_posterior
            bridge.p_posterior = (lambda *a, **k:
                                  orig(*a, **{**k, 'ot_ode': True}))
            try:
                pred = bridge.ddpm_sampling(model, x, cond_img=x,
                                            verbose=False, skip_steps=skip_steps)
            finally:
                bridge.p_posterior = orig
        else:
            pred = bridge.ddpm_sampling(model, x, cond_img=x,
                                        verbose=False, skip_steps=skip_steps)
        out[done:done + b] = pred[:, 0].float().cpu().numpy()
        done += b
    return out


def hann2d(n):
    """The production blending window, floored so the map border is not
    divided by a vanishing weight (production instead reflection-pads)."""
    w = np.sin(np.pi * np.arange(n) / (n - 1)) ** 2
    return np.maximum(np.outer(w, w), 1e-3).astype(np.float32)


class TileNoise:
    """Serves the bridge its noise from a pre-drawn full-map field.

    core/diffusion.py calls torch.randn_like(xt) once per reverse step inside
    p_posterior.  Swapping that for this, with the field indexed by ABSOLUTE
    position, makes overlapping tiles share a realisation: the generative part
    of the prediction is then coherent across tile boundaries and survives the
    blend instead of averaging out.  The bridge's own maths is untouched.
    """

    def __init__(self, fields, patch):
        self.fields, self.patch = fields, patch
        self.wins, self.i = [], 0

    def begin(self, wins):
        self.wins, self.i = wins, 0

    def __call__(self, x, *a, **k):
        f = self.fields[min(self.i, len(self.fields) - 1)]
        self.i += 1
        p = self.patch
        out = torch.stack([f[y:y + p, xx:xx + p] for (y, xx) in self.wins])
        return out[:, None].to(device=x.device, dtype=x.dtype)


@contextlib.contextmanager
def randn_like_as(provider):
    orig = torch.randn_like
    torch.randn_like = provider
    try:
        yield
    finally:
        torch.randn_like = orig


def n_noise_steps(num_steps, skip):
    """How many times p_posterior actually draws noise (the last step does not)."""
    steps = np.arange(num_steps)[::skip][::-1]
    if steps[-1] != 0:
        steps = np.append(steps, 0)
    return max(len(steps) - 2, 1)


@torch.no_grad()
def enable_mc_dropout(model):
    """Put only the Dropout layers back into training mode.

    Everything else -- normalisation statistics above all -- stays in eval, so
    the only thing that varies between passes is which units are dropped.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                          torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            n += 1
    return n


def sample_mosaic(model, bridge, full, aaa, args, skip, device, seed,
                  n_draws=None, fixed_noise=False):
    """n_draws full-region maps, each tiled and Hann-blended.

    fixed_noise=True gives every draw the SAME reverse-SDE realisation, so with
    MC dropout switched on the only thing varying between draws is the dropout
    mask. That separation matters: draws that re-roll both the noise and the
    mask measure aleatoric + epistemic together, not epistemic.
    """
    n_draws = args.n_samples if n_draws is None else n_draws
    H, W = full.shape
    p, st = args.patch_size, args.mosaic_stride
    ys = list(range(0, max(H - p, 0) + 1, st)) or [0]
    xs = list(range(0, max(W - p, 0) + 1, st)) or [0]
    if ys[-1] != H - p: ys.append(H - p)
    if xs[-1] != W - p: xs.append(W - p)
    tiles = [(y, x) for y in ys for x in xs]
    win = hann2d(p)
    clean = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # normalisation: per tile as production does, or once for the whole crop.
    # Note the interaction with coherent noise: the bridge injects its noise in
    # NORMALISED space, so with per-tile constants the same noise value means a
    # slightly different physical amplitude in each tile. The constants vary
    # smoothly between neighbours so the effect is small, but --global-norm
    # removes it entirely if the spread map shows seams.
    if args.global_norm:
        gn, gpar = normalise(clean, aaa)
        norms = [gn[y:y + p, x:x + p] for y, x in tiles]
        pars = [gpar] * len(tiles)
    else:
        both = [normalise(clean[y:y + p, x:x + p], aaa) for y, x in tiles]
        norms = [b[0] for b in both]
        pars = [b[1] for b in both]

    n_noise = n_noise_steps(int(bridge.num_steps), skip)
    out = np.empty((n_draws, H, W), dtype=np.float32)
    print(f"       {len(tiles)} tiles of {p} at stride {st}, "
          f"{args.noise_mode} noise, "
          f"{'global' if args.global_norm else 'per-tile'} normalisation")

    for si in range(n_draws):
        prov = None
        if args.noise_mode == 'coherent':
            g = torch.Generator().manual_seed(seed + (0 if fixed_noise
                                                      else 7919 * si))
            prov = TileNoise([torch.randn(H, W, generator=g)
                              for _ in range(n_noise)], p)
        acc = np.zeros((H, W), dtype=np.float64)
        wsum = np.zeros((H, W), dtype=np.float64)
        for b0 in range(0, len(tiles), args.batch):
            sel = list(range(b0, min(b0 + args.batch, len(tiles))))
            wins = [tiles[i] for i in sel]
            x = torch.from_numpy(np.stack([norms[i] for i in sel]))[:, None].to(device)
            if prov is not None:
                prov.begin(wins)
                with randn_like_as(prov):
                    pred = bridge.ddpm_sampling(model, x, cond_img=x,
                                                verbose=False, skip_steps=skip)
            else:
                torch.manual_seed(seed + (0 if fixed_noise else 7919 * si) + b0)
                pred = bridge.ddpm_sampling(model, x, cond_img=x,
                                            verbose=False, skip_steps=skip)
            raw = pred[:, 0].float().cpu().numpy()
            for j, i in enumerate(sel):
                y, xx = tiles[i]
                acc[y:y + p, xx:xx + p] += denormalise(raw[j], pars[i]) * win
                wsum[y:y + p, xx:xx + p] += win
        out[si] = (acc / np.maximum(wsum, 1e-8)).astype(np.float32)
        print(f"       draw {si+1}/{n_draws} done", flush=True)
    return out


def pick_patches(arr, size, stride, want, max_nan_frac):
    """Patch origins on a regular grid, most-structured first."""
    h, w = arr.shape
    stride = stride or size
    cands = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            sub = arr[y:y + size, x:x + size]
            bad = ~np.isfinite(sub)
            if bad.mean() > max_nan_frac:
                continue
            good = sub[~bad]
            if good.size < 16 or not np.isfinite(good.std()) or good.std() <= 0:
                continue
            cands.append((float(good.std()), y, x))
    if not cands:
        raise RuntimeError("no patch passed the blank-pixel and variance cuts; "
                           "loosen --max-nan-frac or check the input")
    cands.sort(reverse=True)                    # most structure first
    if want and want < len(cands):
        # spread the selection over the ranking instead of taking the top few,
        # so the sample is not all bright filaments
        idx = np.linspace(0, len(cands) - 1, want).round().astype(int)
        cands = [cands[i] for i in idx]
    return [(y, x) for _, y, x in cands]


FLOAT16_MAX = 65504.0


def to_save_dtype(arr, want, tag):
    """Cast for storage, refusing float16 when it would turn data into inf.

    ACT/Planck maps in uK_CMB reach ~1e6 in the Galactic plane. float16 tops out
    at 65504, so a plain .astype(np.float16) silently writes inf for the
    brightest pixels. Those infs then propagate: the ensemble mean is inf there,
    and np.nanstd of an array containing inf is nan, not a number, so the
    summary statistics come back nan without anything having looked wrong.
    """
    if want != 'float16':
        return arr.astype(np.float32)
    peak = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.0
    if peak > FLOAT16_MAX:
        n = int((np.abs(arr) > FLOAT16_MAX).sum())
        print(f"[{tag}] float16 REFUSED: peak |value| {peak:.4g} exceeds the "
              f"float16 limit {FLOAT16_MAX:.0f};")
        print(f"       {n} values ({n/arr.size*100:.3f}%) would become inf. "
              f"Saving float32 instead.")
        print("       Halve the download by cutting --n-samples or the crop "
              "size rather than the dtype.")
        return arr.astype(np.float32)
    return arr.astype(np.float16)


def resolve_device(args):
    """Settle the GPU before anything expensive, and explain any failure here.

    Torch defers most CUDA errors until the first .to(device), which is after
    the checkpoint has been read, so a broken environment surfaces as a wall of
    traceback from inside nn.Module._apply.  Everything that can be checked is
    checked up front instead.
    """
    want = args.device
    host = os.environ.get('SLURMD_NODENAME') or platform.node()
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    in_job = bool(os.environ.get('SLURM_JOB_ID'))

    def bail(msg, extra=()):
        lines = [f"\n{msg}", f"  host                 {host}",
                 f"  SLURM_JOB_ID         {os.environ.get('SLURM_JOB_ID', '(none)')}",
                 f"  CUDA_VISIBLE_DEVICES {cvd if cvd is not None else '(unset)'}"]
        lines += ['  ' + e for e in extra]
        if not in_job:
            lines += ["", "You are not inside a SLURM allocation."]
            if 'login' in host.lower():
                lines += [
                    f"The host name {host!r} says login node, and login nodes "
                    f"are not where GPU work runs."]
            lines += [
                "A CUDA_VISIBLE_DEVICES left over from an earlier session is "
                "exactly what makes",
                "the visible-device list and the runtime disagree. Submit the "
                "job, or take an",
                "interactive allocation, and run it there. If it still fails "
                "inside a job, try",
                "  unset CUDA_VISIBLE_DEVICES",
                "and let SLURM set it."]
        lines += ["", "To run on CPU anyway, pass --allow-cpu. For a mosaic run "
                      "that is days, not hours."]
        sys.exit("\n".join(lines))

    if want.startswith('cpu'):
        print("[dev ] CPU requested")
        return torch.device('cpu')

    if not torch.cuda.is_available():
        if args.allow_cpu:
            print("[dev ] no GPU available, falling back to CPU as asked")
            return torch.device('cpu')
        bail("No CUDA device is available.")

    # device_count() reads NVML and can disagree with the CUDA runtime when
    # CUDA_VISIBLE_DEVICES names a device that is not there.  That mismatch is
    # the 'device >= 0 && device < num_gpus INTERNAL ASSERT FAILED' crash, and
    # it is an environment problem, not a torch bug.
    try:
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
    except Exception as e:                                   # noqa: BLE001
        if args.allow_cpu:
            print(f"[dev ] CUDA present but unusable ({e.__class__.__name__}), "
                  f"falling back to CPU as asked")
            return torch.device('cpu')
        bail("CUDA reports devices but cannot describe them: the visible-device "
             "list and the CUDA\nruntime disagree about how many GPUs exist.",
             [f"underlying error: {e.__class__.__name__}: {e}"])

    if n == 0:
        if args.allow_cpu:
            print("[dev ] zero GPUs visible, falling back to CPU as asked")
            return torch.device('cpu')
        bail("CUDA is available but zero devices are visible.")

    idx = 0
    if ':' in want:
        try:
            idx = int(want.split(':', 1)[1])
        except ValueError:
            sys.exit(f"could not read a device index from --device {want!r}")
        if idx >= n:
            bail(f"--device {want} was asked for but only {n} GPU(s) are "
                 f"visible (0 to {n-1}).")

    dev = torch.device(f'cuda:{idx}')
    try:                       # force initialisation now, not mid-run
        torch.zeros(1, device=dev)
    except Exception as e:                                   # noqa: BLE001
        bail(f"GPU {idx} is visible but cannot be used.",
             [f"underlying error: {e.__class__.__name__}: {e}"])
    free, total = torch.cuda.mem_get_info(dev)
    print(f"[dev ] {dev}  {names[idx]}  "
          f"{free/2**30:.1f} of {total/2**30:.1f} GiB free"
          + (f"   ({n} GPUs visible)" if n > 1 else ""))
    return dev


# ---------------------------------------------------------------------- main
def main():
    args = parse_args()
    given = [bool(args.lowres), bool(args.input), bool(args.region)]
    if sum(given) != 1:
        sys.exit("give exactly one of --lowres, --input or --region")

    device = resolve_device(args)

    print(f"[load] {args.ckpt}")
    cfg, model, bridge, aaa, meta = load_model(args, device)
    skip = args.skip_steps or int(cfg.get('inference', {})
                                  .get('sampling_skip_steps', 10))
    n_eval = int(np.ceil(int(cfg.model.intervals) / skip))
    print(f"[load] epoch {meta['epoch']} | EMA {meta['used_ema']} | "
          f"skip_steps {skip} -> ~{n_eval} network evaluations per draw")

    # ---- one model-grid map per named variant ------------------------------
    inputs, gt = {}, None
    if args.lowres:
        low = load_map(args.lowres)
        print(f"[data] low-res {low.shape} -> x{args.factor}")
        for o in [int(v) for v in args.orders.split(',')]:
            inputs[f'order{o}'] = upsample(low, args.factor, o)
    elif args.region:
        r = np.load(args.region, allow_pickle=True)
        inputs['planck'] = np.asarray(r['planck'], dtype=np.float32)
        for key in r.files:
            if key.startswith('var_'):
                inputs[key[4:]] = np.asarray(r[key], dtype=np.float32)
        if 'act' in r.files:
            gt = np.asarray(r['act'], dtype=np.float32)
        rmeta = json.loads(str(r['meta'])) if 'meta' in r.files else {}
        print(f"[data] region {inputs['planck'].shape} from "
              f"{os.path.basename(args.region)}  variants "
              f"{list(inputs)}  truth {'yes' if gt is not None else 'no'}")
        if rmeta:
            print(f"[data] crop ({rmeta.get('crop_x')}, {rmeta.get('crop_y')}) "
                  f"size {rmeta.get('size')}")
    else:
        inputs['asis'] = load_map(args.input)
        print(f"[data] input {inputs['asis'].shape} (already on the model grid)")

    if args.gt:
        gt = load_map(args.gt)

    # crop everything to a common shape
    shapes = [a.shape for a in inputs.values()] + ([gt.shape] if gt is not None else [])
    H = min(s[0] for s in shapes)
    W = min(s[1] for s in shapes)
    inputs = {k: v[:H, :W] for k, v in inputs.items()}
    if gt is not None:
        gt = gt[:H, :W]
    print(f"[data] common grid {H} x {W}")

    # ---- patch selection, shared across orders so they are comparable ------
    ref = inputs[sorted(inputs)[0]]
    origins = pick_patches(ref, args.patch_size, args.patch_stride,
                           args.n_patches, args.max_nan_frac)
    print(f"[data] {len(origins)} patches of {args.patch_size} px")

    S = args.patch_size
    store = {}
    t0 = time.time()

    if args.mosaic:
        # one "patch" that is the whole crop, so the schema and the analysis
        # script are unchanged: samples stay (n_samples, n_patches, H, W)
        for tag, full in inputs.items():
            print(f"[{tag}] mosaic {full.shape}")
            # The SAME seed for every variant: draw i of 'zoom1' and draw i of
            # 'zoom3' then share a noise realisation, so their difference is the
            # kernel alone. That pairing is what makes the variance split in
            # dsb_ensemble_analyze.py meaningful. (The old abs(hash(tag)) was
            # also not reproducible: str hashing is salted per process.)
            m = sample_mosaic(model, bridge, full, aaa, args, skip, device,
                              args.seed)
            store[f"{tag}_samples"] = to_save_dtype(m[:, None],
                                                   args.save_dtype, tag)
            store[f"{tag}_input"] = np.nan_to_num(full, nan=0.0).astype(np.float32)[None]
            store[f"{tag}_norm"] = np.zeros((1, 3))
            print(f"[{tag}] done  |  {(time.time()-t0)/60:.1f} min elapsed")
        # ---- MC dropout: vary the weights, hold the noise still -----------
        if args.mc_dropout > 0:
            tag = args.mc_dropout_variant or sorted(inputs)[0]
            if tag not in inputs:
                print(f"[mcd ] '{tag}' is not one of {list(inputs)}; skipping "
                      f"MC dropout")
            else:
                n_drop = enable_mc_dropout(model)
                if n_drop == 0:
                    print("[mcd ] no Dropout layers in this model, so the "
                          "epistemic spread would be exactly zero; skipping")
                else:
                    print(f"[mcd ] {n_drop} Dropout layers back in train mode, "
                          f"{args.mc_dropout} draws on '{tag}' with the SDE "
                          f"noise frozen")
                    m = sample_mosaic(model, bridge, inputs[tag], aaa, args,
                                      skip, device, args.seed,
                                      n_draws=args.mc_dropout, fixed_noise=True)
                    store['dropout_samples'] = to_save_dtype(
                        m[:, None], args.save_dtype, 'mcd')
                    store['dropout_variant'] = np.array([tag])
                    model.eval()          # dropout off again

        if gt is not None:
            store['gt'] = gt[None].astype(np.float32)
        origins = [(0, 0)]
        inputs_iter = []
    else:
        inputs_iter = list(inputs.items())

    for tag, full in inputs_iter:
        n_p, n_s = len(origins), args.n_samples
        samp = np.empty((n_s, n_p, S, S), dtype=np.float32)
        inp = np.empty((n_p, S, S), dtype=np.float32)
        norm = np.empty((n_p, 3), dtype=np.float64)

        for i, (y, x) in enumerate(origins):
            patch = np.nan_to_num(full[y:y + S, x:x + S], nan=0.0,
                                  posinf=0.0, neginf=0.0).astype(np.float32)
            inp[i] = patch
            y_norm, par = normalise(patch, aaa)
            norm[i] = par
            raw = draw_samples(model, bridge, y_norm, n_s, skip, args.batch,
                               device, args.seed + 10007 * i, args.ot_ode)
            for j in range(n_s):
                samp[j, i] = denormalise(raw[j], par)
            el = time.time() - t0
            print(f"[{tag}] patch {i+1}/{n_p} at ({y},{x})  "
                  f"{n_s} draws  |  {el/60:.1f} min elapsed", flush=True)

        store[f"{tag}_samples"] = to_save_dtype(samp, args.save_dtype, tag)
        store[f"{tag}_input"] = inp
        store[f"{tag}_norm"] = norm

    if gt is not None and 'gt' not in store:
        store['gt'] = np.stack([gt[y:y + S, x:x + S] for y, x in origins])

    store['origins'] = np.array(origins, dtype=np.int32)
    store['variants'] = np.array([str(k) for k in inputs], dtype=object)
    # Built defensively: everything above this point is expensive, and a
    # missing config key must not throw away a finished run.
    try:
        px_hint = float(getattr(cfg, 'dataset', {}).get('act_pixel_scale', 0.5))
    except Exception:                                            # noqa: BLE001
        px_hint = 0.5
    try:
        store['meta'] = json.dumps(dict(
            meta, skip_steps=skip, n_eval_per_draw=n_eval,
            n_samples=args.n_samples, patch_size=S, factor=args.factor,
            ot_ode=bool(args.ot_ode), mosaic=bool(args.mosaic),
            mosaic_stride=int(args.mosaic_stride), noise_mode=args.noise_mode,
            global_norm=bool(args.global_norm),
            # draw i uses the same noise realisation in every variant, which is
            # what lets the analysis split the variance into a seed part and a
            # kernel part. Older files lack this key.
            paired_seeds=True, save_dtype=args.save_dtype,
            mc_dropout=int(args.mc_dropout),
            lowres=args.lowres, input=args.input,
            gt=args.gt, region=args.region, variants=list(inputs),
            seed=args.seed, pixel_scale_hint=px_hint,
        ), default=str)
    except Exception as e:                                       # noqa: BLE001
        print(f"  warning: could not serialise the metadata ({e}); "
              f"writing a minimal record instead")
        store['meta'] = json.dumps(dict(n_samples=args.n_samples, patch_size=S,
                                        variants=list(inputs), epoch=-1))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.savez_compressed(args.out, **store)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[done] {args.out}  ({mb:.1f} MB)  in {(time.time()-t0)/60:.1f} min")
    print("       download it and run dsb_ensemble_analyze.py on it")


if __name__ == '__main__':
    main()