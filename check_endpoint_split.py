#!/usr/bin/env python3
"""Regression guard for the DESIGN INVARIANT of this pipeline.

    Every observed FEATURE is real and belongs in the ground truth -- the CMB
    included. Only tiny fluctuations (Gaussian noise) and instrument
    distortion are absent from the target.

In pipeline terms: a SKY component is added to base_signal BEFORE the beam
split, so it lands in BOTH the condition and the target (the model sharpens
it). An INSTRUMENT component is added to the condition only (the model removes
it). Nothing may appear in the target alone -- that would be asking the model
to hallucinate.

Run it from the `code/` directory after any edit to data/datasets.py:

    python check_endpoint_split.py configs/SuperRes_dsb_0901.yaml

Exit code 0 = the invariant holds. Non-zero = something moved endpoints.

HOW IT WORKS. For each stochastic component the pipeline is run twice with
identical seeds, once with that component's contribution multiplied by 1 and
once by 0. The wrappers still call the original function, so RNG consumption is
identical and the ONLY difference is that component's contribution.

Two deliberate, benign couplings would otherwise show up as false positives and
are neutralised here, then reported separately at the end:

  * the arcsinh normalisation constants are derived from the NOISY input and
    shared with the target, so removing noise shifts the normalised target;
  * TF.rotate fills the corners it rotates into frame with c_fill =
    median(noisy input), for BOTH endpoints.

Neither puts noise into the target's physical content, and both are invertible
/ identical across the two endpoints, so both are fine -- but they have to be
held fixed for the test to measure what it claims to measure.
"""
import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import datasets as ds                                  # noqa: E402

try:
    from easydict import EasyDict as edict
except ImportError:                                              # not required
    class edict(dict):                                           # noqa: N801
        def __init__(self, d=None, **kw):
            super().__init__(d or {}, **kw)
            for k, v in list(self.items()):
                self[k] = edict(v) if isinstance(v, dict) else v
        __getattr__ = dict.get

CROP, TS, PS, FOV = 768, 256, 51, 128.0

# component -> (how to wrap it, what it must be, label for the report)
FIELD, PATCH, CFG = 'field', 'patch', 'cfg'
COMPONENTS = [
    ('generate_diffuse_field',        FIELD, 'SKY',        'diffuse field (replaces the crop)'),
    ('generate_diffuse_cirrus',       FIELD, 'SKY',        'cirrus foreground'),
    ('generate_magnetic_striations',  FIELD, 'SKY',        'magnetic striations'),
    ('generate_cmb_grf',              FIELD, 'SKY',        'CMB'),
    ('generate_cib_field',            FIELD, 'SKY',        'CIB galaxies'),
    ('generate_stellar_field',        FIELD, 'SKY',        'stellar field'),
    ('inject_point_sources',          PATCH, 'SKY',        'point sources'),
    ('apply_advanced_telescope_noise', PATCH, 'INSTRUMENT', 'detector noise (white + shot)'),
    ('apply_scanning_stripes',        PATCH, 'INSTRUMENT', 'scanning stripes'),
    ('generate_synthetic_act_noise',  FIELD, 'INSTRUMENT', 'correlated 1/f noise'),
    ('generate_clumpy_pepper_noise',  FIELD, 'INSTRUMENT', 'pepper speckle'),
    ('post_term',                     CFG,   'INSTRUMENT', 'post-upsample white term'),
    ('subpixel_shift',                CFG,   'INSTRUMENT', 'sub-pixel misregistration'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--seeds', type=int, default=14)
    ap.add_argument('--tol', type=float, default=1e-6)
    args = ap.parse_args()

    cfg = edict(yaml.safe_load(open(args.config)))
    dcfg = ds.diffuse_cfg_from(cfg)
    ucfg = ds.upsample_cfg_from(cfg) if hasattr(ds, 'upsample_cfg_from') else None
    ncfg = dict(ds.noise_cfg_from(cfg)) if hasattr(ds, 'noise_cfg_from') else None
    if ncfg:                     # force the optional branches so they are tested
        ncfg['corr_prob'] = 1.0
        ncfg['pepper_prob'] = 1.0

    # ---- neutralise the two benign couplings -------------------------------
    ref = {}
    orig_norm = ds.apply_patch_asymmetric_arcsinh
    def pinned_norm(patch, **kw):
        if 'mem' not in ref:
            ref['mem'] = orig_norm(patch, **kw)[1].clone()
        return np.zeros_like(patch), ref['mem'].clone()

    import torchvision.transforms.functional as TVF
    no_rotate = type('T', (), {'hflip': staticmethod(TVF.hflip),
                               'vflip': staticmethod(TVF.vflip),
                               'rotate': staticmethod(lambda t, a, fill=None: t)})
    orig_TF, orig_shift = ds.TF, ds.shift

    def make_crop(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        return (ds.generate_diffuse_field(CROP, CROP, dcfg=dcfg) * 50.0).astype(np.float32)

    def run(seed, crop, nc, stellar_prob):
        random.seed(seed + 1); np.random.seed(seed + 1); torch.manual_seed(seed + 1)
        kw = {'diffuse_cfg': dcfg}
        if ucfg is not None: kw['upsample_cfg'] = ucfg
        if nc is not None:   kw['noise_cfg'] = nc
        return ds.process_noise_to_clean_patch(
            crop, True, TS, PS, FOV, CROP, add_point_sources=True,
            clean_prob=0.0,                    # keep the full contaminant chain
            stellar_prob=stellar_prob,
            **kw)

    def probe(name, kind):
        # The branch under test has to actually fire. stellar_prob is 0.0 in the
        # fiducial config, so force it for its own probe only -- it is an elif
        # against the diffuse branch, so forcing it globally would silence that.
        stellar_prob = 0.9 if name == 'generate_stellar_field' else \
            float(cfg.dataset.get('stellar_prob', 0.0))
        fired = in_target = in_cond = 0
        for s in range(args.seeds):
            crop = make_crop(s)
            got = {}
            for factor in (1.0, 0.0):
                if factor == 1.0:
                    ref.clear()
                orig = getattr(ds, name, None) if kind in (FIELD, PATCH) else None
                ds.apply_patch_asymmetric_arcsinh = pinned_norm
                ds.TF = no_rotate
                nc = None if ncfg is None else dict(ncfg)
                if kind == FIELD:
                    setattr(ds, name, (lambda o=orig, f=factor:
                                       (lambda *a, **k: o(*a, **k) * f))())
                elif kind == PATCH:
                    setattr(ds, name, (lambda o=orig, f=factor:
                                       (lambda p, *a, **k: p + f * (o(p, *a, **k) - p)))())
                elif name == 'post_term' and factor == 0.0 and nc:
                    nc['post_range'] = (0.0, 0.0)
                    nc['post_low_factor'] = (0.0, 0.0)
                elif name == 'subpixel_shift' and factor == 0.0:
                    ds.shift = lambda a, **k: a
                try:
                    got[factor] = run(s, crop, nc, stellar_prob)
                finally:
                    if orig is not None: setattr(ds, name, orig)
                    ds.apply_patch_asymmetric_arcsinh = orig_norm
                    ds.TF, ds.shift = orig_TF, orig_shift
            dy = np.abs(got[1.0][2].numpy() - got[0.0][2].numpy()).max()
            dc = np.abs(got[1.0][1].numpy() - got[0.0][1].numpy()).max()
            if dy > args.tol or dc > args.tol:
                fired += 1
                in_target += dy > args.tol
                in_cond += dc > args.tol
        return fired, in_target, in_cond

    print("=" * 78)
    print("ENDPOINT SPLIT AUDIT   (config: %s)" % os.path.basename(args.config))
    print("  SKY = in both endpoints, the target keeps it, the model sharpens it")
    print("  INSTRUMENT = condition only, the model removes it")
    print("=" * 78)
    print(f"{'component':<38}{'fired':>8}{'target':>8}{'cond':>7}{'verdict':>12}")
    problems = []
    for name, kind, expect, label in COMPONENTS:
        if kind in (FIELD, PATCH) and not hasattr(ds, name):
            print(f"{label:<38}{'absent from this datasets.py':>35}")
            continue
        fired, t, c = probe(name, kind)
        if fired == 0:
            verdict, ok = 'never fired', True     # e.g. a branch disabled by config
        elif t and c:
            verdict, ok = 'SKY', expect == 'SKY'
        elif c:
            verdict, ok = 'INSTRUMENT', expect == 'INSTRUMENT'
        else:
            verdict, ok = 'TARGET-ONLY', False    # always wrong
        print(f"{label:<38}{f'{fired}/{args.seeds}':>8}{t:>8}{c:>7}{verdict:>12}"
              + ('' if ok else f'   <-- EXPECTED {expect}'))
        if not ok:
            problems.append(f"{label}: {verdict}, expected {expect}")

    print()
    if problems:
        print("=" * 78)
        print("INVARIANT VIOLATED:")
        for p in problems:
            print("  * " + p)
        print("=" * 78)
        return 1
    print("=" * 78)
    print("INVARIANT HOLDS: every sky component is in both endpoints, every")
    print("instrument component is in the condition only, and nothing appears")
    print("in the target alone.")
    print("=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())