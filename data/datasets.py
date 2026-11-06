import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, shift, map_coordinates, zoom  # CHANGED v13
import torchvision.transforms.functional as TF

# =========================================================
# CHANGED v11: FIDUCIAL SWITCHES. Everything the fiducial run needs is set
# here, so neither factory.py nor the yaml has to change.
# =========================================================

# Mix weights (dust, SKIRT, FIRE) applied inside CombinedSyntheticDataset,
# overriding whatever factory.py passes. The fiducial model is trained without
# FIRE, so its weight is 0 and the remaining two are renormalized. factory.py
# still constructs the FIRE dataset; it simply never gets sampled. Set to None
# to defer to the fractions factory.py supplies.
DATASET_FRACTIONS_OVERRIDE = (0.50, 0.50, 0.00)

# CHANGED v11: GRAVITY-FREE DIFFUSE SAMPLES. Every source of ground truth in
# this pipeline contains self-gravity: the dust MHD cubes, the SKIRT galaxies
# and FIRE all form dense, high-contrast filament spines, and the v9 stellar
# fields and the CIB injection (p=0.95) add more compact structure on top. The
# network therefore learns that any elongated Planck-scale blob resolves into a
# concentrated fiber, when a large part of the real diffuse ISM is an extended
# undulation with no dense spine at all -- a bend, not a fiber.
#
# With probability DIFFUSE_PROB a sample is replaced by (or blended with) a
# field drawn from turbulence WITHOUT self-gravity: log-normal column density,
# no power-law high-density tail, optionally with wide low-contrast ridges.
# See generate_turbulent_column() for why that is the right null model.
DIFFUSE_PROB = 0.18

# Fraction of the diffuse samples that keep some of the real crop underneath
# rather than replacing it outright.
DIFFUSE_BLEND_PROB = 0.35

# CHANGED v11: the informativeness gate is itself part of the bias. It rejects
# crops whose structure is a small fraction of their brightness -- which is the
# signature of genuinely diffuse gas as much as of a dead pedestal. Letting a
# fraction through unjudged restores real low-contrast fields to the training
# set. Set to 0.0 to recover the old strict behavior.
DIFFUSE_BYPASS_PROB = 0.15

# CHANGED v12: DIFFUSE SAMPLES IN VALIDATION.
#
# Every dataset sets `augment = ... and (mode == 'train')`, and the whole
# diffuse branch lives inside `if augment:`. So the v11 work above changes the
# training distribution and is invisible to validation: the val set contains
# zero gravity-free samples, and no metric you watch can tell you whether the
# diffuse bias improved. You would train the fix blind.
#
# With this > 0, a fixed subset of validation patches is replaced by a diffuse
# field. Which patches, and the field they get, are both hashed from (file
# index, patch number), so the same val patches are diffuse every epoch and
# carry the same underlying structure. The instrument-noise realisation on top
# still varies between epochs -- that is pre-existing behaviour for every val
# patch in this pipeline, not something this switch introduces.
#
# DEFAULT 0.0 ON PURPOSE. Turning it on changes what the validation loss means,
# so a resumed run's val curve would step discontinuously and stop being
# comparable to its own history. Set it for a fresh run, e.g. 0.20, and treat
# the val number as a new baseline.
VAL_DIFFUSE_FRACTION = 0.0


# =========================================================
# CHANGED v13 (0825): DIFFUSE-FIRST RETRAINING.
#
# The 0818 model over-commits to compact sources and under-delivers on extended
# emission. Three things drive that, and all three are addressed here WITHOUT
# touching factory.py, main.py, the trainer or the model:
#
#  1. The stellar branch (v9) replaces the crop with a star/cluster/galaxy field
#     at stellar_prob. Set `stellar_prob: 0.0` in the yaml to switch it off
#     entirely; STELLAR_MODE_WEIGHTS below lets you drop only 'cluster' instead.
#
#  2. The diffuse branch fires too rarely (0.18) and its generator is a
#     monofractal log-normal: correct P(k), but Gaussian phases, no
#     intermittency, no field direction, no coherent structure. See
#     generate_turbulent_column_v2().
#
#  3. THE BEAM. act_fwhm is drawn 0.4-3.5 target pixels for every sample. At the
#     top of that range a diffuse field is smoothed over 3.5 target pixels, so
#     the GT itself carries no small-scale diffuse texture -- the network is
#     literally trained that "diffuse in -> smooth out", and the only thing left
#     worth sharpening is a compact peak. Diffuse samples now get their own,
#     much smaller, target beam so the turbulent cascade survives into the GT.
#     The condition-to-target ratio is a SEPARATE knob (see DIFFUSE_DEFAULTS):
#     the input is always resampled to planck_size, so the true SR factor is
#     fixed at target_size/planck_size ~ 5x whatever the beams do; the ratio
#     only sets whether the condition is beam-limited or pixel-limited.
#
# Everything below is read from `cfg.dataset` by each Dataset's __init__ and
# carried to process_noise_to_clean_patch in ONE dict, so it travels correctly
# to DataLoader workers under both fork and spawn. A config that does not set
# these keys reproduces v12 behaviour exactly.
# =========================================================

# Relative weights for generate_stellar_field's three modes. Set 'cluster' to
# 0.0 to keep field stars and galaxy disks but drop crowded cluster cores.
# Only consulted when the stellar branch actually fires (stellar_prob > 0).
STELLAR_MODE_WEIGHTS = {'field': 0.25, 'cluster': 0.40, 'galaxy': 0.35}

DIFFUSE_DEFAULTS = {
    # --- how often, and how ---
    'prob':             DIFFUSE_PROB,          # 0.18 -> raise for a diffuse run
    'blend_prob':       DIFFUSE_BLEND_PROB,    # keep some real crop underneath
    'bypass_prob':      DIFFUSE_BYPASS_PROB,   # crop-gate bypass
    'val_fraction':     VAL_DIFFUSE_FRACTION,  # hashed diffuse patches in val
    'bend_prob':        0.45,                  # explicit wide ridges on top

    # --- generator ---
    'turb_version':     1,        # 1 = v11 log-normal, 2 = v13 (see below)
    'gen_size':         384,      # v2 only: build at this grid, then upsample.
                                  # Must exceed target_size*1.5 or the GT loses
                                  # power it could otherwise represent.
    'beta':             (2.9, 3.6),   # REQUESTED P(k) exponent -- not the one
                                      # you measure; realises -3.11. See the note
                                      # above generate_turbulent_column_v2.
    'mach':             (0.7, 2.2),   # transonic: sets sigma_lnN ~ 0.29, which
                                      # is the measured Polaris value. NOT the
                                      # molecular-cloud 1.5-8 that v11 used.
    'anisotropy':       (1.0, 1.3),   # k-space axis ratio: B-aligned eddies
    'inject_k':         (0.8, 2.5),   # driving scale, in cycles per box
    'cascade_octaves':  4,
    'cascade_strength': 0.0,          # OFF: intermittency, but a texture real
                                      # cirrus lacks. (0.3, 0.8) to enable.
    'warp_frac':        0.0,          # OFF: phase coherence, but produces a
                                      # curled fingerprint look. (0.0, 0.025).

    # --- beams, diffuse samples only. None -> use the shared draw ---
    'act_fwhm':         None,     # e.g. (0.3, 1.6) target px  (shared: 0.4-3.5)
    'planck_ratio':     None,     # e.g. (3.0, 6.0)            (shared: 3.0-6.0)
    'val_act_fwhm':     1.0,      # fixed, so val stays reproducible
    'val_planck_fwhm':  5.0,
}

_DIFFUSE_RANGE_KEYS = ('beta', 'mach', 'anisotropy', 'inject_k',
                       'cascade_strength', 'warp_frac', 'act_fwhm', 'planck_ratio')


def diffuse_cfg_from(cfg):
    """Build the diffuse config dict from cfg.dataset, falling back to v12."""
    d = dict(DIFFUSE_DEFAULTS)
    ds = getattr(cfg, 'dataset', None)
    if ds is None:
        return d

    d['prob'] = float(ds.get('diffuse_prob', d['prob']))
    d['blend_prob'] = float(ds.get('diffuse_blend_prob', d['blend_prob']))
    d['bypass_prob'] = float(ds.get('diffuse_bypass_prob', d['bypass_prob']))
    d['val_fraction'] = float(ds.get('val_diffuse_fraction', d['val_fraction']))

    for k, v in (ds.get('diffuse') or {}).items():
        if k not in d:
            raise KeyError(f"unknown dataset.diffuse key '{k}'; "
                           f"valid keys: {sorted(DIFFUSE_DEFAULTS)}")
        d[k] = v

    # yaml gives lists; the generators want tuples (and None must survive)
    for k in _DIFFUSE_RANGE_KEYS:
        if isinstance(d[k], (list, tuple)):
            d[k] = tuple(float(x) for x in d[k])

    # Echo once, next to the [Mix] line, so the log says what actually ran
    global _DIFFUSE_ECHOED
    if not _DIFFUSE_ECHOED:
        _DIFFUSE_ECHOED = True
        beam = (f"act {d['act_fwhm']} x ratio {d['planck_ratio'] or (3.0, 6.0)}"
                if d['act_fwhm'] is not None else 'shared (0.4-3.5 x 3.0-6.0)')
        print(f"[Diffuse] prob {d['prob']:.2f} | blend {d['blend_prob']:.2f} | "
              f"gate-bypass {d['bypass_prob']:.2f} | val {d['val_fraction']:.2f} | "
              f"generator v{int(d['turb_version'])} | beam {beam} | stellar modes "
              f"{[m for m, wt in STELLAR_MODE_WEIGHTS.items() if wt > 0]}")
    return d


_DIFFUSE_ECHOED = False


def _u(span):
    """uniform() over a (lo, hi) pair, or the value itself if it is a scalar."""
    if isinstance(span, (list, tuple)):
        return random.uniform(float(span[0]), float(span[1]))
    return float(span)


def _u_log(span):
    """log-uniform over a (lo, hi) pair; the value itself if it is a scalar.

    CHANGED v15: for a range that spans a decade or more, uniform() puts most
    of its mass in the top half -- U(0.003, 0.03) is below 1% only a quarter of
    the time. A noise LEVEL is naturally logarithmic (a 0.3% map and a 3% map
    are one decade apart, not "0.027 apart"), so the low-noise draw is
    log-uniform and covers every decade evenly.
    """
    if isinstance(span, (list, tuple)):
        lo, hi = float(span[0]), float(span[1])
        if lo <= 0.0:
            raise ValueError(f"_u_log needs a positive lower bound, got {lo}")
        return float(np.exp(random.uniform(np.log(lo), np.log(hi))))
    return float(span)


def _mix64(x):
    """splitmix64 finalizer: avalanches every input bit across the output.

    A plain `index * A + patch * B` then `% N` does not. With patch numbers
    running 0..3 and indices sequential, the low bits stay correlated and the
    selected fraction comes out wrong -- measured 0.0 for a requested 0.10.
    """
    x = int(x) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def val_force_diffuse(index, patch_i, fraction=None):
    """Should this validation patch be a gravity-free diffuse field?

    Hashed rather than random, so the same patches are chosen every epoch.
    Returns (flag, seed); the seed also fixes the field that gets generated.
    """
    frac = VAL_DIFFUSE_FRACTION if fraction is None else fraction
    h = _mix64(int(index) * 0x9E3779B97F4A7C15 + int(patch_i) + 0x2545F491)
    if frac <= 0.0:
        return False, h & 0x7FFFFFFF
    return (h >> 11) / float(1 << 53) < frac, h & 0x7FFFFFFF


# =========================================================
# CHANGED v14 (0901): THE CONDITION'S UPSAMPLING KERNEL IS NOW A NUISANCE
# PARAMETER INSTEAD OF A FIXED PART OF THE FORWARD MODEL.
#
# The condition is beam-convolved, resampled DOWN to planck_size (51) and then
# back UP to target_size (256) before the network ever sees it. Through 0825
# that last step was always bicubic, so every training pair carried the same
# cubic-spline signature: the same ringing beside bright edges, the same smooth
# interpolant between the 51 real samples. Nothing stops the network from
# learning that signature and reading sub-beam structure straight out of it --
# which scores perfectly in validation, where the signature is identical, and
# degrades on a real map that some other pipeline regridded. Real conditions
# arrive reprojected by whatever the archive used: drizzle/reproject (roughly
# linear), a nearest-neighbour regrid, or a cubic spline.
#
# Drawing the order per sample takes the kernel out of the information the
# network can rely on. Orders follow the scipy/skimage convention:
#
#   0 -> nearest, i.e. pixel replication. Blocky; the 51 sampled values survive
#        exactly and nothing is invented between them. DO NOT USE -- see the
#        v17 finding below.
#   1 -> bilinear.
#   3 -> bicubic. What every run through 0825 used, and what inference uses.
#
# Weights are relative and normalized on load. THE DEFAULT BELOW IS
# BICUBIC-ONLY: a config that does not set `dataset.upsample` reproduces 0825
# exactly, same convention as v13.
#
# ---------------------------------------------------------------------------
# CHANGED v17 (0910): ORDER 0 IS OUT. IT MADE THE MODEL WORSE, AND THE REASON
# IS NOT THAT IT IS "UNPHYSICAL" -- IT INVERTS THE TASK.
#
# 0901 trained on {0: 1, 1: 1, 3: 1} and came out measurably worse than 0825
# when evaluated the way the model is actually used, with order 3. Measured
# power in the upsampled CONDITION as a fraction of the power in the TARGET the
# network must produce (40 turbulent fields, 8' -> 1' beam pair, 51 -> 256):
#
#     scale        order 0   order 1   order 3
#     20 px          0.34      0.27      0.43
#     10 px          0.01      0.00      0.01
#      6.7 px        0.30      0.02      0.00
#      5.0 px        0.35      0.00      0.00
#      2.9 px        1.32      0.01      0.00
#     median 2.5-20  0.282     0.008     0.004
#
# Orders 1 and 3 hand the network a near-EMPTY canvas at sub-beam scales (under
# 1% of the target power), so the task there is unambiguously GENERATE. Order 0
# fills that same band with ~28% of the target power -- 130% at 2.9 px -- as
# block edges that are spatially UNCORRELATED with the true structure. On those
# samples the sub-beam task is not "generate" but "destroy this, then generate
# something else". Those are opposite operations at the same spatial
# frequencies, and a third of the training set asking for suppression installs
# a high-k suppression prior. At inference with order 3 there is nothing left
# to suppress, but the prior still damps what the network itself generates,
# which reads directly as a reduced band-power recovery fraction F.
#
# Two distinct costs, worth keeping separate: order 0 is actively harmful as
# above, AND it dilutes -- only 1/3 of 0901's samples saw the kernel used at
# deployment. 0910 uses {1: 1, 3: 2}: order 1 stays as the robustness
# perturbation (both are smooth interpolants, and the realistic ambiguity in
# real data is reproject/drizzle ~ linear versus spline ~ cubic), while
# two thirds of training sits on the deployment kernel.
#
# If you are tempted to re-enable order 0, note that nearest-neighbour
# regridding of diffuse maps is rare in practice and is never used at
# inference here, so it buys robustness against a case that does not occur
# while teaching the opposite of super-resolution. upsample_cfg_from prints a
# warning if it is given a positive weight.
# ---------------------------------------------------------------------------
#
# NOT randomized, deliberately: the GT resample and the downsample to
# planck_size both stay bicubic+antialias. Those two stand for the physical
# measurement (beam, then detector sampling). Only the reprojection back onto
# the target grid is a processing choice made downstream of the telescope, so
# only it is a nuisance parameter.
# =========================================================

_ORDER_TO_MODE = {0: 'nearest-exact', 1: 'bilinear', 3: 'bicubic'}


def order_to_mode(order):
    """scipy-style interpolation order -> F.interpolate mode string.

    'nearest-exact' rather than 'nearest': plain 'nearest' in torch has a
    documented half-pixel left bias, while 'nearest-exact' matches scipy
    order=0 and skimage. safe_resize falls back on an older torch build.
    """
    o = int(order)
    if o not in _ORDER_TO_MODE:
        raise ValueError(f"interpolation order {order} not supported; use 0, 1 or 3")
    return _ORDER_TO_MODE[o]


UPSAMPLE_DEFAULTS = {
    # relative weights per order. This default IS the 0825 behaviour.
    'order_weights': {3: 1.0},
    # Validation. A FIXED order keeps the val curve comparable to 0825's.
    # None instead hashes the order from (file index, patch number): val then
    # covers all three kernels while staying identical every epoch, which is
    # what you want if you are watching robustness rather than continuity.
    # Either way the val number moves the moment this is not 3 -- new baseline.
    'val_order': 3,
}

_UPSAMPLE_ECHOED = False


def upsample_cfg_from(cfg):
    """Build the condition-upsampling dict from cfg.dataset.upsample.

    Returns a plain picklable dict (spawn-safe, like diffuse_cfg_from) whose
    weights are pre-normalized into ('_orders', '_cum') for cheap per-sample
    draws in the worker.
    """
    d = {'order_weights': dict(UPSAMPLE_DEFAULTS['order_weights']),
         'val_order': UPSAMPLE_DEFAULTS['val_order']}

    ds = getattr(cfg, 'dataset', None)
    up = (ds.get('upsample') or {}) if ds is not None else {}
    for k in up:
        if k not in d:
            raise KeyError(f"unknown dataset.upsample key '{k}'; "
                           f"valid keys: {sorted(d)}")

    if up.get('order_weights') is not None:
        d['order_weights'] = {int(k): float(v)
                              for k, v in dict(up['order_weights']).items()}
    if 'val_order' in up:
        d['val_order'] = None if up['val_order'] is None else int(up['val_order'])

    for o in d['order_weights']:
        order_to_mode(o)                    # raises on anything but 0, 1, 3
    if d['val_order'] is not None:
        order_to_mode(d['val_order'])

    orders = tuple(sorted(o for o, w in d['order_weights'].items() if w > 0))
    if 0 in orders:
        # Permitted, so the experiment stays repeatable, but never silent.
        print("[Upsample] WARNING: order 0 (nearest) has a positive weight. "
              "0901 showed this inverts the sub-beam task and degrades the "
              "model at inference (order 3); see the v17 block in "
              "data/datasets.py.")
    total = float(sum(w for w in d['order_weights'].values() if w > 0))
    if not orders or total <= 0.0:
        raise ValueError("dataset.upsample.order_weights needs at least one "
                         "positive weight")
    cum, acc = [], 0.0
    for o in orders:
        acc += d['order_weights'][o] / total
        cum.append(acc)
    cum[-1] = 1.0                           # guard against float drift
    d['_orders'], d['_cum'] = orders, tuple(cum)

    # Echo once, next to the [Mix] and [Diffuse] lines, so the log says what ran
    global _UPSAMPLE_ECHOED
    if not _UPSAMPLE_ECHOED:
        _UPSAMPLE_ECHOED = True
        mix = ', '.join(f"{o}={order_to_mode(o)} "
                        f"{d['order_weights'][o] / total:.2f}" for o in orders)
        val = ('hashed per patch' if d['val_order'] is None
               else f"fixed order {d['val_order']}")
        print(f"[Upsample] condition planck->target order mix: {mix} | "
              f"val {val}")
    return d


def pick_upsample_order(up_cfg, augment, seed=None):
    """Which interpolation order lifts the condition to the target grid here.

    Train (augment=True): drawn per sample from order_weights.
    Val: up_cfg['val_order'] when set, else hashed from `seed` -- the same
    per-(index, patch) hash val_force_diffuse returns, re-mixed so the kernel
    does not correlate with which val patches are diffuse. Hashed, so the same
    patch gets the same kernel every epoch.
    up_cfg None -> 3, i.e. any caller that has not been updated (the plotting
    notebook, older scripts) keeps the pre-0901 behaviour.
    """
    if up_cfg is None:
        return 3
    if (not augment) and up_cfg['val_order'] is not None:
        return int(up_cfg['val_order'])
    orders, cum = up_cfg['_orders'], up_cfg['_cum']
    if len(orders) == 1:
        # Single-order config: answer WITHOUT consuming a random number. The
        # draw below would otherwise advance the global RNG by one step even
        # when the outcome is foregone, which shifts every augmentation
        # decision after it (the flips, the rotation angle, the noise). With
        # this short-circuit a bicubic-only config -- i.e. any config that does
        # not set dataset.upsample, 0825 included -- leaves the random stream
        # exactly where it was and still reproduces bit for bit.
        return orders[0]
    if augment:
        u = random.random()
    else:
        h = _mix64(int(seed or 0) * 0x9E3779B97F4A7C15 + 0x6A09E667)
        u = (h >> 11) / float(1 << 53)
    for o, c in zip(orders, cum):
        if u < c:
            return o
    return orders[-1]


# =========================================================
# CHANGED v15 (0901): THE LOW-NOISE *OBSERVED* REGIME.
#
# The condition's instrument noise is injected at planck_size (51x51), i.e.
# after the downsample and before the upsample -- which is right, that is where
# a real detector samples. The problem is the LEVEL. Through v14 the ordinary
# training branch drew it from
#     local_std * U(0.05, 0.30) * U(0.4, 0.8)
# whose product has a HARD FLOOR at 2% of the map's own std and a median of
# ~12.6%. So a Planck-like sample with 1% noise essentially never occurred.
#
# Low-noise samples did exist, but only through clean_mode (clean_prob, 0.15),
# and clean_mode is not "a real map with little noise" -- it also switches OFF
# the CMB, the scanning stripes and the post-upsample term, because it exists to
# model a space telescope. Measured over the realised training mix, 93% of the
# samples with noise below 3% were clean_mode samples. The network has therefore
# never seen the case that matters for real data: the FULL contaminant chain
# (CMB, striping, CIB, the lot) with only low-level Gaussian detector noise.
#
# THE SECOND HALF OF THE PROBLEM, and the reason lowering the floor alone would
# not have worked. The post-upsample white term (1-3% of the upsampled map's
# std) is added at 256x256, so it does NOT pass through the interpolation and is
# white on the target grid. Against a low-res noise draw it dominates:
#
#     low-res noise      post-upsample term, relative to it (bicubic)
#       0.5%                     4.50x
#       1.0%                     2.34x
#       2.0%                     1.17x
#       5.0%                     0.46x
#      15.0%                     0.16x
#
# At 1% the condition's noise would be mostly white-at-256 -- exactly the
# texture a reprojected map does NOT have, and the opposite of what the sample
# is supposed to teach. The same applies to the stripes, which at 10-30% of
# local std would tower over a 1% noise floor. So the low-noise branch scales
# all three together: detector noise, stripes, and the post-upsample term.
#
# low_prob DEFAULTS TO 0.0, so a config that does not set dataset.noise keys
# reproduces v14/0825 exactly -- and note the branch deliberately does not
# consume a random number when it is off, so the augmentation RNG stream is
# untouched too.
# =========================================================

NOISE_DEFAULTS = {
    # --- the low-noise observed branch --------------------------------------
    'low_prob':        0.0,             # of the non-clean TRAIN samples
    'low_range':       (0.003, 0.03),   # BASE, drawn log-uniform (see _u_log).
                                        # Same quantity as the ordinary
                                        # U(0.05, 0.30) below it, so the two
                                        # branches are directly comparable: the
                                        # realised white sigma is 0.4-0.8x the
                                        # base, and with p=0.5 a sqrt(signal)
                                        # shot term adds up to 1.0x more.
                                        # MEASURED realised total noise for
                                        # this range: p5 0.22%, median 0.72%,
                                        # p95 2.39%, max 4.2% of the map's own
                                        # std (ordinary branch: p5 4.2%, median
                                        # 12.5%, hard floor 2.0%) --
                                        # i.e. it fills the 0-2% gap under the
                                        # ordinary branch's hard floor and
                                        # overlaps it at the top, leaving no
                                        # level the network never trains on.
    'low_stripe_range': (0.005, 0.03),  # striping on low-noise samples. Kept
                                        # ABSOLUTE on purpose: residual striping
                                        # is a scanning systematic, not a
                                        # multiple of the white noise, and a
                                        # real map can be stripe-dominated at
                                        # low noise. (ordinary: 0.1-0.3)
    'post_low_factor': (0.05, 0.2),     # post-upsample white term on low-noise
                                        # samples, as a MULTIPLE of the level
                                        # drawn above -- not an absolute range.
                                        # An absolute range leaves a corner
                                        # (lowest noise x highest term) where
                                        # the white-at-256 term still dominates
                                        # the very texture the sample teaches:
                                        # measured 0.41x at a 0.5% draw, and
                                        # worse below that. Scaling it with the
                                        # sample's own level pins the ratio at
                                        # 0.08-0.33x, which is where the
                                        # ordinary regime already sits (0.16x
                                        # measured at 15% noise).

    # --- correlated (non-white) detector noise ------------------------------
    # CHANGED v16: until now the 2D noise was pure WHITE (measured lag-1
    # autocorrelation -0.004). Real low-resolution data is not: scanning
    # instruments leave 1/f-correlated noise, and a network trained only on
    # white noise reads correlated noise blobs as real structure and
    # super-resolves them into filaments that are not there. Both generators
    # needed for this were already in this file and had NO call sites --
    # generate_synthetic_act_noise and generate_clumpy_pepper_noise.
    #
    # The split is VARIANCE-PRESERVING: corr_frac of the white budget's
    # variance is moved into the correlated field, so the realised noise LEVEL
    # distribution (and therefore every SNR statistic) is unchanged and only
    # the noise MORPHOLOGY broadens.
    'corr_prob':      0.0,              # probability the noise has a
                                        # correlated component at all
    'corr_frac':      (0.2, 0.6),       # fraction of the noise VARIANCE that
                                        # is correlated rather than white
    'corr_knee':      (800.0, 1200.0),  # ell knee -> 0.9-4.3 px correlation
                                        # length on the 51x51 condition grid,
                                        # i.e. 0.10-0.47x the CMB scale there:
                                        # a spread from barely-correlated up to
                                        # clearly correlated, all of it clear of
                                        # the CMB. Do NOT raise this: 1400
                                        # already reaches 0.64x and 3000 is
                                        # indistinguishable. The transition is
                                        # sharp between 800 and 1400 -- see the
                                        # table in the docstring of
                                        # generate_synthetic_act_noise.
    'corr_alpha':     3.0,              # slope of the 1/f boost below the knee

    # High-frequency speckle, the other unused generator. OFF by default and
    # riskier than the correlated component: at ~1 px it lives where the CIB
    # and the injected point sources live, and THOSE are in both endpoints, so
    # it recreates the same keep-or-remove ambiguity at the opposite end of the
    # spectrum. Turn it on only if your real data actually shows pixel-scale
    # speckle, and look at the samples if you do.
    'pepper_prob':    0.0,
    'pepper_frac':    (0.05, 0.2),      # fraction of the noise variance

    # --- the pre-existing (ordinary) levels, unchanged ----------------------
    'stripe_range':    (0.1, 0.3),      # v3 value
    'post_range':      (0.01, 0.03),    # v3 value

    # --- validation ---------------------------------------------------------
    'val_level':       0.02,            # v3 value: fixed 2% of local std
    'val_low_fraction': 0.0,            # hashed fraction of val patches that
                                        # instead get val_low_level. > 0 makes
                                        # the low-noise regime visible to the
                                        # val metric -- and moves the baseline,
                                        # exactly like val_diffuse_fraction did.
    'val_low_level':   0.01,            # FIXED, not drawn: val stays identical
                                        # every epoch

    # --- accepted but unused ------------------------------------------------
    'm2_noise_level':  0.01,            # DEAD. Present in every yaml since
                                        # 0701 and read by nothing; listed here
                                        # only so the strict key check below
                                        # does not reject existing configs.
}

_NOISE_RANGE_KEYS = ('low_range', 'low_stripe_range', 'post_low_factor',
                     'stripe_range', 'post_range', 'corr_frac', 'corr_knee',
                     'pepper_frac')

_NOISE_ECHOED = False


def noise_cfg_from(cfg):
    """Build the condition-noise dict from cfg.dataset.noise. Picklable."""
    d = dict(NOISE_DEFAULTS)
    ds = getattr(cfg, 'dataset', None)
    for k, v in ((ds.get('noise') or {}) if ds is not None else {}).items():
        if k not in d:
            raise KeyError(f"unknown dataset.noise key '{k}'; "
                           f"valid keys: {sorted(NOISE_DEFAULTS)}")
        d[k] = v

    for k in _NOISE_RANGE_KEYS:            # yaml gives lists; _u wants tuples
        if isinstance(d[k], (list, tuple)):
            d[k] = tuple(float(x) for x in d[k])
    for k in ('low_prob', 'val_level', 'val_low_fraction', 'val_low_level',
              'corr_prob', 'corr_alpha', 'pepper_prob'):
        d[k] = float(d[k])
    for k in ('low_prob', 'corr_prob', 'pepper_prob'):
        if not 0.0 <= d[k] <= 1.0:
            raise ValueError(f"dataset.noise.{k} must be in [0, 1], got {d[k]}")
    for k in ('corr_frac', 'pepper_frac'):
        if not 0.0 <= d[k][0] <= d[k][1] <= 1.0:
            raise ValueError(f"dataset.noise.{k} must be an ordered pair inside "
                             f"[0, 1] (it is a fraction of the noise variance), "
                             f"got {d[k]}")

    global _NOISE_ECHOED
    if not _NOISE_ECHOED:
        _NOISE_ECHOED = True
        if d['low_prob'] > 0.0:
            lo, hi = d['low_range']
            what = (f"{d['low_prob']:.2f} of non-clean train samples at "
                    f"{lo * 100:.1f}-{hi * 100:.1f}% (log-uniform)")
        else:
            what = "OFF (ordinary 2-24% floor only)"
        corr = (f"{d['corr_prob']:.2f} at knee {d['corr_knee'][0]:.0f}-"
                f"{d['corr_knee'][1]:.0f}, {d['corr_frac'][0]:.2f}-"
                f"{d['corr_frac'][1]:.2f} of variance"
                if d['corr_prob'] > 0.0 else "OFF (white only)")
        print(f"[Noise] correlated component: {corr}"
              + (f" | pepper {d['pepper_prob']:.2f}" if d['pepper_prob'] > 0 else ""))
        print(f"[Noise] low-noise observed branch: {what} | val "
              f"{d['val_level'] * 100:.0f}%"
              + (f", {d['val_low_fraction']:.2f} hashed at "
                 f"{d['val_low_level'] * 100:.1f}%" if d['val_low_fraction'] > 0 else ""))
    return d


def draw_low_noise(noise_cfg, augment, clean_mode):
    """Draw the low-noise base level for this sample, or None if it is not one.

    Train-only, and never with clean_mode (the space-telescope regime, already
    noiseless). Returning the LEVEL rather than a flag is what lets the
    post-upsample term scale with it instead of being an independent range --
    see post_low_factor for why that matters.

    Consumes NO random number when low_prob is 0, so switching the branch off
    leaves the augmentation RNG stream exactly where v14 had it.
    """
    if noise_cfg is None or not augment or clean_mode:
        return None
    if noise_cfg['low_prob'] <= 0.0 or random.random() >= noise_cfg['low_prob']:
        return None
    return _u_log(noise_cfg['low_range'])


# =========================================================
# STEP-BY-STEP DEBUGGER & HELPERS
# =========================================================
def check_finite(name, arr):
    if not np.isfinite(arr).all():
        nans = np.isnan(arr).sum()
        infs = np.isinf(arr).sum()
        raise ValueError(f"🚨 FAILED AT [{name}]! NaNs: {nans}, Infs: {infs}")
    return arr

# CHANGED v5: statistics subsampling. Percentile-based gates/conditioning on
# crops up to 1536^2, repeated across retries, were a measurable cost. All
# statistics below are computed on a strided view capped at ~384^2 samples,
# which is far more than enough for stable percentiles, and applied to the
# full-resolution crop.
def _stat_view(patch, max_side=384):
    s = max(1, patch.shape[0] // max_side, patch.shape[1] // max_side)
    return patch[::s, ::s]


# CHANGED v2: shared crop quality gate, used by BOTH SkirtAstroDataset and
# FireQuicklookDataset. Rejects two degenerate crop types that produce
# delta-function target histograms and contaminant-dominated GTs:
#   (a) pedestal crops: bright but featureless (large median, tiny deviation)
#   (b) empty crops: almost no pixels above the local floor
def crop_is_informative(patch, min_rel_struct=0.15, min_filled_frac=0.01,
                        bypass_prob=None):
    # CHANGED v11: admit a fraction of crops without judging them. Test (a)
    # below cannot tell a dead pedestal from a real low-contrast diffuse field,
    # so rejecting on it systematically strips diffuse morphology out of the
    # training set -- exactly the structure the model is then unable to predict.
    bypass = DIFFUSE_BYPASS_PROB if bypass_prob is None else bypass_prob
    if bypass > 0.0 and random.random() < bypass:
        return True

    patch = _stat_view(np.asarray(patch, dtype=np.float64))  # CHANGED v5
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

    med = np.median(patch)
    dev = np.abs(patch - med)
    p995_dev = np.percentile(dev, 99.5)

    # Overall brightness scale: whichever is larger, the pedestal level or
    # the structural amplitude itself
    scale = max(abs(med), np.percentile(np.abs(patch), 99.5), 1e-12)

    # (a) Pedestal test: structure must be a non-negligible fraction of the
    #     brightness scale. A flat bright floor fails here.
    if (p995_dev / scale) < min_rel_struct:
        return False

    # (b) Occupancy test: enough pixels meaningfully above the floor.
    #     A patch with only a handful of hot pixels fails here.
    n_above = np.count_nonzero(patch > med + 0.05 * p995_dev)
    if n_above < min_filled_frac * patch.size:
        return False

    return True


# CHANGED v4: continuous structure score (same quantity the gate thresholds).
# Used to keep the BEST candidate when all retries fail, instead of blindly
# accepting the last (possibly flat) crop.
def crop_structure_score(patch):
    patch = np.nan_to_num(_stat_view(np.asarray(patch, dtype=np.float64)), nan=0.0, posinf=0.0, neginf=0.0)  # CHANGED v5
    med = np.median(patch)
    p995_dev = np.percentile(np.abs(patch - med), 99.5)
    scale = max(abs(med), np.percentile(np.abs(patch), 99.5), 1e-12)
    return p995_dev / scale


# CHANGED v4: conditioning for extreme-dynamic-range crops (SKIRT bands).
# Two operations, mirroring the FIRE treatment:
#   1. Pedestal removal: SKIRT bands dominated by smooth continuum are
#      bright-but-flat; subtracting the faint-end floor collapses them to
#      ~zero so the informativeness gate can reject them correctly.
#   2. Randomized dynamic-range compression: a compact nucleus 3-4 dex above
#      the diffuse emission otherwise sets local_max and the arcsinh
#      headroom, crushing all diffuse structure into a single histogram bin.
#      Clipping at (bulk emission) x uniform(30, 300) bounds the peak-to-
#      diffuse ratio at ~1.5-2.5 dex, which the arcsinh handles comfortably,
#      and the randomization doubles as brightness-contrast augmentation.
def prepare_extreme_dr_crop(crop, pedestal_percentile=5.0, dr_clip_range=(30.0, 300.0)):
    crop = np.nan_to_num(np.asarray(crop, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    sv = _stat_view(crop)  # CHANGED v5: stats on subsample, ops on full crop
    pos = sv[sv > 0]
    if pos.size > 0:
        pedestal = np.percentile(pos, pedestal_percentile)
        crop = np.clip(crop - pedestal, 0.0, None)

    sv = _stat_view(crop)
    pos = sv[sv > 0]
    if pos.size > 0:
        bulk = np.percentile(pos, 90.0)
        if bulk > 0:
            ceiling = bulk * random.uniform(*dr_clip_range)
            np.clip(crop, 0.0, ceiling, out=crop)

    return crop.astype(np.float32)


def apply_patch_asymmetric_arcsinh(patch, transition_sigmas=3.0, headroom_multiplier=2.5, max_sigma=10.0):
    # CHANGED v8: reverted the v7 median-centering. In emission-dominated
    # patches the median sits at the NOISE floor; centering there allocates
    # the linear regime of the stretch to the noise instead of the emission.
    # The zero anchor keeps background noise compressed at the bottom of the
    # normalized range while the (large-dynamic-range) emission spans it.
    c = 0.0
    s_raw = np.nanstd(patch)

    # Floor the stretch scale relative to the patch amplitude.
    # On near-constant (pedestal) patches the raw std is dominated by injected
    # telescope noise, producing a degenerate arcsinh stretch that collapses
    # the whole patch into one histogram bin. Flooring s at a small fraction
    # of the absolute amplitude keeps the mapping well-conditioned.
    abs_scale = np.nanmax(np.abs(patch))
    if not np.isfinite(abs_scale) or abs_scale == 0:
        abs_scale = 1.0
    s = max(s_raw, 0.02 * abs_scale, 1e-8)

    alpha = s * transition_sigmas
    tf_patch = np.arcsinh(patch / alpha)

    abs_min, abs_max = np.nanmin(patch), np.nanmax(patch)
    perc_min, perc_max = np.nanpercentile(patch, 0.1), np.nanpercentile(patch, 99.9)
    sigma_min, sigma_max = c - max_sigma * s, c + max_sigma * s

    p_min = max(abs_min, min(perc_min, sigma_min))
    p_max = min(abs_max, max(perc_max, sigma_max))

    max_phys = headroom_multiplier * max(0, p_max)
    min_phys = headroom_multiplier * min(0, p_min)

    t_min = np.arcsinh(min_phys / alpha)
    t_max = np.arcsinh(max_phys / alpha)
    if t_max - t_min < 1e-6: t_max = t_min + 1.0

    final_norm = np.clip((tf_patch - t_min) / (t_max - t_min), 0.0, 1.0)
    return final_norm, torch.tensor([c, alpha, t_min, t_max], dtype=torch.float32)


def safe_resize(img_np, size, mode='bicubic', antialias=True, order=None):
    """
    Replaces PIL.Image.resize to strictly enforce anti-aliasing.
    Prevents high-frequency grid artifacts during downsampling.

    CHANGED v14: `order` overrides `mode` with the scipy/skimage convention
    (0 nearest, 1 bilinear, 3 bicubic), so a caller can select the kernel from
    a number drawn at runtime -- see UPSAMPLE_DEFAULTS. `antialias` is only
    defined for bilinear/bicubic in F.interpolate and is dropped for nearest;
    every nearest call here is an UPSAMPLE, where a low-pass prefilter is not
    defined anyway.
    """
    if order is not None:
        mode = order_to_mode(order)
    t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    if mode.startswith('nearest'):
        try:
            t = F.interpolate(t, size=(size, size), mode=mode)
        except (ValueError, NotImplementedError, RuntimeError):
            # 'nearest-exact' predates torch 1.11; plain nearest is the only
            # option there and differs only by a half-pixel bias.
            t = F.interpolate(t, size=(size, size), mode='nearest')
    else:
        t = F.interpolate(t, size=(size, size), mode=mode, antialias=antialias)
    return t.squeeze(0).squeeze(0).numpy()

def inject_point_sources(base_signal, max_sources=15, pixel_scale_arcmin=0.5, aa_sigma=0.0):
    if random.random() < 0.3: return base_signal

    h, w = base_signal.shape
    signal_max = np.nanmax(base_signal)
    if np.isnan(signal_max) or signal_max <= 0: signal_max = 1e-3

    ps_map = np.zeros_like(base_signal)

    if random.random() < 0.5:
        num_sources = random.randint(1, 4)
        for _ in range(num_sources):
            y, x = random.randint(0, h-1), random.randint(0, w-1)
            ps_map[y, x] = signal_max * random.uniform(0.5, 2.0)
    else:
        num_sources = random.randint(1, max_sources)
        for _ in range(num_sources):
            y, x = random.randint(0, h-1), random.randint(0, w-1)
            ps_map[y, x] = signal_max * random.uniform(1.0, 10.0)

        ps_max = np.nanmax(ps_map)
        if ps_max > 0:
            ps_map = ps_map * (signal_max * random.uniform(2.0, 8.0) / ps_max)

    # CHANGED v10: band-limit bright deltas to the target sampling (see
    # stellar-mode comment); renormalize so peak amplitudes are preserved.
    if aa_sigma > 0.3:
        pre_peak = np.nanmax(ps_map)
        if pre_peak > 0:
            ps_map = gaussian_filter(ps_map, sigma=aa_sigma, truncate=4.0)
            post_peak = np.nanmax(ps_map)
            if post_peak > 0:
                ps_map *= (pre_peak / post_peak)

    return base_signal + ps_map


def generate_cmb_grf(h, w, pixel_scale_arcmin):
    pixel_scale_rad = pixel_scale_arcmin * (np.pi / 180.0 / 60.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    ell = 2 * np.pi * f / pixel_scale_rad
    ell[0, 0] = 1.0

    base_power = 1.0 / (ell**2 + 500.0)
    oscillation = np.cos(np.pi * (ell - 220.0) / 300.0)**2
    baryon_dip = 1.0 - 0.55 * np.exp(-0.5 * ((ell - 540.0) / 120.0)**2)
    damping = np.exp(-(ell / 1300.0)**1.5)

    c_ell = base_power * (1.0 + 2.5 * oscillation * baryon_dip) * damping
    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    cmb_map = np.real(np.fft.ifft2(white_noise * np.sqrt(c_ell)))
    return (cmb_map - np.nanmean(cmb_map)) / (np.nanstd(cmb_map) + 1e-8)

def generate_diffuse_cirrus(h, w, pixel_scale_arcmin):
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0

    power_spectrum = 1.0 / (f ** 2.7)
    power_spectrum += 0.05
    power_spectrum[0, 0] = 0.0

    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    cirrus = np.real(np.fft.ifft2(white_noise * np.sqrt(power_spectrum)))
    cirrus = (cirrus - np.nanmean(cirrus)) / (np.nanstd(cirrus) + 1e-8)

    cirrus = np.exp(cirrus * 0.8)
    cirrus = (cirrus - np.nanmin(cirrus)) / (np.nanmax(cirrus) - np.nanmin(cirrus) + 1e-8)
    return cirrus.astype(np.float32)


# CHANGED v11: COLUMN DENSITY OF TURBULENCE WITHOUT SELF-GRAVITY.
#
# The defining property of this generator is what it does *not* produce. In
# isothermal supersonic turbulence with no self-gravity the density PDF is
# log-normal, with log-density width set by the sonic Mach number,
#
#     sigma_s^2 = ln(1 + b^2 M^2)
#
# (b ~ 1/3 solenoidal forcing, ~1 compressive, ~0.4 typical mixed). Once
# self-gravity switches on, collapsing regions add a power-law high-density
# tail and narrow, high-contrast filament spines. Every simulation feeding this
# pipeline has that tail. A pure log-normal has neither the tail nor the
# spines, so a sample drawn here is unambiguously diffuse structure: the model
# sees an extended, low-contrast field that must stay extended and low-contrast
# when sharpened, rather than resolving into a fiber.
#
# Projecting a 3-D log-normal along the line of sight narrows the distribution,
# since independent cells average; the factor here (~2 in sigma) is the usual
# rough approximation, not an exact result. The power-law index beta is for the
# *column density* power spectrum: 2.4-3.2 spans the observed diffuse-ISM
# range, with Planck/IRAS cirrus sitting near 2.7-2.9 -- the same slope the
# existing generate_diffuse_cirrus() hard-codes as 2.7.
def generate_turbulent_column(h, w, beta=None, mach=None, b_forcing=None,
                              projection_narrowing=2.0):
    beta = random.uniform(2.4, 3.2) if beta is None else beta
    mach = random.uniform(1.5, 8.0) if mach is None else mach
    b_forcing = random.uniform(0.3, 0.6) if b_forcing is None else b_forcing

    sigma_s = np.sqrt(np.log(1.0 + (b_forcing * mach) ** 2))
    sigma_ln_n = sigma_s / max(projection_narrowing, 1e-6)

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0

    power = f ** (-beta)
    power[0, 0] = 0.0                       # no DC: the mean is set below

    white = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    g = np.real(np.fft.ifft2(white * np.sqrt(power)))
    g = (g - np.nanmean(g)) / (np.nanstd(g) + 1e-8)

    # exp(sigma*g - sigma^2/2) has unit mean, so the field is a column density
    # contrast rather than an arbitrary scale
    col = np.exp(sigma_ln_n * g - 0.5 * sigma_ln_n ** 2)
    return col.astype(np.float32)


# CHANGED v11: WIDE, LOW-CONTRAST RIDGES -- the "bend, not fiber" morphology.
#
# A self-gravitating filament is narrow (a fraction of the beam once observed),
# has a cusped Plummer-like cross-section and stands 1-2 dex above its
# surroundings. What this draws instead is a smoothly curving enhancement of
# the ambient medium: a Gaussian cross-section several percent of the box wide,
# peaking a factor of a few above the background. Blurring to the Planck beam
# and sharpening back should return the same broad undulation, not a spine.
def generate_diffuse_bend(h, w, n_bends=None):
    canvas = np.zeros((h, w), dtype=np.float32)
    n = random.randint(1, 3) if n_bends is None else n_bends
    span = float(max(h, w))
    t = np.linspace(0.0, 1.0, int(4 * span))

    for _ in range(n):
        ang = random.uniform(0.0, np.pi)
        amp = random.uniform(0.03, 0.18) * span
        k1, k2 = random.uniform(0.4, 1.2), random.uniform(1.5, 3.5)
        ph1, ph2 = random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)

        u = (t - 0.5) * 1.8 * span                                   # along
        v = amp * (np.sin(2 * np.pi * k1 * t + ph1)
                   + 0.4 * np.sin(2 * np.pi * k2 * t + ph2))         # across
        cy = h * random.uniform(0.25, 0.75)
        cx = w * random.uniform(0.25, 0.75)
        ys = cy + u * np.sin(ang) + v * np.cos(ang)
        xs = cx + u * np.cos(ang) - v * np.sin(ang)

        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        if not np.any(ok):
            continue
        line = np.zeros((h, w), dtype=np.float32)
        np.add.at(line, (ys[ok].astype(np.int64), xs[ok].astype(np.int64)), 1.0)

        # WIDE on purpose: a few percent of the box, so the ridge survives the
        # Planck beam as a resolved undulation instead of a point-like spine
        width = random.uniform(0.02, 0.08) * min(h, w)
        line = gaussian_filter(line, sigma=width, truncate=3.0)
        peak = line.max()
        if peak > 0:
            canvas += (line / peak) * random.uniform(0.3, 1.0)

    return canvas


# =========================================================
# CHANGED v13: TURBULENT COLUMN, SECOND GENERATION.
#
# TUNED AGAINST REAL DATA, NOT AGAINST INTUITION. The parameters below were fit
# to Herschel 350um maps of the Polaris Flare (Herschel_test/), the cleanest
# diffuse, non-star-forming cirrus field available here, measured on eight
# 384^2 fully-covered crops. That fit is the whole reason this function exists
# in its present form, and it overturned two assumptions:
#
#   1. v11's CONTRAST WAS ABOUT TWICE TOO HIGH. Real Polaris cirrus has
#      sigma_lnN ~ 0.30 (spread 0.24-0.47 across crops). v11 drew Mach 1.5-8.0,
#      which via sigma_s^2 = ln(1 + b^2 M^2) and the projection factor gives
#      sigma_lnN ~ 0.5-0.7. Mach 1.5-8 is the MOLECULAR CLOUD range; the diffuse
#      cirrus this branch is meant to represent is transonic, M ~ 1-2. Getting
#      this wrong is what made every synthetic diffuse field look punchier than
#      the sky, and it also pushed the flux in the brightest 1% of pixels to
#      3.9% against the real 2.4%.
#
#   2. v11's SPECTRUM WAS TOO SHALLOW. Real Polaris gives a P(k) slope of -3.06
#      to -3.21 over k = 12-90 cycles per box; v11 realised -2.55.
#
# Fixing those two, plus giving the spectrum an explicit driving scale so a
# single box-crossing gradient cannot dominate the field, reproduces the real
# statistics to within a few percent:
#
#     quantity (384^2 crops)     REAL Polaris    v11      v13
#     P(k) slope, k=12-90           -3.06       -2.55    -3.11
#     sigma_lnN                      0.30        0.49     0.29
#     flux in brightest 1%           2.2%        3.3%     2.1%
#     power-spectrum axis ratio      1.17        1.03     1.17
#     zeta(6)/zeta(3)                1.89        2.00     2.01
#     cost per field (768^2)          --         54 ms    31 ms
#
# WHY THE CASCADE AND THE WARP DEFAULT TO OFF. Both are implemented below and
# both do what the textbook says: the random-beta cascade makes the field
# intermittent (zeta(6)/zeta(3) 2.01 -> 1.91, against She-Leveque 1.78), and the
# advective warp gives it phase coherence, which a Gaussian-phase field lacks.
# They can be matched to Polaris on every scalar statistic in that table. And
# they still look wrong: side by side with Polaris the warped field shows a
# fine curled, fingerprint-like texture that real cirrus simply does not have.
# The scalar statistics above cannot see it, which is a useful warning about
# tuning a generator on summary statistics alone -- LOOK AT THE FIELDS.
#
# So they are kept, documented, and off:
#     cascade_strength: 0.0   ->  set (0.3, 0.8) to switch intermittency on
#     warp_frac:        0.0   ->  set (0.0, 0.025) to switch coherence on
# If you turn either on, re-measure the slope (see the beta note below) and
# re-render plot_diffuse_samples.py before trusting it.
#
# What remains, deliberately, is what a gravity-free null model should be: a
# log-normal column with no power-law high-density tail and no filament spines,
# so the network sees an extended, low-contrast field that must stay extended
# and low-contrast when sharpened. Real Polaris shows more coherent bright lanes
# than this produces; that is the self-gravitating structure the MHD cubes and
# SKIRT already supply, and it is not this branch's job.
#
# READ THIS BEFORE CHANGING `beta`. `beta` is the exponent REQUESTED of the
# anisotropic 2-D spectrum, and it is NOT the slope you measure on the output.
# The k-space ellipse flattens the angle-averaged spectrum (an anisotropic field
# genuinely does have a shallower angle average -- physics, not a bug), and the
# k_inj knee sits inside any practical fit range. Measured: beta 2.9-3.6
# realises -3.11, which is the Polaris value. Re-measure if you change
# anisotropy, inject_k, or either of the two switches above.
def _rgrf(h, w, power_half):
    """Unit-variance real GRF from a half-spectrum |k| power array."""
    wr = np.random.normal(0, 1, power_half.shape).astype(np.float32)
    wi = np.random.normal(0, 1, power_half.shape).astype(np.float32)
    spec = (wr + 1j * wi).astype(np.complex64) * np.sqrt(power_half).astype(np.float32)
    g = np.fft.irfft2(spec, s=(h, w))
    g = g - g.mean()
    return g / (g.std() + 1e-8)


def _half_k(h, w, pa=0.0, axis_ratio=1.0):
    """|k| on the rfft2 half-grid, in cycles per box, on a rotated ellipse."""
    fy = np.fft.fftfreq(h).astype(np.float32)[:, None] * h
    fx = np.fft.rfftfreq(w).astype(np.float32)[None, :] * w
    kpar = fx * np.cos(pa) + fy * np.sin(pa)         # along B
    kperp = -fx * np.sin(pa) + fy * np.cos(pa)       # across B
    k = np.sqrt((kpar / axis_ratio) ** 2 + kperp ** 2).astype(np.float32)
    k[0, 0] = 1e-3
    return k


def generate_turbulent_column_v2(h, w, beta=(2.9, 3.6), mach=(0.7, 2.2), b_forcing=None,
                                 projection_narrowing=2.0,
                                 anisotropy=(1.0, 1.3), position_angle=None,
                                 inject_k=(0.8, 2.5),
                                 cascade_octaves=4, cascade_strength=0.0,
                                 warp_frac=0.0, gen_size=None):
    # Same physics anchor as v11: Mach sets the log-normal width, projection
    # along the line of sight narrows it.
    beta = random.uniform(2.9, 3.6) if beta is None else _u(beta)
    mach = random.uniform(0.7, 2.2) if mach is None else _u(mach)
    b_forcing = random.uniform(0.3, 0.6) if b_forcing is None else _u(b_forcing)
    sigma_s = np.sqrt(np.log(1.0 + (b_forcing * mach) ** 2))
    sigma_target = sigma_s / max(projection_narrowing, 1e-6)

    # Build on a coarser grid and upsample. The crop is beam-convolved and
    # resampled to target_size anyway, so any power above ~target_size/2 cycles
    # per box is discarded downstream; gen_size only has to clear that.
    if gen_size and h == w and h > int(gen_size):
        gh = gw = int(gen_size)
    else:
        gh, gw = h, w

    A = _u(anisotropy)
    pa = random.uniform(0.0, np.pi) if position_angle is None else float(position_angle)
    k_inj = _u(inject_k)

    k = _half_k(gh, gw, pa, A)
    power = (k ** 2 + k_inj ** 2) ** (-0.5 * beta)   # flat below k_inj, k^-beta above
    power[0, 0] = 0.0
    s = _rgrf(gh, gw, power)                          # log-column, unit variance

    n_oct = int(cascade_octaves)
    if n_oct > 0:
        amp = _u(cascade_strength)
        wj = np.array([2.0 ** (j * (2.0 - beta) / 2.0) for j in range(n_oct)])
        wj *= amp / np.sqrt(np.sum(wj ** 2))
        logk = np.log(np.maximum(k, 1e-3))
        for j in range(n_oct):
            band = np.exp(-0.5 * ((logk - np.log(k_inj * 2.0 ** (j + 1))) / 0.45) ** 2)
            band[0, 0] = 0.0
            gj = _rgrf(gh, gw, band)
            q = random.uniform(0.1, 0.9)              # filling factor of this octave
            u = 1.0 / (1.0 + np.exp(-(gj - q) * 2.2))  # smooth sparse support
            s += wj[j] * (u - u.mean()) / (u.std() + 1e-8)

    frac = _u(warp_frac)
    if frac > 0:
        disp = min(gh, gw) * frac
        nl = 48                                        # displacement field is
        kl = _half_k(nl, nl)                           # smooth: build it tiny
        low = (kl ** 2 + 4.0) ** (-2.0)
        low[0, 0] = 0.0
        dy = zoom(_rgrf(nl, nl, low), (gh / nl, gw / nl), order=1)[:gh, :gw] * disp
        dx = zoom(_rgrf(nl, nl, low), (gh / nl, gw / nl), order=1)[:gh, :gw] * disp
        yy, xx = np.mgrid[0:gh, 0:gw]
        s = map_coordinates(s, [yy + dy, xx + dx], order=1, mode='reflect')

    if (gh, gw) != (h, w):
        s = safe_resize(np.ascontiguousarray(s, dtype=np.float32), max(h, w),
                        mode='bicubic', antialias=False)

    # Pin the contrast to the Mach-set width. Rescaling in LOG space keeps the
    # non-Gaussian shape (and therefore the intermittency) intact.
    s = (s - s.mean()) / (s.std() + 1e-8) * sigma_target
    col = np.exp(np.clip(s, -20.0, 20.0))
    col /= col.mean()                                  # unit-mean column contrast
    return col.astype(np.float32)


# CHANGED v11: the composed gravity-free field handed to the training pipeline.
# CHANGED v13: dcfg selects the generator and its parameters; dcfg=None keeps
# the exact v11/v12 behaviour.
def generate_diffuse_field(h, w, bend_prob=0.6, dcfg=None):
    if dcfg is None or int(dcfg.get('turb_version', 1)) < 2:
        field = generate_turbulent_column(h, w)
        if random.random() < (bend_prob if dcfg is None else dcfg.get('bend_prob', bend_prob)):
            # peak contrast of a few over the ambient medium -- an order of
            # magnitude below a self-gravitating spine
            field = field * (1.0 + random.uniform(0.5, 2.5)
                             * generate_diffuse_bend(h, w))
        return field.astype(np.float32)

    field = generate_turbulent_column_v2(
        h, w,
        beta=dcfg.get('beta', (2.9, 3.6)),
        mach=dcfg.get('mach', (0.7, 2.2)),
        anisotropy=dcfg.get('anisotropy', (1.0, 1.3)),
        inject_k=dcfg.get('inject_k', (0.8, 2.5)),
        cascade_octaves=dcfg.get('cascade_octaves', 4),
        cascade_strength=dcfg.get('cascade_strength', 0.0),
        warp_frac=dcfg.get('warp_frac', 0.0),
        gen_size=dcfg.get('gen_size', 384))

    # The cascade and the warp already make coherent lanes, so the explicit
    # bends are needed less often and less strongly than in v11.
    if random.random() < dcfg.get('bend_prob', 0.45):
        gs = int(dcfg.get('gen_size') or 0)
        if gs and h == w and h > gs:                   # bends are the expensive
            bend = generate_diffuse_bend(gs, gs)       # part: build them small
            bend = safe_resize(bend.astype(np.float32), h, mode='bilinear', antialias=False)
        else:
            bend = generate_diffuse_bend(h, w)
        field = field * (1.0 + random.uniform(0.3, 1.8) * bend)

    return field.astype(np.float32)


def generate_clumpy_pepper_noise(h, w, pixel_scale_arcmin):
    pixel_scale_rad = pixel_scale_arcmin * (np.pi / 180.0 / 60.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)

    ell = 2 * np.pi * f / pixel_scale_rad
    ell[0, 0] = 1.0

    ell_peak = 6000.0
    width = 3000.0

    power_spectrum = np.exp(-((np.log(ell) - np.log(ell_peak))**2) / (2 * (width/ell_peak)**2))
    power_spectrum += 0.05

    taper = np.exp(-(ell / 15000.0)**3)
    power_spectrum *= taper

    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    noise_fourier = white_noise * np.sqrt(power_spectrum)

    noise_map = np.real(np.fft.ifft2(noise_fourier))
    noise_map = (noise_map - np.nanmean(noise_map)) / (np.nanstd(noise_map) + 1e-8)

    return noise_map

# CHANGED v12: SOFTENED. The previous version produced razor-thin, very bright
# scratches rather than ISM striations. Measured on the old output, normalised
# to [0,1]: median 0.0010, mean 0.0080, only 0.07% of pixels above 0.5, and an
# internal dynamic range (p99.9/median) of 854. Since the caller scales this by
# a single amplitude, the entire striation budget landed in a fraction of a
# percent of the pixels -- hence lines that look drawn on, not observed.
#
# Three causes, all fixed here:
#
#  * exp(g * 1.5..3.0) is a log-normal of width up to sigma_ln = 3, i.e. ~e^6 of
#    range. Real striation contrast is a few percent to a few tens of percent
#    (Herschel/Planck in Taurus, Musca, Polaris), so sigma_ln is now 0.15-0.5.
#  * width_scale down to 1.0 put the cross-field cutoff at the pixel scale. The
#    perpendicular knee sits at a wavelength of `width_scale` pixels, so 8-40
#    gives ridges that are resolved rather than one pixel wide.
#  * min-max normalisation let the single brightest pixel define 1.0 and crushed
#    everything else toward 0. Percentile scaling is robust to that.
#
# Return convention is unchanged -- [0,1], additive-ready -- but the field is
# now well filled rather than empty with a few spikes, so the same amplitude
# reads as a gentle modulation. The anisotropy that makes it magnetic is
# untouched: the k-space ellipse is still length_scale/width_scale ~ 10-50.
def generate_magnetic_striations(h, w, contrast=None, n_layers=None):
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]

    # a region has one dominant field direction; three crossing sets look like
    # a mesh, which is why 1 is now the usual case
    num_layers = random.choice([1, 1, 1, 2]) if n_layers is None else n_layers
    acc = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_layers):
        angle = random.uniform(0, np.pi)

        kx = fx * np.cos(angle) + fy * np.sin(angle)
        ky = -fx * np.sin(angle) + fy * np.cos(angle)

        # Pin the ELLIPSE, not the two axes independently. Drawing them
        # separately let the ratio fall to ~4 on unlucky draws, and the ratio is
        # the whole magnetic signal; the floor on width keeps ridges resolved.
        length_scale = random.uniform(150.0, 600.0)   # coherent along B
        axis_ratio = random.uniform(10.0, 30.0)
        width_scale = max(length_scale / axis_ratio, 6.0)   # RESOLVED across B

        k_sq = (kx * length_scale)**2 + (ky * width_scale)**2
        k_sq[0, 0] = 1.0

        power_spectrum = 1.0 / (k_sq ** random.uniform(1.2, 1.8))
        power_spectrum[0, 0] = 0.0

        white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
        layer = np.real(np.fft.ifft2(white_noise * np.sqrt(power_spectrum)))

        layer = (layer - np.nanmean(layer)) / (np.nanstd(layer) + 1e-8)
        acc += layer.astype(np.float32)

    acc /= np.sqrt(num_layers)            # unit variance whatever the layer count

    sigma = random.uniform(0.15, 0.5) if contrast is None else float(contrast)
    field = np.exp(sigma * acc - 0.5 * sigma ** 2)      # unit mean, low contrast

    # Striations are patchy, not box-filling. The envelope is isotropic, so it
    # injects isotropic power and dilutes the very anisotropy that makes this
    # field magnetic -- at 0.35 + 0.65*env the measured axis ratio fell from
    # 5.1 to 2.4. Kept shallow for that reason.
    env_f = np.sqrt(fx**2 + fy**2)
    env_f[0, 0] = 1.0
    env_p = 1.0 / (env_f ** 3.0)
    env_p[0, 0] = 0.0
    env = np.real(np.fft.ifft2((np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))) * np.sqrt(env_p)))
    env = (env - np.nanmin(env)) / (np.nanmax(env) - np.nanmin(env) + 1e-8)
    field = field * (0.70 + 0.30 * env)

    lo, hi = np.nanpercentile(field, [1.0, 99.0])       # robust, not min/max
    field = np.clip((field - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return field.astype(np.float32)


# CHANGED v9: SYNTHETIC STELLAR FIELD GENERATOR. The training set was
# dominated by diffuse gas morphology, biasing the SR model against compact
# sources (star clusters, galaxy stellar light). This generates point-source
# fields with realistic spatial statistics; the sources are deposited as
# Dirac deltas, so the downstream beam convolution turns them into PSF stars
# in the GT and blurred blobs in the LR input -- i.e., a direct point-source
# deconvolution training signal.
def generate_stellar_field(h, w, mode=None):
    if mode is None:
        # CHANGED v13: weights are a module switch, so a run can drop crowded
        # cluster cores ('cluster': 0.0) while keeping field stars and galaxy
        # disks. Setting stellar_prob: 0.0 in the yaml disables all of it.
        modes = [m for m in ('field', 'cluster', 'galaxy')
                 if STELLAR_MODE_WEIGHTS.get(m, 0.0) > 0.0]
        if not modes:
            raise ValueError('STELLAR_MODE_WEIGHTS has no positive weight; set '
                             'stellar_prob: 0.0 to turn the stellar branch off '
                             'instead of zeroing every mode.')
        mode = random.choices(modes, weights=[STELLAR_MODE_WEIGHTS[m] for m in modes])[0]

    star_map = np.zeros((h, w), dtype=np.float32)

    def draw_fluxes(n):
        # Power-law luminosity function dN/dS ~ S^-alpha (stellar-IMF-like
        # bright-end behavior), capped at ~3.5 dex of dynamic range.
        alpha = random.uniform(1.6, 2.4)
        u = np.random.uniform(0, 1, n)
        s = (1.0 - u) ** (-1.0 / (alpha - 1.0))
        return np.clip(s, 1.0, 3000.0)

    def deposit(ys, xs, fluxes):
        m = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        np.add.at(star_map, (ys[m].astype(np.int64), xs[m].astype(np.int64)), fluxes[m])

    if mode == 'field':
        # Uniform field stars, sparse to moderately crowded
        n = int(10 ** random.uniform(2.0, 3.7))
        deposit(np.random.uniform(0, h, n), np.random.uniform(0, w, n), draw_fluxes(n))

    elif mode == 'cluster':
        # 1-4 star clusters (Gaussian radial profiles, crowded cores) plus
        # a sparse field-star background
        for _ in range(random.randint(1, 4)):
            n = int(10 ** random.uniform(2.5, 4.0))
            cy, cx = random.uniform(0.2*h, 0.8*h), random.uniform(0.2*w, 0.8*w)
            sig = random.uniform(0.01, 0.08) * min(h, w)
            deposit(np.random.normal(cy, sig, n), np.random.normal(cx, sig, n), draw_fluxes(n))
        n_bg = int(10 ** random.uniform(1.5, 2.5))
        deposit(np.random.uniform(0, h, n_bg), np.random.uniform(0, w, n_bg), draw_fluxes(n_bg))

    else:  # 'galaxy': inclined exponential stellar disk + optional bulge
        n = int(10 ** random.uniform(3.0, 4.3))
        r_d = random.uniform(0.05, 0.20) * min(h, w)
        cy, cx = random.uniform(0.35*h, 0.65*h), random.uniform(0.35*w, 0.65*w)
        r = np.random.exponential(r_d, n)
        th = np.random.uniform(0, 2*np.pi, n)
        q = random.uniform(0.25, 1.0)                    # axis ratio (inclination)
        pa = random.uniform(0, np.pi)                    # position angle
        dx, dy = r*np.cos(th), r*np.sin(th)*q
        ys = cy + dx*np.sin(pa) + dy*np.cos(pa)
        xs = cx + dx*np.cos(pa) - dy*np.sin(pa)
        deposit(ys, xs, draw_fluxes(n))
        if random.random() < 0.6:                        # central bulge
            nb = int(n * random.uniform(0.1, 0.4))
            sb = r_d * random.uniform(0.1, 0.3)
            deposit(np.random.normal(cy, sb, nb), np.random.normal(cx, sb, nb), draw_fluxes(nb))

    peak = star_map.max()
    if peak <= 0:
        star_map[h//2, w//2] = 1.0
        peak = 1.0
    star_map /= peak

    # Optional smooth unresolved-light component (sub-detection stars /
    # diffuse stellar continuum), a blurred copy of the source distribution
    if random.random() < 0.6:
        diffuse = gaussian_filter(star_map, sigma=random.uniform(0.02, 0.06) * min(h, w))
        dmax = diffuse.max()
        if dmax > 0:
            star_map = star_map + diffuse * (random.uniform(0.1, 0.5) / dmax)

    return star_map


def generate_cib_field(h, w, local_signal_max, pixel_scale_arcmin=0.5, dense_mode=False):
    cib_map = np.zeros((h, w), dtype=np.float32)

    num_sources = random.randint(30000, 80000) if dense_mode else random.randint(10000, 30000)

    flux_scale = local_signal_max * (random.uniform(0.02, 0.08) if dense_mode else 0.015)
    fluxes = np.random.exponential(scale=flux_scale, size=num_sources)

    y_coords = np.random.randint(0, h, num_sources)
    x_coords = np.random.randint(0, w, num_sources)
    np.add.at(cib_map, (y_coords, x_coords), fluxes)

    return cib_map


def apply_filament_fragmentation(patch):
    h, w = patch.shape
    noise = np.random.normal(0, 1, (h, w))

    blur_sigma = random.uniform(h/12, h/4)
    mask = gaussian_filter(noise, sigma=blur_sigma)

    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-8)
    mask = mask ** random.uniform(2.0, 4.0)

    floor = random.uniform(0.05, 0.25)
    mask = floor + (1.0 - floor) * mask

    return patch * mask

def generate_synthetic_act_noise(h, w, pixel_scale_arcmin, ell_knee=3000.0, alpha=3.0):
    """Correlated (non-white) instrument noise: n_ell = 1 + (ell_knee/ell)^alpha.

    CHANGED v16: ell_knee and alpha are parameters. They used to be hardcoded
    at 3000 / 3.0, which is right for the TARGET grid (0.5'/px, ell out to
    ~21600, so the knee sits mid-band and the noise is white above it) and
    degenerate on the CONDITION grid, where ell only reaches ~4300 and a knee
    at 3000 boosts essentially the whole band. Mean 1/e correlation length on
    the 51x51 condition grid (interpolated, 60 realisations per knee):

        ell_knee    800    900   1000   1100   1200   1400   2000   3000
        corr len   0.92   1.32   2.74   3.58   4.34   5.88   8.34   9.02  px
        vs CMB     0.10x  0.14x  0.30x  0.39x  0.47x  0.64x  0.91x  0.99x

    where the CMB, as it appears on that same grid after the Planck beam and
    the downsample, sits at 9.16 px REGARDLESS of the beam draw (measured flat
    across planck_fwhm 3-21 target px, because the CMB is already smooth below
    the damping tail at ell ~ 1300 and the beam has nothing left to remove).

    That matters because the CMB is SKY: it is injected before the beam split,
    so it is in BOTH endpoints and the target keeps it -- the network has to
    SHARPEN it. This noise is in the condition only, so the network has to
    REMOVE it. At a knee near 3000 the two are the same field to within a
    correlation length, the problem is unidentifiable, and the MSE-optimal
    answer is to partially damp both -- losing exactly the Gaussian-field
    robustness you care about. Stay at or below ~1200 (under half the CMB
    scale). Note the per-realisation scatter is wide: a 51^2 grid holds only a
    handful of independent modes at these scales, so design against the mean,
    not against a single field.
    """
    pixel_scale_rad = pixel_scale_arcmin * (np.pi / 180.0 / 60.0)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx**2 + fy**2)

    ell = 2 * np.pi * f / pixel_scale_rad
    ell[0, 0] = 1.0

    n_ell = 1.0 + (float(ell_knee) / ell)**float(alpha)
    taper = np.exp(-(ell / 15000.0)**3)

    power_spectrum = n_ell * taper

    white_noise = np.random.normal(0, 1, (h, w)) + 1j * np.random.normal(0, 1, (h, w))
    noise_fourier = white_noise * np.sqrt(power_spectrum)

    noise_map = np.real(np.fft.ifft2(noise_fourier))
    noise_map = (noise_map - np.nanmean(noise_map)) / (np.nanstd(noise_map) + 1e-8)
    return noise_map


def apply_scanning_stripes(patch, intensity=0.15):
    h, w = patch.shape
    local_std = np.nanstd(patch) + 1e-8
    f = np.fft.fftfreq(h)
    f[0] = 1.0
    amp = 1.0 / np.sqrt(np.abs(f))
    amp[0] = 0.0
    complex_noise = np.random.normal(0, 1, h) + 1j * np.random.normal(0, 1, h)
    stripe_1d = np.real(np.fft.ifft(complex_noise * amp))
    stripe_1d = (stripe_1d - np.nanmean(stripe_1d)) / (np.nanstd(stripe_1d) + 1e-8)
    return patch + (np.tile(stripe_1d[:, None], (1, w)) * local_std * intensity)

def apply_advanced_telescope_noise(low_res_map, augment=True, clean_mode=False,
                                   low_level=None, noise_cfg=None, seed=None,
                                   pixel_scale_arcmin=None):
    local_std = np.nanstd(low_res_map) + 1e-8

    # CHANGED v3: space-telescope (HST/JWST-like) regime. Noise is either
    # exactly zero or a token 0.2-2% of local std, mimicking the effectively
    # noiseless deep-field limit while keeping a continuum with the noisy mode.
    if clean_mode:
        if random.random() < 0.3:
            return low_res_map.astype(np.float32)
        return (low_res_map + np.random.normal(0, local_std * random.uniform(0.002, 0.02), low_res_map.shape)).astype(np.float32)

    if not augment:
        # CHANGED v15: the level is a config value (default 0.02, the v3
        # constant). With val_low_fraction > 0 a hashed subset of val patches
        # gets val_low_level instead -- FIXED, not drawn, so the patch is
        # identical every epoch, same contract as val_force_diffuse.
        ncfg = NOISE_DEFAULTS if noise_cfg is None else noise_cfg
        level = ncfg['val_level']
        if ncfg['val_low_fraction'] > 0.0 and seed is not None:
            h = _mix64(int(seed) * 0x9E3779B97F4A7C15 + 0x51ED270B)
            if (h >> 11) / float(1 << 53) < ncfg['val_low_fraction']:
                level = ncfg['val_low_level']
        return (low_res_map + np.random.normal(0, local_std * level, low_res_map.shape)).astype(np.float32)

    ncfg = NOISE_DEFAULTS if noise_cfg is None else noise_cfg
    h, w = low_res_map.shape
    total_noise = np.zeros_like(low_res_map)
    # CHANGED v3: floor lowered 0.10 -> 0.05 so noise level forms a continuum
    # from the clean regime (0-2%) up to heavily noisy (30%), with no gap the
    # network never trains on.
    # CHANGED v15: the low-noise observed branch. Same shape of noise (white
    # + optional sqrt(signal) shot term), just a level drawn from a decade far
    # below the ordinary floor. See the v15 header for why 2% was the floor.
    if low_level is not None:
        dynamic_noise_base = local_std * float(low_level)
    else:
        dynamic_noise_base = local_std * random.uniform(0.05, 0.30)
    white_sigma = dynamic_noise_base * random.uniform(0.4, 0.8)

    # CHANGED v16: split the white budget with a correlated and/or a speckle
    # field, VARIANCE-PRESERVING (the fractions are of the variance, hence the
    # sqrt), so the realised noise level is untouched and only the morphology
    # changes. Neither branch consumes a random number when its prob is 0, so
    # a config that leaves these off is bit-identical to v15.
    f_corr = f_pep = 0.0
    if pixel_scale_arcmin is not None:
        if ncfg['corr_prob'] > 0.0 and random.random() < ncfg['corr_prob']:
            f_corr = _u(ncfg['corr_frac'])
        if ncfg['pepper_prob'] > 0.0 and random.random() < ncfg['pepper_prob']:
            f_pep = _u(ncfg['pepper_frac'])
    f_white = max(0.0, 1.0 - f_corr - f_pep)

    total_noise += np.random.normal(0, white_sigma * np.sqrt(f_white), (h, w))
    if f_corr > 0.0:
        total_noise += generate_synthetic_act_noise(
            h, w, pixel_scale_arcmin, ell_knee=_u(ncfg['corr_knee']),
            alpha=ncfg['corr_alpha']) * (white_sigma * np.sqrt(f_corr))
    if f_pep > 0.0:
        total_noise += generate_clumpy_pepper_noise(
            h, w, pixel_scale_arcmin) * (white_sigma * np.sqrt(f_pep))

    if random.random() < 0.50:
        shifted_signal = np.clip(low_res_map - np.nanmin(low_res_map) + 1e-6, 0.0, None)
        noise_envelope = np.sqrt(shifted_signal)
        noise_envelope /= (np.nanmean(noise_envelope) + 1e-8)
        total_noise += np.random.normal(0, 1, (h, w)) * noise_envelope * (dynamic_noise_base * random.uniform(0.3, 1.0))
    return (low_res_map + total_noise).astype(np.float32)

# =========================================================
# CORE PROCESSING PIPELINE (Shared across datasets)
# =========================================================
def process_noise_to_clean_patch(cropped_view, augment, target_size, planck_size, total_fov_arcmin, crop_size, add_point_sources=True, clean_prob=0.15, work_size=768, stellar_prob=0.12, diffuse_prob=None, force_diffuse=False, diffuse_seed=None, diffuse_cfg=None, upsample_cfg=None, noise_cfg=None):
    # CHANGED v13: one dict carries every diffuse switch. None -> v12 defaults.
    dcfg = DIFFUSE_DEFAULTS if diffuse_cfg is None else diffuse_cfg
    # CHANGED v3: with probability clean_prob, produce a "space telescope"
    # pair: the input is degraded ONLY by the PSF + downsampling, with zero
    # or near-zero instrument noise, no scanning stripes, and no CMB (which
    # HST/JWST do not see). Signal-side augmentations (cirrus, CIB galaxies,
    # point sources) are kept, since deep space-telescope fields contain them.
    clean_mode = bool(augment) and (random.random() < clean_prob)

    cropped_view = np.nan_to_num(cropped_view, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # CHANGED v2: robust hot-pixel / corrupted-value clip. A single pixel
    # thousands of times brighter than the 99.9th percentile (numerical
    # artifact, or a corrupted quicklook value) previously set local_max and
    # compressed the entire patch into a delta-function histogram. Allowing
    # up to 1e3x the 99.9th percentile preserves genuine bright cores/AGN
    # while killing garbage values.
    p_hi = np.percentile(np.abs(cropped_view), 99.9)
    if np.isfinite(p_hi) and p_hi > 0:
        ceiling = p_hi * 1e3
        np.clip(cropped_view, -ceiling, ceiling, out=cropped_view)

    local_max = np.nanmax(np.abs(cropped_view))

    if local_max == 0 or np.isnan(local_max): local_max = 1.0
    cropped_view = (cropped_view / local_max).astype(np.float32)
    cropped_view = check_finite("Base Crop", cropped_view)

    # CHANGED v5: WORK-RESOLUTION PROCESSING. Everything downstream is
    # convolved with the ACT beam and resampled to target_size (256), so a
    # crop carries no information at scales below ~target resolution that
    # survives to the GT. Downsampling large crops to work_size (3x target)
    # BEFORE injections and blurs cuts the FFT contaminant generators and
    # Gaussian blurs from up to 1536^2 to <=768^2 with no effect on the
    # final training pair statistics. The field of view (and therefore all
    # angular scales) is preserved: pixel_scale_arcmin uses the effective
    # size. Set work_size=None to recover the old exact behavior.
    if work_size is not None and crop_size > work_size:
        cropped_view = safe_resize(cropped_view, work_size, mode='bicubic', antialias=True)
        crop_size = work_size

    base_signal = cropped_view.copy()
    pixel_scale_arcmin = total_fov_arcmin / crop_size

    is_null_patch = False
    heavy_cib_mode = False
    stellar_mode = False
    diffuse_mode = False                      # CHANGED v11
    if diffuse_prob is None:
        diffuse_prob = dcfg['prob']           # CHANGED v13

    # CHANGED v12: the validation route into the diffuse branch. Deliberately
    # placed before, and outside, the `if augment:` block -- that block is the
    # reason validation never saw one of these. Only fires when augment is off,
    # so the training path below is byte-for-byte what v11 did.
    if force_diffuse and not augment:
        diffuse_mode = True
        h_crop, w_crop = base_signal.shape
        st_np, st_py = np.random.get_state(), random.getstate()
        if diffuse_seed is not None:
            np.random.seed(int(diffuse_seed) % (2 ** 32 - 1))
            random.seed(int(diffuse_seed))
        try:
            diffuse = generate_diffuse_field(h_crop, w_crop, dcfg=dcfg)
        finally:
            np.random.set_state(st_np)        # never leak the fixed seed into
            random.setstate(st_py)            # the rest of the pipeline
        crop_scale = max(np.nanpercentile(np.abs(cropped_view), 99.5), 1e-3)
        d_scale = max(np.nanpercentile(diffuse, 99.5), 1e-8)
        base_signal = diffuse * (crop_scale / d_scale)

    if augment:
        rand_roll = random.random()
        h_crop, w_crop = base_signal.shape

        # CHANGED v9: STELLAR-FIELD SAMPLE. With probability stellar_prob the
        # gas crop is replaced by (or blended under) a synthetic star field /
        # cluster / galaxy stellar disk. Combined with clean_prob this also
        # yields HST/JWST-like noiseless star-field pairs.
        if random.random() < stellar_prob:
            stellar_mode = True
            stellar = generate_stellar_field(h_crop, w_crop)

            # CHANGED v10: BAND-LIMIT the delta sources to the TARGET
            # sampling before any resampling. Sub-Nyquist deltas pushed
            # through the bicubic downsample imprint the interpolation
            # kernel's ringing phase-locked to the pixel grid; with
            # thousands of stars this tiles into a mesh artifact. A minimum
            # effective PSF of ~1.7 target pixels (sigma = 0.7 x the
            # downsample factor) is the physical Nyquist-sampled limit of a
            # real instrument, and removes the moire entirely. Peaks are
            # renormalized so source amplitudes survive the blur.
            aa_sigma = 0.7 * (crop_size / target_size)
            if aa_sigma > 0.3:
                stellar = gaussian_filter(stellar, sigma=aa_sigma, truncate=4.0)
                smax = stellar.max()
                if smax > 0:
                    stellar = stellar / smax

            if random.random() < 0.4:
                # stars embedded in faint gas (e.g., cluster still in its cloud)
                base_signal = base_signal * random.uniform(0.05, 0.35) + stellar
            else:
                base_signal = stellar

        # CHANGED v11: GRAVITY-FREE DIFFUSE SAMPLE. Mutually exclusive with the
        # stellar branch: one replaces the crop with compact sources, this one
        # with extended structure that has no dense spine to find.
        elif random.random() < diffuse_prob:
            diffuse_mode = True
            diffuse = generate_diffuse_field(h_crop, w_crop, dcfg=dcfg)  # CHANGED v13
            # match the crop's own amplitude so every downstream scale
            # (effective_scale, contaminant caps, arcsinh) behaves unchanged
            crop_scale = max(np.nanpercentile(np.abs(cropped_view), 99.5), 1e-3)
            d_scale = max(np.nanpercentile(diffuse, 99.5), 1e-8)
            diffuse = diffuse * (crop_scale / d_scale)

            if random.random() < dcfg['blend_prob']:
                # a real field seen through a diffuse foreground/envelope
                base_signal = base_signal * random.uniform(0.05, 0.35) + diffuse
            else:
                base_signal = diffuse

        if stellar_mode or diffuse_mode:
            pass  # CHANGED v9/v11: skip vacuum/fragmentation for both
        elif rand_roll < 0.05:
            base_signal = np.zeros_like(base_signal)
            is_null_patch = True
            heavy_cib_mode = True

        elif rand_roll < 0.15:
            base_signal = base_signal * random.uniform(0.05, 0.2)
            heavy_cib_mode = True

        elif random.random() < 0.3:
            base_signal = apply_filament_fragmentation(base_signal)
            base_signal += random.uniform(0.05, 0.15) * np.nanmax(base_signal)

        if stellar_mode or diffuse_mode:
            # CHANGED v9/v11: scale anchor from the synthesised field itself,
            # not the (replaced) gas crop
            effective_scale = max(np.nanpercentile(base_signal, 99.5), 1e-3)
        elif is_null_patch:
            effective_scale = random.uniform(0.1, 0.3)
        else:
            effective_scale = max(np.nanpercentile(cropped_view, 99.5), 1e-3)

        # CHANGED: measure the STRUCTURAL contrast of the signal itself.
        # effective_scale (a high percentile) saturates near 1 even for a
        # smooth/flat crop, which previously let the injected CMB/cirrus
        # become the dominant structure in featureless patches. Capping the
        # contaminant amplitudes at a multiple of the signal's own std keeps
        # the galaxy/dust structure as the primary content of the GT.
        structure_std = float(np.nanstd(base_signal)) + 1e-8

        if not is_null_patch and random.random() < 0.75:
            cirrus_bg = generate_diffuse_cirrus(h_crop, w_crop, pixel_scale_arcmin)
            cirrus_amp = effective_scale * random.uniform(0.05, 0.25)
            # CHANGED: contaminant cap (skip cap in heavy_cib / suppressed modes,
            # where a contaminant-dominated patch is the intended behavior)
            if not heavy_cib_mode:
                cirrus_amp = min(cirrus_amp, structure_std * random.uniform(0.5, 1.5))
            base_signal += (cirrus_bg * cirrus_amp)

        if not is_null_patch and not stellar_mode and random.random() < 0.50:
            striations = generate_magnetic_striations(h_crop, w_crop)
            if diffuse_mode:
                # CHANGED v12: on a diffuse sample the striations MODULATE the
                # gas instead of being laid on top of it. A striation is a
                # density modulation, so it has to vanish where there is no gas
                # and scale with the column where there is. Added additively it
                # reads as bright lines drawn over empty sky -- the main reason
                # the old output looked unlike real data. Depth 5-35% spans the
                # observed contrast range.
                depth = random.uniform(0.05, 0.35)
                base_signal = base_signal * (
                    1.0 + depth * 2.0 * (striations - float(np.mean(striations))))
            else:
                striation_amp = effective_scale * random.uniform(0.1, 0.4)
                # CHANGED: contaminant cap
                if not heavy_cib_mode:
                    striation_amp = min(striation_amp, structure_std * random.uniform(0.5, 1.5))
                base_signal += (striations * striation_amp)

        # CHANGED v3: no CMB in clean (HST/JWST-like) pairs
        if (not clean_mode) and random.random() < 0.9:
            cmb_base = generate_cmb_grf(h_crop, w_crop, pixel_scale_arcmin)
            if is_null_patch or heavy_cib_mode:
                cmb_amp = effective_scale * random.uniform(0.15, 0.4)
            else:
                cmb_amp = effective_scale * random.uniform(0.01, 0.05)
                # CHANGED: contaminant cap -- prevents CMB-dominated GTs on
                # smooth/faint galaxy crops (the blobby +/- patches you saw)
                cmb_amp = min(cmb_amp, structure_std * random.uniform(0.3, 0.8))
            base_signal += (cmb_base * cmb_amp)

        if heavy_cib_mode or random.random() < 0.95:
            # CHANGED v4: cap the CIB flux scale by the structural amplitude
            # of the actual signal (same principle as the CMB/cirrus caps).
            # Prevents a weakly structured crop from producing a GT that is
            # pure injected speckle. heavy_cib_mode keeps the full amplitude
            # since a CIB-dominated patch is the intended behavior there.
            if heavy_cib_mode:
                cib_signal_scale = effective_scale
            else:
                cib_signal_scale = min(effective_scale, structure_std * random.uniform(3.0, 8.0))
            cib_base = generate_cib_field(h_crop, w_crop, cib_signal_scale, pixel_scale_arcmin, dense_mode=heavy_cib_mode)
            # CHANGED v10: band-limit CIB deltas too (gaussian_filter
            # conserves total flux, so the confusion level is unchanged;
            # real CIB confusion is beam-convolved, never single-pixel)
            cib_aa = 0.7 * (crop_size / target_size)
            if cib_aa > 0.3:
                cib_base = gaussian_filter(cib_base, sigma=cib_aa, truncate=4.0)
            base_signal += cib_base

        # CHANGED v11: no injected point sources on a diffuse sample. The CIB
        # above is left in place -- it is genuinely present in the sky -- but
        # sprinkling bright deltas onto a field whose whole purpose is to have
        # no compact structure would undo the point of the sample.
        if add_point_sources and not is_null_patch and not stellar_mode and not diffuse_mode:
            base_signal = inject_point_sources(base_signal, max_sources=15, pixel_scale_arcmin=pixel_scale_arcmin,
                                               aa_sigma=0.7 * (crop_size / target_size))

    base_signal = check_finite("Base Signal", base_signal)


    # CHANGED v13: DIFFUSE SAMPLES GET THEIR OWN TARGET BEAM.
    #
    # act_fwhm is in TARGET pixels (0.5' each here), so the shared 0.4-3.5 draw
    # spans 0.2'-1.75'. At the top of that range a gravity-free field is
    # smoothed over 3.5 target pixels and the GT retains no small-scale diffuse
    # texture at all -- the sample then teaches "diffuse in, smooth out", which
    # is the opposite of what the diffuse branch exists to teach. A smaller
    # target beam keeps the turbulent cascade alive down to ~1-2 target pixels
    # in the GT, so the network has something diffuse to learn to generate.
    #
    # planck_ratio is deliberately separate. The condition is resampled to
    # planck_size regardless, so the true super-resolution factor is fixed at
    # target_size/planck_size (~5x). The ratio only decides whether the
    # condition is beam-limited (ratio >> 5) or pixel-limited (ratio << 5);
    # lower it if you want the diffuse samples to be an easier restoration.
    if augment:
        if diffuse_mode and dcfg.get('act_fwhm') is not None:
            act_fwhm = _u(dcfg['act_fwhm'])
            planck_fwhm = act_fwhm * _u(dcfg.get('planck_ratio') or (3.0, 6.0))
        else:
            act_fwhm = random.uniform(0.4, 3.5)
            planck_fwhm = act_fwhm * random.uniform(3.0, 6.0)
    else:
        if diffuse_mode and dcfg.get('act_fwhm') is not None:
            # fixed, not drawn: val patches must be identical every epoch
            act_fwhm = float(dcfg.get('val_act_fwhm', 1.0))
            planck_fwhm = float(dcfg.get('val_planck_fwhm', 5.0))
        else:
            act_fwhm, planck_fwhm = 3.0, 15.0

    # Clean Target
    act_sigma = (act_fwhm * (crop_size / target_size)) / 2.355
    act_blurred = check_finite("ACT Blur", gaussian_filter(base_signal, sigma=act_sigma, truncate=4.0).astype(np.float32))

    gt_raw_np = safe_resize(act_blurred, target_size, mode='bicubic', antialias=True)
    gt_raw_np = check_finite("GT Target", gt_raw_np)

    # Degraded Input
    planck_sigma = (planck_fwhm * (crop_size / target_size)) / 2.355
    planck_blurred = check_finite("Planck Blur", gaussian_filter(base_signal, sigma=planck_sigma, truncate=4.0).astype(np.float32))

    planck_linear = safe_resize(planck_blurred, planck_size, mode='bicubic', antialias=True)
    planck_linear = check_finite("Planck Linear", planck_linear)

    # CHANGED v15: one draw decides whether this is a low-noise OBSERVED
    # sample; if so the detector noise, the striping and the post-upsample term
    # below are ALL scaled down together, so the condition stays internally
    # consistent instead of a 1% map under a 20% stripe. No RNG is consumed
    # when noise.low_prob is 0.
    ncfg = NOISE_DEFAULTS if noise_cfg is None else noise_cfg
    low_level = draw_low_noise(ncfg, augment, clean_mode)   # None = ordinary

    planck_noisy = apply_advanced_telescope_noise(planck_linear, augment=augment, clean_mode=clean_mode,
                                                  low_level=low_level, noise_cfg=ncfg, seed=diffuse_seed,
                                                  pixel_scale_arcmin=total_fov_arcmin / planck_size)
    # CHANGED v3: scanning stripes are an ACT/Planck artifact; skip when clean
    if augment and (not clean_mode) and random.random() < 0.5:
        stripe_amp = _u(ncfg['low_stripe_range'] if low_level is not None
                        else ncfg['stripe_range'])
        planck_noisy = apply_scanning_stripes(planck_noisy, intensity=stripe_amp)
    planck_noisy = check_finite("Planck Noise", planck_noisy.astype(np.float32))

    # CHANGED v14 (0901): the kernel that lifts the condition back onto the
    # target grid is drawn per sample now -- 0 replicate / 1 linear / 3 cubic.
    # upsample_cfg None keeps it at bicubic. See UPSAMPLE_DEFAULTS for why.
    up_order = pick_upsample_order(upsample_cfg, augment, seed=diffuse_seed)
    planck_upsampled = safe_resize(planck_noisy, target_size, order=up_order, antialias=False)

    if augment:
        # CHANGED v3: post-upsample noise also skipped in clean mode
        if not clean_mode:
            # CHANGED v15: this term is white on the TARGET grid -- it never
            # passes through the interpolation -- so on a low-noise sample it
            # would be 2-4x the real low-res noise and swamp the very texture
            # the sample exists to teach. Scaled with the regime.
            post_amp = (low_level * _u(ncfg['post_low_factor'])
                        if low_level is not None else _u(ncfg['post_range']))
            planck_upsampled += np.random.normal(0, (np.nanstd(planck_upsampled) + 1e-8) * post_amp, planck_upsampled.shape).astype(np.float32)

        if random.random() < 0.5 and not is_null_patch:
            shift_y = random.uniform(-0.5, 0.5)
            shift_x = random.uniform(-0.5, 0.5)
            planck_upsampled = shift(planck_upsampled, shift=(shift_y, shift_x), order=1, mode='nearest').astype(np.float32)

    input_raw_np = check_finite("Input Final", planck_upsampled)

    c_fill = float(np.nanmedian(input_raw_np))
    if np.isnan(c_fill): c_fill = 0.0

    t_in, t_gt = torch.from_numpy(input_raw_np).unsqueeze(0), torch.from_numpy(gt_raw_np).unsqueeze(0)
    t_mask = torch.ones_like(t_gt)

    if augment:
        if random.random() < 0.5: t_in, t_gt = TF.hflip(t_in), TF.hflip(t_gt)
        if random.random() < 0.5: t_in, t_gt = TF.vflip(t_in), TF.vflip(t_gt)
        if random.random() < 0.5:
            angle = random.uniform(0.0, 360.0)
            t_in, t_gt = TF.rotate(t_in, angle, fill=c_fill), TF.rotate(t_gt, angle, fill=c_fill)

    _, memory_tensor = apply_patch_asymmetric_arcsinh(input_raw_np, headroom_multiplier=2.5)

    alpha, t_min, t_max = memory_tensor[1].item(), memory_tensor[2].item(), memory_tensor[3].item()
    in_norm_np = np.clip((np.arcsinh(t_in[0].numpy() / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)
    gt_final_norm = np.clip((np.arcsinh(t_gt[0].numpy() / alpha) - t_min) / (t_max - t_min), 0.0, 1.0)

    memory_tensor[0] *= local_max
    memory_tensor[1] *= local_max

    patch_cond = torch.tensor(in_norm_np, dtype=torch.float32).unsqueeze(0)
    patch_y = torch.tensor(gt_final_norm, dtype=torch.float32).unsqueeze(0)

    return patch_cond.clone(), patch_cond, patch_y, t_mask[0], memory_tensor


# =========================================================
# DATASET 1: RAW DUST MAPS (H, W, Views)
# =========================================================
class SyntheticAstroDataset(Dataset):
    def __init__(self, cfg, file_list, mode='train'):
        self.cfg = cfg
        self.file_list = file_list
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)

        self.target_size = 256
        self.planck_size = 51
        self.min_crop, self.max_crop = 512, 1536
        self.val_crop_scales = [512, 768, 1024, 1280, 1536]
        self.total_fov_arcmin = self.target_size * 0.5
        # CHANGED v3: fraction of clean (space-telescope-like) pairs
        self.clean_prob = cfg.dataset.get('clean_prob', 0.15)
        # CHANGED v9: fraction of stellar-field (star/cluster/galaxy) pairs
        self.stellar_prob = cfg.dataset.get('stellar_prob', 0.12)
        # CHANGED v13: every diffuse switch in one picklable dict
        self.dcfg = diffuse_cfg_from(cfg)
        # CHANGED v14: condition-upsampling kernel mix, picklable the same way
        self.upcfg = upsample_cfg_from(cfg)
        # CHANGED v15: condition-noise levels, same treatment
        self.ncfg = noise_cfg_from(cfg)

    def __len__(self): return len(self.file_list)

    def __getitem__(self, index):
        raw_cube = np.load(self.file_list[index], mmap_mode='c')
        h, w, num_views = raw_cube.shape
        max_c = min(h, w)
        act_min, act_max = min(self.min_crop, max_c), min(self.max_crop, max_c)

        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []

        for p_i in range(self.num_patches):   # CHANGED v12: patch index
            # CHANGED v7: dust maps also contain smooth featureless regions;
            # same gate + best-candidate retry as SKIRT/FIRE. No pedestal or
            # DR conditioning here -- dust maps are signed and well-scaled.
            best_crop, best_score, best_c = None, -1.0, act_min
            for attempt in range(8):
                v_idx = random.randint(0, num_views - 1)
                if self.mode == 'train':
                    c_size = random.randint(act_min, act_max)
                    y_off, x_off = random.randint(0, h - c_size), random.randint(0, w - c_size)
                else:
                    valid = [c for c in self.val_crop_scales if c <= max_c]
                    c_size = random.choice(valid) if valid else max_c
                    y_off, x_off = (h - c_size) // 2, (w - c_size) // 2

                crop = raw_cube[y_off:y_off+c_size, x_off:x_off+c_size, v_idx].copy()
                if self.mode != 'train' or crop_is_informative(crop, bypass_prob=self.dcfg['bypass_prob']):
                    best_crop, best_c = crop, c_size
                    break
                score = crop_structure_score(crop)
                if score > best_score:
                    best_crop, best_score, best_c = crop, score, c_size

            crop, c_size = best_crop, best_c
            fd, fseed = val_force_diffuse(index, p_i, self.dcfg['val_fraction']) if self.mode != 'train' else (False, None)
            p1, p2, py, pm, pmem = process_noise_to_clean_patch(crop, self.augment, self.target_size, self.planck_size, self.total_fov_arcmin, c_size, add_point_sources=True, clean_prob=self.clean_prob, stellar_prob=self.stellar_prob, force_diffuse=fd, diffuse_seed=fseed, diffuse_cfg=self.dcfg, upsample_cfg=self.upcfg, noise_cfg=self.ncfg)

            x_prim_list.append(p1); x_cond_list.append(p2); y_list.append(py); mask_list.append(pm); memory_list.append(pmem)

        return (torch.stack(x_prim_list), torch.stack(x_cond_list), torch.stack(y_list), torch.stack(mask_list), torch.stack(memory_list), index)


# =========================================================
# DATASET 2: SKIRT MULTIBAND (Views, Bands, H, W)
# =========================================================
class SkirtAstroDataset(Dataset):
    def __init__(self, cfg, file_list, mode='train'):
        self.cfg = cfg
        self.file_list = file_list
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)

        self.target_size = 256
        self.planck_size = 51
        self.min_crop, self.max_crop = 512, 1536
        self.val_crop_scales = [512, 768, 1024, 1280, 1536]
        self.total_fov_arcmin = self.target_size * 0.5
        # CHANGED v3: fraction of clean (space-telescope-like) pairs
        self.clean_prob = cfg.dataset.get('clean_prob', 0.15)
        # CHANGED v9: fraction of stellar-field (star/cluster/galaxy) pairs
        self.stellar_prob = cfg.dataset.get('stellar_prob', 0.12)
        # CHANGED v13: every diffuse switch in one picklable dict
        self.dcfg = diffuse_cfg_from(cfg)
        # CHANGED v14: condition-upsampling kernel mix, picklable the same way
        self.upcfg = upsample_cfg_from(cfg)
        # CHANGED v15: condition-noise levels, same treatment
        self.ncfg = noise_cfg_from(cfg)

        # CHANGED v2: SKIRT multiband cubes contain near-flat faint bands and
        # crops on empty sky, which previously produced the same degenerate
        # delta-histogram samples as FIRE. Gate crops the same way.
        self.max_crop_tries = 8

    def __len__(self): return len(self.file_list)

    def __getitem__(self, index):
        raw_cube = np.load(self.file_list[index], mmap_mode='c')
        num_views, num_bands, h, w = raw_cube.shape
        max_c = min(h, w)
        act_min, act_max = min(self.min_crop, max_c), min(self.max_crop, max_c)

        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []

        for p_i in range(self.num_patches):   # CHANGED v12: patch index
            # CHANGED v4: condition the crop (pedestal removal + randomized
            # dynamic-range compression) BEFORE gating, then keep the BEST
            # scoring candidate if no attempt passes the gate. Previously,
            # cubes where most view/band combos are flat exhausted the 8
            # retries and blindly accepted the last flat crop.
            best_crop, best_score, best_c = None, -1.0, act_min
            crop = None
            c_size = act_min
            for attempt in range(self.max_crop_tries):
                v_idx = random.randint(0, num_views - 1)
                b_idx = random.randint(0, num_bands - 1)

                if self.mode == 'train':
                    c_size = random.randint(act_min, act_max)
                    y_off, x_off = random.randint(0, h - c_size), random.randint(0, w - c_size)
                else:
                    valid = [c for c in self.val_crop_scales if c <= max_c]
                    c_size = random.choice(valid) if valid else max_c
                    y_off, x_off = (h - c_size) // 2, (w - c_size) // 2

                crop = raw_cube[v_idx, b_idx, y_off:y_off+c_size, x_off:x_off+c_size].copy()
                crop = prepare_extreme_dr_crop(crop)

                if self.mode != 'train' or crop_is_informative(crop, bypass_prob=self.dcfg['bypass_prob']):
                    best_crop, best_c = crop, c_size
                    break

                score = crop_structure_score(crop)
                if score > best_score:
                    best_crop, best_score, best_c = crop, score, c_size

            crop, c_size = best_crop, best_c

            fd, fseed = val_force_diffuse(index, p_i, self.dcfg['val_fraction']) if self.mode != 'train' else (False, None)
            p1, p2, py, pm, pmem = process_noise_to_clean_patch(crop, self.augment, self.target_size, self.planck_size, self.total_fov_arcmin, c_size, add_point_sources=False, clean_prob=self.clean_prob, stellar_prob=self.stellar_prob, force_diffuse=fd, diffuse_seed=fseed, diffuse_cfg=self.dcfg, upsample_cfg=self.upcfg, noise_cfg=self.ncfg)

            x_prim_list.append(p1); x_cond_list.append(p2); y_list.append(py); mask_list.append(pm); memory_list.append(pmem)

        return (torch.stack(x_prim_list), torch.stack(x_cond_list), torch.stack(y_list), torch.stack(mask_list), torch.stack(memory_list), index)


# =========================================================
# DATASET 3: FIRE QUICKLOOK (1536x1536 .npz arrays)
# =========================================================
class FireQuicklookDataset(Dataset):
    CACHE_VERSION = 2  # CHANGED v6: bump when _preprocess_cube semantics change

    def __init__(self, cfg, file_list, mode='train'):
        self.cfg = cfg
        self.file_list = file_list
        self.mode = mode
        self.augment = cfg.dataset.get('augment', True) and (mode == 'train')
        self.num_patches = cfg.dataset.get('num_patches_train', 1) if mode == 'train' else cfg.dataset.get('num_patches_val', 4)

        self.target_size = 256
        self.planck_size = 51
        self.min_crop, self.max_crop = 512, 1536
        self.val_crop_scales = [512, 768, 1024, 1280, 1536]
        self.total_fov_arcmin = self.target_size * 0.5
        # CHANGED v3: fraction of clean (space-telescope-like) pairs
        self.clean_prob = cfg.dataset.get('clean_prob', 0.15)
        # CHANGED v9: fraction of stellar-field (star/cluster/galaxy) pairs
        self.stellar_prob = cfg.dataset.get('stellar_prob', 0.12)
        # CHANGED v13: every diffuse switch in one picklable dict
        self.dcfg = diffuse_cfg_from(cfg)
        # CHANGED v14: condition-upsampling kernel mix, picklable the same way
        self.upcfg = upsample_cfg_from(cfg)
        # CHANGED v15: condition-noise levels, same treatment
        self.ncfg = noise_cfg_from(cfg)

        # CHANGED: 1e-5 was far too permissive for maps with ~5 dex of
        # dynamic range; the bounding box ended up spanning the pedestal
        # halo, so crops routinely landed on featureless regions.
        self.threshold_ratio = 1e-2

        # CHANGED: crop quality gate parameters
        self.max_crop_tries = 8       # resample attempts per patch
        self.min_crop_contrast = 3.0  # required p99.5 / p50 within a crop
        self.min_filled_frac = 0.02   # min fraction of pixels above zero

        # CHANGED: pedestal removal percentile (FIRE quicklook has a
        # diffuse uniform floor that carries no morphological information)
        self.pedestal_percentile = 5.0

        # CHANGED v5: optional on-disk preprocessing cache. The sanitize /
        # corrupted-view filter / pedestal removal / outlier clip / peak
        # normalization chain is deterministic per file but was re-executed
        # on the FULL float64 cube (plus npz decompression) at EVERY
        # __getitem__. With fire_cache_dir set, it runs once per file and
        # the float32 result is memory-mapped thereafter, so a __getitem__
        # only ever touches the pixels of the sampled crop. The random
        # Kelvin peak is a uniform scale that the pipeline's local_max
        # normalization absorbs, so applying it to the crop at runtime is
        # mathematically identical to applying it to the whole cube.
        self.cache_dir = cfg.dataset.get('fire_cache_dir', None)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_paths(self, src_path):
        stem = os.path.splitext(os.path.basename(src_path))[0]
        return (os.path.join(self.cache_dir, stem + '_prep.npy'),
                os.path.join(self.cache_dir, stem + '_meta.npz'))

    def _preprocess_cube(self, src_path):
        """Deterministic per-file pipeline (v2 semantics): float64 sanitize,
        corrupted-view drop, pedestal removal, outlier clip, peak-normalize
        to p99.9 = 1. Returns (float32 cube, bboxes int64 (V,4))."""
        images = np.asarray(np.load(src_path)['images'], dtype=np.float64)
        images = np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)

        # CHANGED v6: UNIT-AGNOSTIC view handling. Quicklook units are
        # arbitrary, so an absolute "corrupted above 1e30" threshold was
        # wrong -- files whose genuine values exceeded it lost every view,
        # and the fallback then walked the whole (all-empty) file list.
        # Now: drop a view only if it is EMPTY (max <= 0 after sanitize);
        # garbage pixels are handled RELATIVELY, by clipping each view at
        # 1e3x its own 99.9th positive percentile. An entirely-garbage
        # uniform view survives this but becomes flat, and the crop
        # informativeness gate rejects it downstream.
        good = []
        for v in range(images.shape[0]):
            vmax = images[v].max()
            if not np.isfinite(vmax) or vmax <= 0:
                continue
            pos = _stat_view(images[v])
            pos = pos[pos > 0]
            if pos.size > 0:
                p999 = np.percentile(pos, 99.9)
                if np.isfinite(p999) and p999 > 0:
                    np.clip(images[v], None, p999 * 1e3, out=images[v])
            good.append(v)
        images = images[np.asarray(good, dtype=np.int64)] if good else images[:0]

        for v in range(images.shape[0]):
            pos = images[v][images[v] > 0]
            if pos.size > 0:
                pedestal = np.percentile(pos, self.pedestal_percentile)
                images[v] = np.clip(images[v] - pedestal, 0.0, None)

        valid = images[images > 0]
        peak = np.percentile(valid, 99.9) if valid.size > 0 else 1.0
        if not np.isfinite(peak) or peak <= 0:
            peak = 1.0
        np.clip(images, 0.0, peak * 1e3, out=images)
        images = (images / peak).astype(np.float32)

        bboxes = np.zeros((images.shape[0], 4), dtype=np.int64)
        for v in range(images.shape[0]):
            bboxes[v] = self.get_valid_crop_limits(images[v])
        return images, bboxes

    def _load_prepared(self, index):
        """Returns (images, bboxes) where images is float32 with p99.9 = 1.
        Cached path: images is a read-only memory map (near-zero load cost)."""
        src_path = self.file_list[index]
        if not self.cache_dir:
            return self._preprocess_cube(src_path)

        prep_path, meta_path = self._cache_paths(src_path)

        # CHANGED v6: cache versioning -- caches built by an older
        # preprocessing rule are rebuilt automatically.
        rebuild = True
        if os.path.exists(prep_path) and os.path.exists(meta_path):
            meta = np.load(meta_path)
            if 'version' in meta and int(meta['version']) == self.CACHE_VERSION:
                images = np.load(prep_path, mmap_mode='c')
                return images, meta['bboxes']

        images, bboxes = self._preprocess_cube(src_path)
        # atomic writes so concurrent DataLoader workers never read partials
        tmp = prep_path + f'.tmp.{os.getpid()}.npy'
        np.save(tmp, images)
        os.replace(tmp, prep_path)
        tmp = meta_path + f'.tmp.{os.getpid()}.npz'
        np.savez(tmp, bboxes=bboxes, version=self.CACHE_VERSION)
        os.replace(tmp, meta_path)
        return images, bboxes

    def __len__(self):
        return len(self.file_list)

    def get_valid_crop_limits(self, img):
        """Finds the tightest bounding box containing valid signal to avoid pure vacuum."""
        threshold = img.max() * self.threshold_ratio
        mask = img > threshold
        valid_rows = np.where(np.any(mask, axis=1))[0]
        valid_cols = np.where(np.any(mask, axis=0))[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return 0, img.shape[0], 0, img.shape[1]

        return valid_rows[0], valid_rows[-1], valid_cols[0], valid_cols[-1]

    # CHANGED v2: delegate to the shared module-level gate, which adds the
    # pedestal test (large median, tiny deviation). The old positive-pixel
    # contrast test let bright-but-flat crops through (smooth halo gradients
    # pass p99.5/p50 > 3 while carrying no usable structure).
    def _crop_is_informative(self, patch):
        return crop_is_informative(patch, bypass_prob=self.dcfg['bypass_prob'])

    def __getitem__(self, index):
        # CHANGED v5: all deterministic per-file preprocessing lives in
        # _load_prepared (cached + memory-mapped when fire_cache_dir is set).
        # CHANGED v6: the empty-file fallback is now an ITERATIVE bounded
        # scan. The previous recursive fallback hit the recursion limit when
        # many/all files came back empty (which the old absolute corruption
        # threshold could cause on perfectly good data).
        images, bboxes = None, None
        n_files = len(self.file_list)
        for probe in range(n_files):
            images, bboxes = self._load_prepared((index + probe) % n_files)
            if images.shape[0] > 0:
                break
        else:
            raise RuntimeError(
                "FireQuicklookDataset: every file in file_list produced zero "
                "usable views (all views empty/non-finite after sanitization). "
                "Check the source files, and delete the fire_cache_dir if it "
                "was built with an older version of this script.")

        # Random per-file brightness anchor (applied to the crop, see note
        # in __init__: identical to scaling the whole cube).
        target_kelvin_peak = random.uniform(0.05, 3.0)

        num_views = images.shape[0]
        x_prim_list, x_cond_list, y_list, mask_list, memory_list = [], [], [], [], []

        for p_i in range(self.num_patches):   # CHANGED v12: patch index
            # CHANGED v5: retry loop now keeps the BEST-scoring candidate if
            # no attempt passes the gate (same policy as SKIRT), and uses the
            # precomputed bounding boxes instead of rescanning the view.
            best_patch, best_score, best_c = None, -1.0, self.min_crop
            for attempt in range(self.max_crop_tries):
                v_idx = random.randint(0, num_views - 1)
                rmin, rmax, cmin, cmax = bboxes[v_idx]

                # np.array on an mmap slice copies ONLY the crop pixels
                signal_crop = np.array(images[v_idx, rmin:rmax, cmin:cmax], dtype=np.float32)
                bb_h, bb_w = signal_crop.shape

                # Pad if the galaxy is smaller than our minimum required crop.
                # After pedestal removal, constant-zero padding is physically
                # consistent with the (now zero) background.
                pad_h = max(0, self.min_crop - bb_h)
                pad_w = max(0, self.min_crop - bb_w)
                if pad_h > 0 or pad_w > 0:
                    signal_crop = np.pad(signal_crop, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')

                h, w = signal_crop.shape
                max_c = min(h, w)
                act_min, act_max = min(self.min_crop, max_c), min(self.max_crop, max_c)

                if self.mode == 'train':
                    c_size = random.randint(act_min, act_max)
                    y_off, x_off = random.randint(0, h - c_size), random.randint(0, w - c_size)
                else:
                    valid = [c for c in self.val_crop_scales if c <= max_c]
                    c_size = random.choice(valid) if valid else max_c
                    y_off, x_off = (h - c_size) // 2, (w - c_size) // 2

                candidate = signal_crop[y_off:y_off+c_size, x_off:x_off+c_size].copy()

                # Validation uses deterministic center crops; only gate in train
                if self.mode != 'train' or self._crop_is_informative(candidate):
                    best_patch, best_c = candidate, c_size
                    break

                score = crop_structure_score(candidate)
                if score > best_score:
                    best_patch, best_score, best_c = candidate, score, c_size

            final_patch = best_patch * target_kelvin_peak
            c_size = best_c

            # Pass to the master processing pipeline
            fd, fseed = val_force_diffuse(index, p_i, self.dcfg['val_fraction']) if self.mode != 'train' else (False, None)
            p1, p2, py, pm, pmem = process_noise_to_clean_patch(
                final_patch, self.augment, self.target_size, self.planck_size,
                self.total_fov_arcmin, c_size, add_point_sources=False, clean_prob=self.clean_prob, stellar_prob=self.stellar_prob,
                force_diffuse=fd, diffuse_seed=fseed, diffuse_cfg=self.dcfg, upsample_cfg=self.upcfg, noise_cfg=self.ncfg
            )

            x_prim_list.append(p1); x_cond_list.append(p2); y_list.append(py); mask_list.append(pm); memory_list.append(pmem)

        return (torch.stack(x_prim_list), torch.stack(x_cond_list), torch.stack(y_list), torch.stack(mask_list), torch.stack(memory_list), index)

# =========================================================
# WRAPPER: THREE-WAY MIXER (50% Dust, 25% SKIRT, 25% FIRE)
# =========================================================
class CombinedSyntheticDataset(Dataset):
    """Weighted mixture of the dust, SKIRT and FIRE datasets.

    CHANGED v11: a component with zero weight, or with no files, is dropped and
    the remaining weights are renormalized. Previously a zero weight was not
    expressible (the FIRE branch was the unconditional else), and an empty FIRE
    dataset silently sent its 25% share to dust, so the realised mix was not
    the requested one. DATASET_FRACTIONS_OVERRIDE at the top of this file is
    what turns FIRE off for the fiducial model without touching factory.py.
    """

    def __init__(self, dust_dataset, skirt_dataset, fire_dataset, fractions=(0.50, 0.25, 0.25)):
        if DATASET_FRACTIONS_OVERRIDE is not None:
            fractions = DATASET_FRACTIONS_OVERRIDE
        self.requested = tuple(float(f) for f in fractions)

        pool = [('dust', dust_dataset, self.requested[0]),
                ('skirt', skirt_dataset, self.requested[1]),
                ('fire', fire_dataset, self.requested[2])]

        self.names, self.datasets, weights, dropped = [], [], [], []
        for name, ds, wgt in pool:
            n = len(ds) if ds is not None else 0
            if wgt <= 0.0 or n == 0:
                dropped.append(f'{name} (weight {wgt:.2f}, {n} files)')
                continue
            self.names.append(name)
            self.datasets.append(ds)
            weights.append(wgt)

        if not self.datasets:
            raise ValueError('CombinedSyntheticDataset: every component is '
                             'empty or zero-weighted; nothing to sample.')

        total = float(sum(weights))
        self.weights = [w / total for w in weights]
        self.cum_probs = np.cumsum(self.weights).tolist()
        self.cum_probs[-1] = 1.0                  # guard against fp drift

        mix = ', '.join(f'{n} {w:.0%}' for n, w in zip(self.names, self.weights))
        print(f'[Mix] sampling {mix}'
              + (f'   dropped: {"; ".join(dropped)}' if dropped else ''))

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, index):
        r = random.random()
        for cum, ds in zip(self.cum_probs, self.datasets):
            if r < cum:
                return ds[random.randint(0, len(ds) - 1)]
        ds = self.datasets[-1]
        return ds[random.randint(0, len(ds) - 1)]