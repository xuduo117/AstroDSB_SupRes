#!/usr/bin/env python3
"""
dsb_ensemble_analyze.py  --  read the .npz from dsb_ensemble_infer.py and
                             produce the figures and numbers.  RUNS LOCALLY.

Four outputs, each skipped cleanly if its inputs are absent:

  <prefix>_posterior.pdf   input, truth, two independent draws, the ensemble
                           mean, and the per-pixel spread.  What the model's
                           uncertainty actually looks like.

  <prefix>_pk.pdf          P(k) of the individual draws as a median with a
                           16/84 band, against the input, the truth, and the
                           ENSEMBLE MEAN.  The lower panel is the mean-to-draw
                           power ratio: the production sliding window at
                           stride 32 averages about 28 effectively independent
                           draws per pixel, so the delivered map sits near the
                           mean, and this panel is how much sub-beam power that
                           costs.

  <prefix>_calibration.pdf are the error bars honest?  The standardised
                           residual (mean - truth)/sigma should be N(0,1), and
                           the PIT histogram should be flat.  A narrow z or a
                           U-shaped PIT means over-confidence; wide z or a
                           dome means under-confidence.

  <prefix>_kernel.pdf      order 1 against order 3 on the same region: P(k),
                           coherence C(k), and the three paper metrics.

P(k), C(k) and the eta / F / f_theta definitions are imported from your own
astrodsb_evaluate.py and section51_unified.py, so every number here is
computed the same way as the numbers in the manuscript.

  python dsb_ensemble_analyze.py ensemble_act_0910.npz \
      --pixel-scale 0.5 --input-beam 5.0 --prefix ensemble_act_0910
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

# All-NaN slices are expected wherever the truth is off-footprint or a float16
# save overflowed. The counts are reported explicitly on load, so the per-slice
# RuntimeWarnings add noise without adding information.
warnings.filterwarnings('ignore', r'.*[Aa]ll-NaN slice.*')
warnings.filterwarnings('ignore', r'.*Degrees of freedom <= 0.*')
warnings.filterwarnings('ignore', r'.*[Mm]ean of empty slice.*')
warnings.filterwarnings('ignore', r'.*invalid value encountered.*')

# ---- fixed series colours.  Validated for colour-vision deficiency: worst
# ---- adjacent pair dE 10.7 (deuteranopia), 16.5 (tritanopia), all five above
# ---- the 3:1 contrast floor.  Assigned by role and never cycled.
C_INPUT = '#2E77C4'      # blue    - low-resolution input
C_DRAW = '#CE7310'      # orange  - individual posterior draw
C_MEAN = '#8352A8'      # purple  - ensemble mean
C_ORD1 = '#189AA0'      # teal    - interpolation order 1
C_ORD3 = '#C2255C'      # crimson - interpolation order 3
C_TRUTH = '#1A1A1A'      # ink     - ground truth, a reference not a category
INK, MUTED = '#1A1A1A', '#6B6B6B'


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('npz')
    p.add_argument('--prefix', default=None, help="output stem, default from the npz name")
    p.add_argument('--pixel-scale', type=float, default=0.5, help="arcmin per pixel")
    p.add_argument('--input-beam', type=float, default=5.0,
                   help="FWHM in arcmin of the input beam, sets k_beam for F")
    p.add_argument('--patch', type=int, default=0, help="patch index shown in the image figure")
    p.add_argument('--eval-dir', default=None,
                   help="folder holding astrodsb_evaluate.py and section51_unified.py; "
                        "default is two levels up from this script")
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--unc-kernels', default=None,
                   help="comma list of variants entering the KERNEL term, e.g. "
                        "planck,zoom1,zoom3. Default: all of them. Leave out "
                        "any variant you would never actually deploy "
                        "(hpx_nearest, say) or it inflates the term.")
    return p.parse_args()


def load_eval(eval_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [eval_dir] if eval_dir else []
    cand += [os.path.abspath(os.path.join(here, '..', '..')),
             os.path.abspath(os.path.join(here, '..')), here, os.getcwd()]
    for d in cand:
        if d and os.path.isfile(os.path.join(d, 'astrodsb_evaluate.py')):
            sys.path.insert(0, d)
            import astrodsb_evaluate as ae
            try:
                import section51_unified as s51
            except ImportError:
                s51 = None
            print(f"[eval] using {d}/astrodsb_evaluate.py"
                  + ("" if s51 else "  (section51_unified.py not found, "
                                    "eta / F / f_theta will be skipped)"))
            return ae, s51
    sys.exit("could not find astrodsb_evaluate.py; pass --eval-dir")


def style(ax):
    """Recessive axes and grid so the data carries the figure."""
    ax.grid(True, which='both', lw=0.5, alpha=0.25, color=MUTED)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(INK)


def pk_of(ae, ref, arr, px):
    """P(k) of `arr`, and its coherence with `ref`."""
    k, P_ref, P_arr, C = ae.compute_pk_ck(ref, arr, pixel_scale=px)
    return k, P_ref, P_arr, C


def main():
    a = parse_args()
    prefix = a.prefix or os.path.splitext(os.path.basename(a.npz))[0]
    ae, s51 = load_eval(a.eval_dir)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(a.npz, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    key = 'variants' if 'variants' in d.files else 'orders'
    orders = [str(o) for o in d[key]]
    if key == 'orders':                      # npz from before the rename
        orders = [f'order{o}' for o in orders]
    gt = d['gt'] if 'gt' in d.files else None
    px = a.pixel_scale
    print(f"[data] orders {orders} | {meta['n_samples']} draws | "
          f"epoch {meta['epoch']} | truth {'yes' if gt is not None else 'no'}")

    S = {}
    for o in orders:
        stem = o if f'{o}_samples' in d.files else f'order{o}'
        S[o] = dict(samples=np.asarray(d[f'{stem}_samples'], dtype=np.float32),
                    inp=np.asarray(d[f'{stem}_input'], dtype=np.float32))
        S[o]['mean'] = S[o]['samples'].mean(axis=0)
        S[o]['std'] = S[o]['samples'].std(axis=0)

    # ---- what actually arrived ---------------------------------------------
    # Two things silently wreck every statistic below and neither announces
    # itself: inf from a float16 save (these maps reach 1e6 in uK_CMB, float16
    # stops at 65504), and the exactly-zero pixels off the ACT footprint.
    # np.nanstd ignores nan but NOT inf, so one inf turns a summary into nan.
    bad_any = False
    for o in orders:
        sm = S[o]['samples']
        n_inf = int(np.isinf(sm).sum())
        n_nan = int(np.isnan(sm).sum())
        if n_inf or n_nan:
            bad_any = True
            print(f"[warn] {o}: {n_inf} inf and {n_nan} nan in the samples "
                  f"({(n_inf+n_nan)/sm.size*100:.3f}%)")
            if n_inf:
                print("       inf almost always means --save-dtype float16 "
                      "overflowed. Re-run with float32;")
                print("       the affected pixels are the brightest ones, so "
                      "they are not ignorable.")
        # inf would poison every mean and std, so drop it to nan here
        S[o]['samples'] = np.where(np.isfinite(sm), sm, np.nan)
        S[o]['mean'] = np.nanmean(S[o]['samples'], axis=0)
        S[o]['std'] = np.nanstd(S[o]['samples'], axis=0)

    # Valid = the truth is real there. compute_pk_ck already applies this mask
    # internally, but the calibration statistics did not, and on a crop that
    # clips the survey edge the off-footprint zeros dominate them.
    if gt is not None:
        valid = np.isfinite(gt) & (gt != 0.0)
        frac = float(valid.mean())
        print(f"[data] truth is live on {frac*100:.1f}% of the crop"
              + ("" if frac > 0.999 else
                 f"; the other {(1-frac)*100:.1f}% is off the ACT footprint "
                 f"and is excluded"))
        if frac < 0.9:
            print("       That is a large fraction. Those pixels are exactly "
                  "0.0, so any statistic that")
            print("       does not mask them compares the prediction against "
                  "zero and reports nonsense.")
            print("       Consider re-cropping somewhere fully inside the "
                  "footprint.")
    else:
        valid = None

    # the variant shown in the single-variant figures: prefer the one that
    # reproduces production, then anything cubic, then whatever came first
    main_o = next((o for o in ('planck', 'hpx_bilinear', 'order3', 'zoom3',
                               'asis') if o in S), orders[0])
    M = S[main_o]
    n_s, n_p = M['samples'].shape[:2]
    ip = min(a.patch, n_p - 1)
    summary = dict(meta=meta, pixel_scale=px, input_beam_arcmin=a.input_beam,
                   n_samples=int(n_s), n_patches=int(n_p), orders=orders)

    # ------------------------------------------------ 1. what it looks like
    panels = [(M['inp'][ip], 'low-resolution input', 'magma', None)]
    if gt is not None:
        panels.append((gt[ip], 'ground truth', 'magma', None))
    panels += [(M['samples'][0, ip], 'posterior draw 1', 'magma', None),
               (M['samples'][1 % n_s, ip], 'posterior draw 2', 'magma', None),
               (M['mean'][ip], f'ensemble mean of {n_s}', 'magma', None),
               (M['std'][ip], 'per-pixel spread', 'Purples', 'sigma')]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    ref_imgs = [p[0] for p in panels if p[3] is None]
    vmin = float(np.nanpercentile(np.concatenate([r.ravel() for r in ref_imgs]), 1))
    vmax = float(np.nanpercentile(np.concatenate([r.ravel() for r in ref_imgs]), 99.5))
    for ax, (img, ttl, cmap, kind) in zip(axes, panels):
        kw = dict(cmap=cmap) if kind else dict(cmap=cmap, vmin=vmin, vmax=vmax)
        im = ax.imshow(img, origin='lower', **kw)
        ax.set_title(ttl, fontsize=10.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        if kind:
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8, colors=MUTED)
    for ax in axes[len(panels):]:
        ax.axis('off')
    # nanstd does not ignore inf, so compute both on finite pixels only
    _m, _sd = M['mean'][ip], M['std'][ip]
    _f = np.isfinite(_m) & np.isfinite(_sd)
    if valid is not None and valid.shape == _m.shape:
        _f &= valid
    rel = (float(np.median(_sd[_f]) / (np.std(_m[_f]) + 1e-12))
           if _f.sum() > 100 else float('nan'))
    fig.suptitle(f"Astro-DSB posterior, {main_o}, patch {ip}"
                 f"    median spread = {rel*100:.1f}% of the predicted structure",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.2)
    fig.savefig(f'{prefix}_posterior.pdf', dpi=a.dpi)
    plt.close(fig)
    print(f"[fig ] {prefix}_posterior.pdf")
    summary['median_spread_frac_of_predicted_sigma'] = rel

    # ------------------------------- 2. P(k) with a posterior band
    ref = gt[ip] if gt is not None else M['mean'][ip]
    k, P_ref, _, _ = pk_of(ae, ref, M['inp'][ip], px)
    _, _, P_inp, _ = pk_of(ae, ref, M['inp'][ip], px)
    P_draws = np.stack([pk_of(ae, ref, M['samples'][j, ip], px)[2]
                        for j in range(n_s)])
    _, _, P_mean, _ = pk_of(ae, ref, M['mean'][ip], px)
    lo, med, hi = np.percentile(P_draws, [16, 50, 84], axis=0)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.6, 1]))
    if gt is not None:
        ax.plot(k, P_ref, lw=2.4, color=C_TRUTH, label='ground truth', zorder=5)
    ax.plot(k, P_inp, lw=2, color=C_INPUT, ls='--', label='low-resolution input')
    ax.fill_between(k, lo, hi, color=C_DRAW, alpha=0.28, lw=0,
                    label=f'posterior draws, 16-84 percentile (n={n_s})')
    ax.plot(k, med, lw=2, color=C_DRAW, label='posterior median')
    ax.plot(k, P_mean, lw=2, color=C_MEAN, ls='-.',
            label='ensemble mean map')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylabel('P(k)', fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False, labelcolor=INK)
    ax.set_title(f'Power spectrum of individual draws against their mean '
                 f'({main_o}, patch {ip})', fontsize=12, color=INK)
    style(ax)

    ratio = P_mean / np.clip(med, 1e-30, None)
    axr.plot(k, ratio, lw=2, color=C_MEAN)
    axr.axhline(1.0, lw=1, color=MUTED, ls=':')
    kb = ae.beam_k_cutoff(a.input_beam)
    axr.axvline(kb, lw=1, color=MUTED, ls='--')
    axr.text(kb * 0.94, 0.08, f'input beam {a.input_beam:.1f} arcmin',
             fontsize=8, color=MUTED, ha='right',
             transform=axr.get_xaxis_transform())
    axr.set_xscale('log'); axr.set_ylim(0, 1.35)
    axr.set_xlabel('k  (cycles / arcmin)', fontsize=11, color=INK)
    axr.set_ylabel('mean / draw', fontsize=10, color=INK)
    style(axr)
    sub = k > kb
    loss = float(np.median(ratio[sub])) if sub.any() else np.nan
    axr.set_title(f'averaging the posterior keeps {loss*100:.0f}% of the '
                  f'sub-beam power of a single draw', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(f'{prefix}_pk.pdf', dpi=a.dpi)
    plt.close(fig)
    print(f"[fig ] {prefix}_pk.pdf")
    summary['subbeam_power_kept_by_mean'] = loss

    # ------------------------------------------------- 3. calibration
    if gt is not None:
        # Off-footprint pixels are exactly 0.0 in the ACT arrays. Including
        # them compares the prediction against zero, which produces a huge
        # residual and piles the truth at rank 0 -- an artefact that reads
        # exactly like over-confidence. Mask before judging calibration.
        vm = valid if valid is not None else np.ones_like(gt, dtype=bool)
        vm = vm & np.isfinite(M['mean']) & np.isfinite(M['std'])
        kept = float(vm.mean())
        z = ((M['mean'] - gt) / np.clip(M['std'], 1e-12, None))[vm]
        z = z[np.isfinite(z)]
        rank = (M['samples'] < gt[None]).sum(axis=0)[vm]          # 0..n_s
        print(f"[calb] {kept*100:.1f}% of pixels used "
              f"({(1-kept)*100:.1f}% off-footprint or non-finite)")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
        ax1.hist(np.clip(z, -6, 6), bins=80, density=True, color=C_DRAW,
                 alpha=0.75, lw=0)
        xs = np.linspace(-6, 6, 300)
        ax1.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), lw=2,
                 color=C_TRUTH, label='N(0,1), perfect calibration')
        ax1.set_xlabel('(mean - truth) / sigma', fontsize=11, color=INK)
        ax1.set_ylabel('density', fontsize=11, color=INK)
        ax1.legend(fontsize=9, frameon=False, labelcolor=INK)
        ax1.set_title(f'standardised residual, sigma(z) = {z.std():.2f}',
                      fontsize=11, color=INK)
        style(ax1)

        ax2.hist(rank, bins=np.arange(n_s + 2) - 0.5, density=True,
                 color=C_MEAN, alpha=0.75, lw=0)
        ax2.axhline(1.0 / (n_s + 1), lw=2, color=C_TRUTH,
                    label='flat, perfect calibration')
        ax2.set_xlabel('rank of the truth within the ensemble', fontsize=11, color=INK)
        ax2.set_ylabel('frequency', fontsize=11, color=INK)
        ax2.legend(fontsize=9, frameon=False, labelcolor=INK)
        verdict = ('over-confident, spread too small' if z.std() > 1.15 else
                   'under-confident, spread too large' if z.std() < 0.85 else
                   'calibrated')
        ax2.set_title(f'PIT histogram  ({verdict})', fontsize=11, color=INK)
        style(ax2)
        fig.tight_layout()
        fig.savefig(f'{prefix}_calibration.pdf', dpi=a.dpi)
        plt.close(fig)
        print(f"[fig ] {prefix}_calibration.pdf")
        summary['calibration_pixels_used_frac'] = kept
        summary['z_std'] = float(z.std())
        summary['z_mean'] = float(z.mean())
        summary['calibration'] = verdict
    else:
        print("[skip] calibration needs --gt at inference time")

    # --------------------------------------- 4. interpolation kernel
    num = orders if len(orders) >= 2 else []
    if len(num) >= 2 and gt is not None:
        if len(num) > 5:
            print(f"[warn] {len(num)} variants; plotting the first 5 to keep "
                  f"the colours distinguishable")
            num = num[:5]
        fig = plt.figure(figsize=(12.4, 6.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.0], hspace=0.42)
        axp, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        axt = fig.add_subplot(gs[1, :]); axt.axis('off')
        # fixed assignment by role, never cycled into generated hues
        pinned = {'order1': C_ORD1, 'zoom1': C_ORD1, 'hpx_nearest': C_ORD1,
                  'order3': C_ORD3, 'zoom3': C_ORD3,
                  'planck': C_INPUT, 'asis': C_INPUT,
                  'hpx_bilinear': C_MEAN}
        spare = [C_DRAW, C_MEAN, C_ORD1, C_ORD3, C_INPUT]
        cols, used = {}, set()
        for o in num:
            c = pinned.get(o)
            if c is None or c in used:
                c = next((x for x in spare if x not in used), C_DRAW)
            cols[o] = c; used.add(c)
        rows = []
        axp.plot(k, P_ref, lw=2.4, color=C_TRUTH, label='ground truth', zorder=5)
        for o in num:
            col = cols[o]
            Pd = np.stack([pk_of(ae, gt[ip], S[o]['samples'][j, ip], px)[2]
                           for j in range(n_s)])
            l16, l50, l84 = np.percentile(Pd, [16, 50, 84], axis=0)
            ko, Pg, Pi, Ci = pk_of(ae, gt[ip], S[o]['inp'][ip], px)
            _, _, _, Cd = pk_of(ae, gt[ip], S[o]['mean'][ip], px)
            axp.fill_between(ko, l16, l84, color=col, alpha=0.22, lw=0)
            axp.plot(ko, l50, lw=2, color=col, label=f'prediction, {o}')
            axp.plot(ko, Pi, lw=1.6, ls='--', color=col, alpha=0.8,
                     label=f'input, {o}')
            axc.plot(ko, Ci, lw=1.6, ls='--', color=col, alpha=0.8,
                     label=f'input, {o}')
            axc.plot(ko, Cd, lw=2, color=col, label=f'prediction, {o}')
            r = dict(order=o)
            if s51 is not None:
                kb_ = ae.beam_k_cutoff(a.input_beam)
                _, _, eta = s51.compute_M1_eta(ko, Ci, Cd)
                F, _, _ = s51.compute_M2_F(ko, Pg, Pi, l50, kb_)
                # returns (theta90_input, theta90_dsb, f_theta) in that order
                th_in, th_dsb, fth = s51.compute_M3_ftheta(ko, Ci, Cd)
                r.update(eta=float(eta), F=float(F), f_theta=float(fth),
                         theta90_in=float(th_in), theta90_dsb=float(th_dsb))
            rows.append(r)
        axp.set_xscale('log'); axp.set_yscale('log')
        axp.set_xlabel('k  (cycles / arcmin)', fontsize=11, color=INK)
        axp.set_ylabel('P(k)', fontsize=11, color=INK)
        axp.set_title('does the interpolation kernel change the recovery?',
                      fontsize=12, color=INK)
        axp.legend(fontsize=8, frameon=False, labelcolor=INK, ncol=2)
        style(axp)
        axc.set_xscale('log'); axc.set_ylim(0, 1.02)
        axc.axhline(0.9, lw=1, color=MUTED, ls=':')
        axc.set_xlabel('k  (cycles / arcmin)', fontsize=11, color=INK)
        axc.set_ylabel('coherence C(k)', fontsize=11, color=INK)
        axc.set_title('phase agreement with the truth', fontsize=12, color=INK)
        axc.legend(fontsize=8, frameon=False, labelcolor=INK,
                   loc='lower left', ncol=2)
        style(axc)
        if rows and 'eta' in rows[0]:
            tbl = axt.table(
                cellText=[[r['order'], f"{r['eta']:.2f}", f"{r['F']:.3f}",
                           f"{r['f_theta']:.2f}", f"{r['theta90_dsb']:.2f}"]
                          for r in rows],
                colLabels=['variant', 'eta', 'F', 'f_theta',
                           'theta90_DSB (arcmin)'],
                cellLoc='center', loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.45)
            for (ri, _), cell in tbl.get_celld().items():
                cell.set_edgecolor('#DDDDDD')
                cell.set_text_props(color=INK)
                if ri == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#F4F4F2')
            # no title here: the column headers say it, and a title
            # at this position collides with the header row
        # explicit margins rather than tight_layout: the table axis is not
        # compatible with it and matplotlib warns
        fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.06,
                            wspace=0.22)
        fig.savefig(f'{prefix}_kernel.pdf', dpi=a.dpi)
        plt.close(fig)
        print(f"[fig ] {prefix}_kernel.pdf")
        summary['kernel'] = rows
    elif len(num) >= 2:
        print("[skip] the kernel metrics need --gt at inference time")

    # ------------------------------------- 5. where the uncertainty comes from
    # Law of total variance over the design (kernel k, seed s):
    #
    #     Var_{k,s}[x]  =  E_k[ Var_s(x) ]   +   Var_k[ E_s(x) ]
    #                      \_____________/       \_____________/
    #                       the generative         the choice of
    #                       process, at fixed      interpolation kernel,
    #                       input   -> MODEL       at fixed process
    #                                              -> DATA
    #
    # Both terms use ddof=0, so the identity holds exactly and the two maps add
    # to the total. Var_k is the spread of the K kernels you actually ran, a
    # sensitivity to a discrete choice, NOT a sample from some population of
    # kernels: with three or four variants it is a real measurement of "does the
    # preprocessing matter", not an error bar to quote as sigma.
    #
    # This is only meaningful if draw i used the same noise realisation in every
    # variant. dsb_ensemble_infer.py guarantees that and records paired_seeds.
    unc_k = orders if not a.unc_kernels else \
        [v.strip() for v in a.unc_kernels.split(',') if v.strip() in S]
    if a.unc_kernels and len(unc_k) < 2:
        print(f"[warn] --unc-kernels matched {unc_k}; falling back to all")
        unc_k = orders
    if len(unc_k) >= 2 and n_s >= 2:
        if not meta.get('paired_seeds', False):
            print("[warn] this npz does not record paired_seeds. If the "
                  "variants were sampled with")
            print("       different noise, the split below attributes seed "
                  "noise to the kernel and is")
            print("       not trustworthy. Re-run with the current "
                  "dsb_ensemble_infer.py.")
        if unc_k != orders:
            print(f"[unc ] kernel term over {unc_k} "
                  f"(excluded: {[o for o in orders if o not in unc_k]})")
        A = np.stack([S[o]['samples'][:, ip] for o in unc_k])      # (K, n_s, H, W)
        K = A.shape[0]
        mu_k = np.nanmean(A, axis=1)                               # (K, H, W)
        var_s_k = np.nanvar(A, axis=1)                             # within kernel
        v_alea = np.nanmean(var_s_k, axis=0)       # reverse-SDE spread
        v_kern = np.nanvar(mu_k, axis=0)            # interpolation kernel
        sig_alea, sig_kern = np.sqrt(v_alea), np.sqrt(v_kern)

        # third term, if MC dropout was run: weights vary, noise held fixed
        sig_epi = None
        if 'dropout_samples' in d.files:
            D = np.asarray(d['dropout_samples'], dtype=np.float32)[:, ip]
            D = np.where(np.isfinite(D), D, np.nan)
            dv = str(d['dropout_variant'][0]) if 'dropout_variant' in d.files \
                else '?'
            sig_epi = np.sqrt(np.nanvar(D, axis=0))
            print(f"[unc ] epistemic term from {D.shape[0]} MC-dropout draws "
                  f"on '{dv}'")
        grand = np.nanmean(mu_k, axis=0)
        v_tot = v_alea + v_kern
        frac_kern = np.where(v_tot > 0, v_kern / np.maximum(v_tot, 1e-30), np.nan)

        vmask = np.isfinite(grand) & np.isfinite(v_tot)
        if valid is not None and valid.shape == grand.shape:
            vmask &= valid
        med_m = float(np.median(sig_alea[vmask])) if vmask.sum() else float('nan')
        med_d = float(np.median(sig_kern[vmask])) if vmask.sum() else float('nan')
        med_e = (float(np.median(sig_epi[vmask]))
                 if sig_epi is not None and vmask.sum() else None)
        med_f = float(np.median(frac_kern[vmask & np.isfinite(frac_kern)])) \
            if vmask.sum() else float('nan')
        struct = float(np.std(grand[vmask])) if vmask.sum() else float('nan')

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(13.4, 8.6))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        show = [(grand, f'ensemble mean, {K} kernels x {n_s} draws', 'magma', False),
                (sig_alea, 'ALEATORIC\nreverse-SDE noise, weights and input fixed',
                 'Purples', True),
                (sig_kern, 'KERNEL\ninterpolation choice: '
                           + ', '.join(unc_k), 'YlOrBr', True)]
        if sig_epi is not None:
            show.append((sig_epi, 'EPISTEMIC\nMC dropout, SDE noise frozen',
                         'GnBu', True))
        vlo = float(np.nanpercentile(grand[vmask], 1)) if vmask.sum() else 0.0
        vhi = float(np.nanpercentile(grand[vmask], 99.5)) if vmask.sum() else 1.0
        # one scale for both sigma maps, so the comparison is by eye
        sv = max([float(np.nanpercentile(x[vmask], 99.5)) if vmask.sum() else 1.0
                  for x in ([sig_alea, sig_kern] +
                            ([sig_epi] if sig_epi is not None else []))])
        for j, (img, ttl, cmap, is_sig) in enumerate(show):
            ax = fig.add_subplot(gs[0, j] if j < 3 else gs[1, 0])
            kw = dict(cmap=cmap, vmin=0, vmax=sv) if is_sig else \
                dict(cmap=cmap, vmin=vlo, vmax=vhi)
            im = ax.imshow(img, origin='lower', **kw)
            ax.set_title(ttl, fontsize=10.5, color=INK)
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8, colors=MUTED)

        ax = fig.add_subplot(gs[1, 1] if sig_epi is not None else gs[1, 0])
        im = ax.imshow(frac_kern, origin='lower', cmap='coolwarm', vmin=0, vmax=1)
        ax.set_title('kernel share of (aleatoric + kernel)\n'
                     'blue = the SDE dominates, red = the kernel does',
                     fontsize=10.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8, colors=MUTED)

        # --- at which angular scales does each term live ---------------------
        axs = fig.add_subplot(gs[1, 2] if sig_epi is not None else gs[1, 1:])
        ref = gt[ip] if gt is not None else grand
        nd = min(n_s, 6)                 # the spectra converge fast; cap the cost
        try:
            Pm = np.mean([pk_of(ae, ref, A[k, j] - mu_k[k], px)[2]
                          for k in range(K) for j in range(nd)], axis=0)
            Pd = np.mean([pk_of(ae, ref, mu_k[k] - grand, px)[2]
                          for k in range(K)], axis=0)
            kk, _, Pmean, _ = pk_of(ae, ref, grand, px)
            axs.loglog(kk, Pmean, lw=2, color=C_TRUTH, label='ensemble mean')
            axs.loglog(kk, Pm, lw=2, color=C_MEAN, label='aleatoric (SDE)')
            axs.loglog(kk, Pd, lw=2, color=C_DRAW, label='kernel')
            if a.input_beam:
                axs.axvline(ae.beam_k_cutoff(a.input_beam), lw=1, ls=':',
                            color=MUTED)
                axs.annotate('input beam', xy=(ae.beam_k_cutoff(a.input_beam), 1),
                             xycoords=('data', 'axes fraction'),
                             xytext=(-4, -12), textcoords='offset points',
                             ha='right', fontsize=8, color=MUTED)
            axs.set_xlabel('k  (cycles / arcmin)', fontsize=11, color=INK)
            axs.set_ylabel('variance power', fontsize=11, color=INK)
            axs.set_title('which scales each source of uncertainty lives at',
                          fontsize=11, color=INK)
            axs.legend(fontsize=9, frameon=False, labelcolor=INK)
            style(axs)
        except Exception as e:                                   # noqa: BLE001
            axs.axis('off')
            axs.text(0.5, 0.5, f'spectral split unavailable\n{e}', ha='center',
                     va='center', fontsize=9, color=MUTED)

        ttl = (f"Astro-DSB uncertainty    median sigma: aleatoric {med_m:.4g}, "
               f"kernel {med_d:.4g}"
               + (f", epistemic {med_e:.4g}" if med_e is not None else "")
               + f"    kernel share {med_f*100:.1f}%"
               f"    (structure sigma {struct:.4g})")
        fig.suptitle(ttl, fontsize=12, color=INK)
        fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.07,
                            wspace=0.22, hspace=0.22)
        fig.savefig(f'{prefix}_uncertainty.pdf', dpi=a.dpi)
        plt.close(fig)
        print(f"[fig ] {prefix}_uncertainty.pdf")
        summary['uncertainty'] = dict(
            n_kernels=int(K), kernels=list(unc_k), n_draws=int(n_s),
            median_sigma_aleatoric=med_m, median_sigma_kernel=med_d,
            median_sigma_epistemic=med_e,
            median_kernel_variance_share=med_f,
            structure_sigma=struct,
            aleatoric_over_structure=med_m / (struct + 1e-30),
            kernel_over_structure=med_d / (struct + 1e-30),
            epistemic_over_structure=(None if med_e is None
                                      else med_e / (struct + 1e-30)),
            paired_seeds=bool(meta.get('paired_seeds', False)))
        print(f"       aleatoric/structure {med_m/(struct+1e-30):.2e}   "
              f"kernel/structure {med_d/(struct+1e-30):.2e}"
              + ("" if med_e is None else
                 f"   epistemic/structure {med_e/(struct+1e-30):.2e}"))
    elif len(unc_k) < 2:
        print("[skip] the uncertainty split needs at least two variants")
    else:
        print("[skip] the uncertainty split needs at least two draws")

    with open(f'{prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[out ] {prefix}_summary.json")
    for key in ('median_spread_frac_of_predicted_sigma', 'subbeam_power_kept_by_mean',
                'z_std', 'calibration'):
        if key in summary:
            print(f"       {key} = {summary[key]}")
    if 'kernel' in summary:
        for r in summary['kernel']:
            print(f"       {r['order']}: " +
                  "  ".join(f"{k} {v:.3f}" for k, v in r.items() if k != 'order'))


if __name__ == '__main__':
    main()