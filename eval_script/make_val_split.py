#!/usr/bin/env python3
"""
make_val_split.py  --  regenerate the Validation_Split arrays with Planck from
                       Peter's folder, so the whole ACT test runs on one release.
                       RUNS ON TRILLIUM (that is where the data is).

This is extract_val_region.py brought up to date. Same geometry, same
conventions, three differences:

  * Planck comes from Peter's cleaned maps rather than R4.00, found by glob from
    the DIRECTORY you pass and with its unit read from TUNIT in the header.
  * A .fits is written beside each .npy, carrying the WCS and the provenance.
    The missing WCS on the old .npy files is why a crop taken from them could
    not be checked or re-sampled, and why the notebook indices were ambiguous.
  * The new arrays are compared against the old ones before anything is
    replaced, so a geometry change cannot pass silently.

READ THIS BEFORE USING --replace

The stored predictions were produced FROM the old input. Every
Recon_Planck_Val_Input.npy under final_super_res_maps* corresponds to the R4.00
array, not to this one. Overwrite the input on its own and the notebook will
happily load a Peter-derived input alongside an R4.00-derived prediction, crop
both at the same indices, and report metrics for a pair that never went through
the model together. Nothing errors. The numbers are simply wrong.

So the consistent end state is BOTH regenerated:

    1. run this, which writes tagged files and touches nothing else
    2. re-run inference on the new Planck_Val_Input_<tag>.npy
    3. only then --replace, which moves the old canonical files aside

This script refuses --replace unless you pass --predictions-regenerated, which
is there purely to make step 2 a thing you have to have done rather than a thing
you meant to do.

TYPICAL USE

  python make_val_split.py \
      --act-fits /scratch/xuduo117/Real_Obs_Data/Planck_Data/act-planck_dr6.02_coadd_AA_night_f220_map.fits \
      --healpix  /scratch/xuduo117/Real_Obs_Data/Planck_Data/Planck_Peter --band 217 \
      --outdir   /scratch/xuduo117/Project_DSB/AstroDSB_SupRes_3/act_inference_data/Validation_Split \
      --tag 0913
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
try:
    from crop_region import (val_split_patch, resolve_healpix,   # noqa: E402
                             healpix_unit_scale, UNSEEN_LIMIT)
except ImportError:
    sys.exit("make_val_split.py needs crop_region.py beside it (it reuses the "
             "validation-geometry rebuild, so the two cannot drift apart)")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--act-fits', required=True, help="the raw ACT coadd FITS")
    p.add_argument('--healpix', required=True,
                   help="Peter's Planck directory, or a single HFI_SkyMap FITS")
    p.add_argument('--band', type=int, default=217, help="Planck band in GHz")
    p.add_argument('--outdir', required=True, help="the Validation_Split folder")
    p.add_argument('--tag', default=time.strftime('%m%d'),
                   help="suffix for the new files, e.g. 0913 gives "
                        "Planck_Val_Input_0913.npy")

    g = p.add_argument_group("geometry")
    g.add_argument('--split-frac', type=float, default=0.70,
                   help="column split, as in extract_val_region.py")
    g.add_argument('--val-bbox', help="y0,y1,x0,x1, to skip the bbox scan")
    g.add_argument('--chunk', type=int, default=256,
                   help="rows sampled at a time. The whole patch at once would "
                        "need about 8 GB for the interpolation weights alone.")
    g.add_argument('--nan-policy', default='zero', choices=['zero', 'nan'],
                   help="what to write off the footprint. 'zero' matches "
                        "extract_val_region.py and the existing arrays; 'nan' is "
                        "better for P(k) but makes the new files behave "
                        "differently from the old ones.")

    g = p.add_argument_group("outputs and safety")
    g.add_argument('--no-fits', action='store_true',
                   help="skip the .fits companions. Not recommended: they are "
                        "what carries the WCS.")
    g.add_argument('--skip-act', action='store_true',
                   help="only rebuild the Planck side. ACT does not depend on "
                        "the Planck release, so this is safe and faster.")
    g.add_argument('--replace', action='store_true',
                   help="also take over the canonical unsuffixed names, moving "
                        "the existing files aside first. Read the header.")
    g.add_argument('--predictions-regenerated', action='store_true',
                   help="confirms inference has been re-run on the new input. "
                        "Required by --replace.")
    g.add_argument('--recon-scan', default=None,
                   help="folder to scan for stale Recon_Planck_Val_Input*.npy. "
                        "Default: two levels up from --outdir.")
    return p.parse_args()


def sample_planck(patch, hp_map, scale, chunk, nan_policy):
    """Bilinear from the sphere onto the validation grid, in row chunks.

    Identical mathematics to extract_val_region.py, which used hp.get_interp_val;
    get_interp_weights plus the weighted sum is the same 4-point interpolation
    written out, and is what act_planck_reproject.py uses.
    """
    import healpy as hp
    nside = hp.get_nside(hp_map)
    H, W = patch.shape
    out = np.empty((H, W), dtype=np.float32)
    rot = hp.Rotator(coord=['C', 'G'])
    t0 = time.time()
    for y in range(0, H, chunk):
        y1 = min(y + chunk, H)
        sub = patch[y:y1]
        dec, ra = sub.posmap()
        th, ph = rot((np.pi / 2.0 - dec).ravel(), ra.ravel())
        pix, wgt = hp.get_interp_weights(nside, th, ph, nest=False)
        vals = np.einsum('ij,ij->j', hp_map[pix], wgt)
        out[y:y1] = (vals * scale).reshape(y1 - y, W)
        done = y1 / H
        el = time.time() - t0
        print(f"       rows {y1:>6}/{H}  {done*100:5.1f}%  "
              f"elapsed {el/60:.1f} min, eta {el*(1-done)/max(done,1e-9)/60:.1f} min",
              flush=True)
    if nan_policy == 'zero':
        bad = ~np.isfinite(out)
        if bad.any():
            print(f"       {int(bad.sum())} non-finite pixels -> 0.0 "
                  f"(--nan-policy zero, matching the old arrays)")
            out[bad] = 0.0
    return out


def write_fits(path, arr, wcs, hdr_extra):
    from astropy.io import fits
    h = wcs.to_header()
    for k, v in hdr_extra.items():
        h[k] = v
    fits.PrimaryHDU(data=arr, header=h).writeto(path, overwrite=True)


def compare(new, old_path, label):
    """New against old: same sky, different release, so correlation is the test."""
    if not os.path.isfile(old_path):
        print(f"       {label:<8} no old file at {os.path.basename(old_path)}, "
              f"nothing to compare")
        return None
    old = np.load(old_path, mmap_mode='r')
    if old.shape != new.shape:
        print(f"       {label:<8} SHAPE CHANGED {old.shape} -> {new.shape}")
        print("                the geometry is not what produced the old files; "
              "stop and check --act-fits and --split-frac")
        return dict(shape_changed=True, old_shape=list(old.shape))
    good = np.isfinite(old) & np.isfinite(new) & (np.asarray(old) != 0.0)
    if good.sum() < 1000:
        print(f"       {label:<8} too few comparable pixels")
        return None
    a = np.asarray(old)[good].astype(np.float64)
    b = np.asarray(new)[good].astype(np.float64)
    r = float(np.corrcoef(a, b)[0, 1])
    rel = float(np.std(a - b) / (np.std(a) + 1e-30))
    print(f"       {label:<8} vs {os.path.basename(old_path)}:  "
          f"r = {r:.6f}   rms difference {rel*100:.3f}% of the old rms")
    if r < 0.9:
        # Two releases of the same sky agree to better than a per cent in
        # correlation.  This does not.
        print(f"       {'':<8} WARNING: r = {r:.3f} is far too low for a release "
              f"change. The new array")
        print(f"       {'':<8} is not the same sky as the old one. Check "
              f"--act-fits, --split-frac and")
        print(f"       {'':<8} that --healpix points at the band you think it "
              f"does, before using this.")
    return dict(corr=r, rms_frac=rel, against=os.path.basename(old_path),
                suspicious=bool(r < 0.9))


def main():
    a = parse_args()
    import healpy as hp
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag if a.tag.startswith('_') or not a.tag else '_' + a.tag
    if a.replace and not a.predictions_regenerated:
        sys.exit(
            "--replace without --predictions-regenerated.\n\n"
            "Every Recon_Planck_Val_Input.npy you have was produced from the OLD\n"
            "input. Replacing the input now makes each of them silently mismatched\n"
            "with it: the notebook will load the new input beside an old prediction\n"
            "and report metrics for a pair that never went through the model\n"
            "together. Run this without --replace, re-run inference on the tagged\n"
            "file it writes, then come back with both flags.")

    # ---- geometry ----------------------------------------------------------
    patch = val_split_patch(a.act_fits, a.split_frac, a.val_bbox)
    H, W = patch.shape
    wcs = patch.wcs

    # ---- Planck ------------------------------------------------------------
    hp_path = resolve_healpix(a.healpix, a.band)
    scale, why = healpix_unit_scale(hp_path, a.band, None)
    print(f"[hpx ] {os.path.basename(hp_path)}")
    print(f"[hpx ] unit scale {scale:g}  ({why})")
    m = np.asarray(hp.read_map(hp_path, field=0, nest=False, dtype=np.float32),
                   dtype=np.float32)
    bad = ~np.isfinite(m) | (m < UNSEEN_LIMIT)
    if bad.any():
        m[bad] = np.nan
        print(f"[hpx ] {int(bad.sum())} UNSEEN pixels -> NaN")
    n = H * W
    cnt = f"{n/1e6:.0f}M" if n >= 1e6 else f"{n}"
    print(f"[hpx ] nside {hp.get_nside(m)}; sampling {H} x {W} = "
          f"{cnt} positions in chunks of {a.chunk} rows")
    planck = sample_planck(patch, m, scale, a.chunk, a.nan_policy)
    del m

    # ---- ACT ---------------------------------------------------------------
    act = None
    if not a.skip_act:
        act = np.asarray(patch, dtype=np.float32)
        if a.nan_policy == 'zero':
            act[~np.isfinite(act)] = 0.0
        live = float((np.isfinite(act) & (act != 0.0)).mean())
        print(f"[act ] {act.shape}, {live*100:.1f}% live pixels")

    # ---- compare with what is already there --------------------------------
    print("[cmp ] new against old. A high correlation with a small rms "
          "difference is the")
    print("       expected signature of a release change; a low correlation "
          "means the geometry moved.")
    report = {'planck': compare(planck, os.path.join(a.outdir, 'Planck_Val_Input.npy'),
                                'Planck')}
    if act is not None:
        report['act'] = compare(act, os.path.join(a.outdir, 'ACT_Val_GroundTruth.npy'),
                                'ACT')

    # ---- write -------------------------------------------------------------
    prov = dict(SRCFILE=os.path.basename(hp_path), FREQ=(a.band, 'GHz'),
                UNITSCL=(scale, 'factor applied to the HEALPix values'),
                BUNIT='uK_CMB' if scale != 1.0 else 'MJy/sr',
                INTERP=('bilinear', 'healpy 4-point interpolation'),
                COORDSRC=('GALACTIC', 'input frame; rotated C->G per pixel'),
                ORIGIN='make_val_split.py', SPLITFRC=(a.split_frac, 'column split'),
                NANPOL=(a.nan_policy, 'off-footprint convention'))
    written = []

    def put(name, arr, extra):
        npy = os.path.join(a.outdir, name + tag + '.npy')
        np.save(npy, arr)
        written.append(npy)
        print(f"[out ] {os.path.basename(npy)}  ({os.path.getsize(npy)/1e6:.0f} MB)")
        if not a.no_fits:
            fts = os.path.join(a.outdir, name + tag + '.fits')
            write_fits(fts, arr, wcs, extra)
            written.append(fts)
            print(f"[out ] {os.path.basename(fts)}   (carries the WCS)")

    put('Planck_Val_Input', planck, prov)
    if act is not None:
        put('ACT_Val_GroundTruth', act,
            dict(ORIGIN='make_val_split.py', CONTENT='ACT DR6 coadd, validation split',
                 SPLITFRC=(a.split_frac, 'column split'),
                 NANPOL=(a.nan_policy, 'off-footprint convention')))

    manifest = dict(tag=tag, act_fits=os.path.abspath(a.act_fits),
                    healpix=os.path.abspath(hp_path), band=a.band,
                    unit_scale=scale, split_frac=a.split_frac,
                    shape=[int(H), int(W)], nan_policy=a.nan_policy,
                    written=[os.path.basename(f) for f in written],
                    comparison=report, when=time.strftime('%Y-%m-%d %H:%M:%S'))
    mpath = os.path.join(a.outdir, f'val_split{tag}.json')
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[out ] {os.path.basename(mpath)}")

    # ---- stale predictions -------------------------------------------------
    scan = a.recon_scan or os.path.abspath(
        os.path.join(a.outdir, os.pardir, os.pardir))
    stale = sorted(glob.glob(os.path.join(scan, '**', 'Recon_Planck_Val_Input*.npy'),
                             recursive=True))
    if stale:
        print(f"\n[stale] {len(stale)} prediction file(s) under {scan} were made "
              f"from the OLD input:")
        for s in stale[:12]:
            print("         " + os.path.relpath(s, scan))
        if len(stale) > 12:
            print(f"         ... and {len(stale)-12} more")
        print("        Each is now paired with an input it never saw. Re-run "
              "inference on")
        print(f"        Planck_Val_Input{tag}.npy before comparing anything, or "
              f"keep using the")
        print("        old input with them.")

    # ---- optionally take over the canonical names --------------------------
    if a.replace:
        print("\n[repl] taking over the canonical names, old files moved aside")
        for name, arr in (('Planck_Val_Input', planck),
                          ('ACT_Val_GroundTruth', act)):
            if arr is None:
                continue
            canon = os.path.join(a.outdir, name + '.npy')
            if os.path.isfile(canon):
                keep = os.path.join(a.outdir, f'{name}_pre{tag.lstrip("_")}.npy')
                if os.path.exists(keep):
                    sys.exit(f"{keep} already exists; move it before rerunning")
                shutil.move(canon, keep)
                print(f"       {name}.npy -> {os.path.basename(keep)}")
            shutil.copyfile(os.path.join(a.outdir, name + tag + '.npy'), canon)
            print(f"       {name}.npy is now the {tag.lstrip('_')} version")

    print("\n[done] the tagged files are the new reference. Point "
          "crop_region.py at the")
    print("       .fits with --planck/--act and it needs no --val-split rebuild: "
          "the WCS")
    print("       travels with the data now.")


if __name__ == '__main__':
    main()