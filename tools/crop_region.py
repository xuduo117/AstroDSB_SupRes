#!/usr/bin/env python3
"""
crop_region.py  --  cut one test region out of the ACT/Planck data, and
                    optionally re-derive the Planck side with a different
                    resampling so the interpolation kernel can be tested.
                    RUNS ON TRILLIUM (that is where the data is).

WHICH ARRAY ARE YOUR CROP INDICES ON?  This is the thing that goes wrong.

    crop_x, crop_y, img_size = 1700, 2900, 2048

was chosen in data_planck_ACT.ipynb on the VALIDATION arrays, the ones
extract_val_region.py writes:

    Planck_Val_Input.npy / ACT_Val_GroundTruth.npy      (10222 x 12961)

Those cover the right 30% of the ACT map only, cropped to its active bbox. The
Full_Footprint_Inference arrays cover the whole ACT footprint and are a
different size, so the SAME indices point at completely different sky. Feeding
the full-footprint arrays a crop corner picked on the validation arrays is the
usual cause of "the crop looks wrong".

So there are two modes.

  --val-split     Rebuilds the validation geometry from the raw ACT coadd FITS:
                  the 0.70 column split and the active bbox, exactly as
                  extract_val_region.py does. Pixel indices then mean what they
                  mean in the notebook. This also RECOVERS THE WCS, which the
                  stored .npy files do not carry, and the WCS is what makes the
                  kernel variants possible at all.

  (default)       Crop the stored co-registered arrays as they are. Use it for
                  the Full_Footprint_Inference products, with --ra/--dec rather
                  than pixel indices if you are moving between geometries.

PLANCK ALWAYS COMES FROM PETER'S FOLDER.  extract_val_region.py sampled
HFI_SkyMap_217_2048_R4.00_full.fits. The current standard is the cleaned maps in
Planck_Peter/, so in --val-split mode the Planck side is re-sampled from there
onto the recovered WCS rather than cropped out of the stored .npy. Pass the
DIRECTORY to --healpix and the band to --band; the file is found by glob and its
unit is read from TUNIT in the header, the same way act_planck_reproject.py does
it, so there is no 1e6 to get wrong by hand.

The stored .npy is still worth pointing at through --verify-dir. It cannot
validate the Planck values (different release, so they SHOULD differ) but it
validates the INDEXING, which is the part that was wrong: ACT must match bit for
bit, and Planck must correlate at r > 0.99. Anything less means the geometry
recovery is off and nothing downstream is trustworthy.

THE KERNEL VARIANTS (need a WCS, so --val-split or a .fits)

There are TWO interpolations in this pipeline and only one of them has an order.

  1. The PROJECTION, HEALPix sphere -> ACT grid. healpy offers exactly two
     kernels: hp.get_interp_val / get_interp_weights, which is 4-point
     BILINEAR, and hp.ang2pix, which is NEAREST. There is no bicubic on the
     sphere, so "projection at order 3" does not exist and cannot be tested.
  2. The GRID UPSAMPLE, coarse grid -> fine grid, done by scipy.ndimage.zoom in
     dsb_upsample_pipeline.py and by F.interpolate in datasets.py. Orders 0, 1
     and 3 all exist here. THIS is the step the training mix {"1": 1, "3": 2}
     refers to.

So the variants cover both questions:

  hpx_bilinear   the projection at order 1. Reproduces production; in
                 --val-split mode this is already the primary array, so asking
                 for it only gives you a duplicate.
  hpx_nearest    the projection at order 0. Together with the above this is the
                 whole order comparison the projection step supports.
  hpx_exact      no kernel at all: expand the map in spherical harmonics and
                 evaluate that expansion on the CAR grid, which is exact for a
                 band-limited field. The reference the other two are errors
                 against. Slow, and only meaningful for a beam-limited map.
  zoom1, zoom3   sample onto a grid coarser by --zoom-factor, then lift back
                 with scipy.ndimage.zoom at order 1 or 3. zoom1 is the CONTROL
                 for zoom3: they differ only in the order, so the pair isolates
                 it, which comparing either one against hpx_bilinear would not
                 (that would confound sphere-vs-grid sampling with the order).

MEASURED, on a scale model of Planck 217 on the ACT grid (beam 5', HEALPix
pixel 1.72', target 0.5'), each kernel against exact band-limited resampling:

    hpx bilinear (production)   0.63% of map rms
    hpx nearest                 2.29%
    coarse x5 -> zoom order 1   1.33%
    coarse x5 -> zoom order 3   0.69%

The projection kernel is therefore not a significant lever: bilinear is already
within a per cent of exact, and the error lives almost entirely ABOVE the beam
scale, where the real sky has no power and the model is generating rather than
sharpening. Note also that zoom order 3 is nearer to the production projection
(0.32% apart) than zoom order 1 is (0.83%), at every zoom factor tested. The
word "bilinear" appears in both, but a 4-point spherical interpolation from a
grid only 2.9 pixels per beam coarse is much closer to exact than a x5 grid
bilinear zoom. Weighting training toward order 3 is defensible on that basis.

TYPICAL USE

  # the notebook's region, Planck from Peter's folder, plus the kernel test
  python crop_region.py --val-split \
      --act-fits /scratch/xuduo117/Real_Obs_Data/Planck_Data/act-planck_dr6.02_coadd_AA_night_f220_map.fits \
      --healpix  /scratch/xuduo117/Real_Obs_Data/Planck_Data/Planck_Peter --band 217 \
      --crop-x 1700 --crop-y 2900 --size 2048 \
      --verify-dir /scratch/xuduo117/Project_DSB/AstroDSB_SupRes_3/act_inference_data/Validation_Split \
      --variants zoom1,zoom3,hpx_nearest --zoom-factor 3 \
      --out region_val_2048.npz

  # a region of the full-footprint arrays, addressed on the sky
  python crop_region.py \
      --planck .../Full_Footprint_Inference/Planck_FullSky_Input_217.fits \
      --act    .../Full_Footprint_Inference/ACT_FullSky_GroundTruth.fits \
      --ra 35.0 --dec -5.0 --size 2048 \
      --out region_a.npz
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

UNSEEN_LIMIT = -1e30        # HEALPix bad value is -1.6375e30

# Mirrors act_planck_reproject.py so the two scripts cannot disagree about
# units.  K_CMB -> uK_CMB; MJy/sr is left alone, because converting it to
# uK_CMB needs the RIMO bandpass coefficients and only means anything for a
# CMB-like SED.
UNIT_RULES = {'K_CMB': 1e6, 'KCMB': 1e6, 'K': 1e6, 'MJY/SR': 1.0}
BAND_UNIT = {100: 'K_CMB', 143: 'K_CMB', 217: 'K_CMB',
             353: 'K_CMB', 545: 'MJy/sr', 857: 'MJy/sr'}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', required=True, help="output .npz")

    g = p.add_argument_group("validation-split mode (indices as in the notebook)")
    g.add_argument('--val-split', action='store_true',
                   help="rebuild the extract_val_region geometry from --act-fits, "
                        "so --crop-x/--crop-y mean what they mean in "
                        "data_planck_ACT.ipynb, and a WCS is available")
    g.add_argument('--act-fits',
                   help="the raw ACT coadd FITS, e.g. "
                        "act-planck_dr6.02_coadd_AA_night_f220_map.fits")
    g.add_argument('--split-frac', type=float, default=0.70,
                   help="column split used by extract_val_region.py; validation "
                        "is everything to the right of int(width * this)")
    g.add_argument('--val-bbox',
                   help="y0,y1,x0,x1 of the active bbox, to skip the scan. The "
                        "script prints the right value on its first run.")
    g.add_argument('--verify-dir',
                   help="folder holding the stored ACT_Val_GroundTruth*.npy and "
                        "Planck_Val_Input*.npy, used to check the indexing")
    g.add_argument('--ignore-verify', action='store_true',
                   help="write the crop even if --verify-dir says the indexing "
                        "is wrong. For diagnosis only.")

    g = p.add_argument_group("stored-array mode")
    g.add_argument('--planck', help=".npy or .fits already on the ACT grid")
    g.add_argument('--act', help=".npy or .fits ground truth on the same grid")
    g.add_argument('--wcs-from', help="FITS to read the WCS from, if you passed "
                                      ".npy above but still want --ra/--dec or "
                                      "the variants")

    g = p.add_argument_group("where")
    g.add_argument('--crop-x', type=int, help="column index of the crop corner")
    g.add_argument('--crop-y', type=int, help="row index of the crop corner")
    g.add_argument('--ra', type=float, help="right ascension of the crop centre, degrees")
    g.add_argument('--dec', type=float, help="declination of the crop centre, degrees")
    g.add_argument('--size', type=int, default=2048, help="square crop side in pixels")

    g = p.add_argument_group("Planck source and kernel variants")
    g.add_argument('--healpix',
                   help="Peter's Planck directory, or a single HFI_SkyMap FITS. "
                        "Required in --val-split mode, since that mode re-samples "
                        "Planck rather than cropping the stored array.")
    g.add_argument('--band', type=int, default=217,
                   help="Planck band in GHz, used to pick the file from a directory")
    g.add_argument('--variants', default='',
                   help="comma list from hpx_bilinear, hpx_nearest, hpx_exact, "
                        "zoom1, zoom3. hpx_exact is the band-limited reference "
                        "and is slow; the others are cheap.")
    g.add_argument('--zoom-factor', type=int, default=3,
                   help="coarsening factor for the zoom variants. 3 puts the "
                        "coarse grid at 1.5 arcmin, closest to the 1.72 arcmin "
                        "nside-2048 pixel production actually starts from; 5 "
                        "matches the 51 -> 256 training geometry and is the "
                        "harsher test.")
    g.add_argument('--exact-iter', type=int, default=1,
                   help="map2alm iterations for hpx_exact. 0 is fast and "
                        "noticeably less accurate; 3 is healpy's default.")
    g.add_argument('--unit-scale', type=float, default=None,
                   help="override the multiplier applied to the HEALPix values. "
                        "Leave unset and it is read from TUNIT in the header.")
    return p.parse_args()


# ==========================================================================
# inputs
# ==========================================================================

def open_array(path):
    """Memory-map a .npy, or lazily open a .fits.  Returns (array, header)."""
    if path.lower().endswith(('.fits', '.fit', '.fz')):
        from astropy.io import fits
        hdul = fits.open(path, memmap=True)
        for h in hdul:
            if getattr(h, 'data', None) is not None and h.data.ndim >= 2:
                return h.data, h.header
        raise ValueError(f"no image HDU in {path}")
    return np.load(path, mmap_mode='r'), None


def get_wcs(header, path_hint=''):
    if header is None:
        return None
    from astropy.wcs import WCS
    try:
        return WCS(header).celestial
    except Exception as e:                                   # noqa: BLE001
        print(f"  warning: could not build a WCS from {path_hint}: {e}")
        return None


def resolve_healpix(spec, band):
    """--healpix may be Peter's directory or one file; a directory is globbed."""
    if not os.path.isdir(spec):
        return spec
    cand = sorted(glob.glob(os.path.join(spec, 'HFI_SkyMap_*.fits')))
    hit = [p for p in cand
           if re.search(rf'HFI_SkyMap_0*{band}[_-]', os.path.basename(p))]
    if not hit:
        have = ', '.join(os.path.basename(c) for c in cand) or 'nothing'
        sys.exit(f"no HFI_SkyMap_{band}_*.fits in {spec}\n  found: {have}")
    if len(hit) > 1:
        print(f"[hpx ] {len(hit)} files match band {band}, taking the first:")
        for p in hit:
            print("         " + os.path.basename(p))
    return hit[0]


def healpix_unit_scale(path, band, override):
    """The multiplier and why, read from the file the way act_planck_reproject does."""
    if override is not None:
        return float(override), "given with --unit-scale"
    from astropy.io import fits
    native = ''
    try:
        native = str(fits.getheader(path, 1).get('TUNIT1', '')).strip()
    except Exception:                                        # noqa: BLE001
        pass
    expected = BAND_UNIT.get(band)
    if not native:
        native = expected or 'K_CMB'
        why = f"no TUNIT in the header, assuming {native}"
    else:
        why = f"TUNIT1 = {native!r}"
        if expected and (native.upper().replace(' ', '')
                         != expected.upper().replace(' ', '')):
            why += f", but the band table expects {expected!r}; trusting the header"
    key = native.upper().replace(' ', '')
    if key not in UNIT_RULES:
        sys.exit(f"band {band} has unit {native!r}, which this script does not "
                 f"know how to convert. Add it to UNIT_RULES, or pass "
                 f"--unit-scale explicitly.")
    return UNIT_RULES[key], why


# ==========================================================================
# geometry
# ==========================================================================

def crop_wcs(wcs, x0, y0):
    """Shift the reference pixel so the sub-array keeps its astrometry."""
    w = wcs.deepcopy()
    w.wcs.crpix = [wcs.wcs.crpix[0] - x0, wcs.wcs.crpix[1] - y0]
    return w


def coarse_wcs(wcs, f):
    """WCS of a grid coarser by f, under the pixel-as-area convention.

    A coarse pixel J covers fine pixels [J*f, (J+1)*f), so its centre sits at
    fine index f*J + (f-1)/2.  Matching world coordinates at that point gives
    CRPIX_coarse = (CRPIX_fine + (f-1)/2) / f, with CDELT scaled by f.  This is
    the same convention scipy.ndimage.zoom(grid_mode=True) undoes.
    """
    w = wcs.deepcopy()
    w.wcs.crpix = [(wcs.wcs.crpix[0] + (f - 1) / 2.0) / f,
                   (wcs.wcs.crpix[1] + (f - 1) / 2.0) / f]
    w.wcs.cdelt = [wcs.wcs.cdelt[0] * f, wcs.wcs.cdelt[1] * f]
    return w


def val_split_patch(act_fits, frac, bbox_str):
    """Reproduce extract_val_region.py's geometry, returning the enmap patch.

    enmap carries the WCS through both slices, so the returned object knows
    where it is on the sky -- which is exactly what the stored .npy throws away.
    """
    from pixell import enmap
    shape, wcs = enmap.read_map_geometry(act_fits)
    h, w = shape[-2:]
    split_x = int(w * frac)
    print(f"[val ] ACT map {h} x {w}; split at x = {split_x}, "
          f"validation is x > {split_x}")

    region = enmap.read_map(act_fits,
                            sel=(Ellipsis, slice(None), slice(split_x, None)))
    if region.ndim > 2:
        region = region[0]

    if bbox_str:
        y0, y1, x0, x1 = (int(v) for v in bbox_str.split(','))
        print(f"[val ] active bbox given: y {y0}:{y1}  x {x0}:{x1}")
    else:
        # isfinite & != 0, the same test extract_val_region.py applies.  Reduced
        # per axis rather than np.where on the whole array, which would
        # materialise two index arrays the size of the data.
        m = np.isfinite(np.asarray(region)) & (np.asarray(region) != 0.0)
        rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if rows.size == 0:
            sys.exit("no valid data found in the validation split")
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        del m, rows, cols
        print(f"[val ] active bbox y {y0}:{y1}  x {x0}:{x1}")
        print(f"[val ] pass --val-bbox {y0},{y1},{x0},{x1} next time to skip the scan")

    patch = region[y0:y1, x0:x1]
    print(f"[val ] validation patch {patch.shape}  "
          f"(the stored Val .npy should be exactly this shape)")
    return patch


def sample_healpix(hp_map, nside, wcs, shape, mode, scale):
    """Sample a HEALPix map onto a WCS grid, equatorial grid to galactic map."""
    import healpy as hp
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = wcs.wcs_pix2world(xx.ravel().astype(np.float64),
                                yy.ravel().astype(np.float64), 0)
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    return _sample(hp_map, nside, theta, phi, mode, scale, (ny, nx))


def sample_healpix_posmap(hp_map, nside, patch, mode, scale):
    """As above but taking the angles from enmap.posmap().

    This is literally what extract_val_region.py does, so it removes any
    question about WCS conventions between pixell and astropy.
    """
    dec_rad, ra_rad = patch.posmap()
    return _sample(hp_map, nside, (np.pi / 2.0 - dec_rad).ravel(), ra_rad.ravel(),
                   mode, scale, patch.shape)


def sample_exact(hp_map, wcs, shape, scale, n_iter=1):
    """Band-limited resampling through the spherical harmonics: no kernel at all.

    healpy offers exactly two projection kernels, 4-point bilinear and nearest.
    There is no bicubic on the sphere, so "order 3 projection" cannot be tested
    directly.  What CAN be done is better: expand the map in spherical harmonics
    and evaluate that expansion on the CAR grid, which is exact for a
    band-limited field.  Every kernel is then measurable against truth rather
    than against another approximation.

    Only meaningful for a map whose power dies well before the band limit, which
    a beam-smoothed Planck map does and a white-noise map does not.  Costly (a
    full map2alm at the map's own resolution) and it needs the map to be
    complete, so UNSEEN pixels are zero-filled first and the ringing that causes
    stays near the mask edge.

    The HEALPix pixel window is deconvolved, without which this is not exact in
    exactly the sub-beam band the comparison is about.
    """
    import healpy as hp
    from pixell import curvedsky, enmap
    nside = hp.get_nside(hp_map)
    lmax = min(2 * nside, 4096)          # 2*nside is where map2alm stays reliable
    filled = np.nan_to_num(np.asarray(hp_map, dtype=np.float64), nan=0.0)
    print(f"       hpx_exact: map2alm at lmax {lmax}, iter {n_iter}; "
          f"this is the slow one, minutes at nside 2048")
    alm = hp.map2alm(filled, lmax=lmax, iter=n_iter)
    # The other kernels rotate each SAMPLE POSITION from equatorial to galactic.
    # This one evaluates the field on the equatorial CAR grid instead, so the
    # FIELD has to be rotated, once, in harmonic space.  Without this the patch
    # comes out of a completely different part of the sky and looks merely odd
    # rather than obviously broken.
    alm = hp.Rotator(coord=['G', 'C']).rotate_alm(alm)
    pw = hp.pixwin(nside, lmax=lmax)
    pw = np.where(pw < 1e-3, 1e-3, pw)   # never divide the noise floor up
    alm = hp.almxfl(alm, 1.0 / pw)
    out = curvedsky.alm2map(alm.astype(np.complex128),
                            enmap.zeros(tuple(shape), wcs, dtype=np.float64))
    return (np.asarray(out) * scale).astype(np.float32)


def _sample(hp_map, nside, theta, phi, mode, scale, shape):
    import healpy as hp
    theta, phi = hp.Rotator(coord=['C', 'G'])(theta, phi)
    if mode == 'nearest':
        vals = hp_map[hp.ang2pix(nside, theta, phi, nest=False)]
    else:                                             # bilinear, as production
        pix, wgt = hp.get_interp_weights(nside, theta, phi, nest=False)
        vals = np.einsum('ij,ij->j', hp_map[pix], wgt)
    return (vals * scale).reshape(shape).astype(np.float32)


def zoom_up(data, factor, order):
    """dsb_upsample_pipeline.upsample(), reproduced so the test is the real path."""
    f = int(factor)
    if order == 0:
        return np.repeat(np.repeat(data, f, axis=0), f, axis=1)
    from scipy.ndimage import zoom as _ndzoom
    bad = ~np.isfinite(data)
    filled = data
    if bad.any():
        fill = np.nanmedian(data) if np.isfinite(data).any() else 0.0
        filled = np.where(bad, fill, data)
    out = _ndzoom(filled, f, order=order, grid_mode=True, mode='nearest')
    if bad.any():
        out[np.repeat(np.repeat(bad, f, axis=0), f, axis=1)] = np.nan
    return out.astype(np.float32)


# ==========================================================================
# cross-checks
# ==========================================================================

def find_stored(d, *stems):
    """The stored Val arrays, preferring an exact stem then any dated variant."""
    for s in stems:
        exact = os.path.join(d, s + '.npy')
        if os.path.isfile(exact):
            return exact
        hit = sorted(glob.glob(os.path.join(d, s + '*.npy')))
        if hit:
            return hit[0]
    return None


def verify(vdir, patch_shape, sl, act_crop, planck_crop):
    """Confirm the crop indices land on the same sky as the stored arrays.

    ACT must agree bit for bit: same file, same slicing.  Planck cannot, because
    the stored array came from R4.00 and we now sample Peter's cleaned map, so
    the test there is CORRELATION, which is insensitive to the release but very
    sensitive to a misplaced crop.
    """
    print("[vrfy] checking the indexing against the stored validation arrays")
    print("       ACT must be identical; Planck only has to correlate, because "
          "the stored")
    print("       array came from R4.00 and this crop is sampled from Peter's "
          "cleaned map")
    ok = True
    for label, stems, mine, tol in (
            ('ACT', ('ACT_Val_GroundTruth',), act_crop, None),
            ('Planck', ('Planck_Val_Input',), planck_crop, 0.99)):
        path = find_stored(vdir, *stems)
        if path is None:
            print(f"       {label:<7} no stored array found in {vdir}")
            continue
        ref_full = np.load(path, mmap_mode='r')
        name = os.path.basename(path)
        if ref_full.shape != patch_shape:
            print(f"       {label:<7} {name}  SHAPE MISMATCH "
                  f"{ref_full.shape} vs rebuilt {patch_shape}")
            print(f"               the geometry recovery is wrong; check "
                  f"--split-frac and that --act-fits is the same coadd")
            ok = False
            continue
        ref = np.asarray(ref_full[sl], dtype=np.float32)
        if mine is None:
            print(f"       {label:<7} {name}  shape OK, nothing to compare")
            continue
        # the stored arrays have NaN replaced by 0.0; compare where both are real
        good = np.isfinite(ref) & np.isfinite(mine) & (ref != 0.0)
        if good.sum() < 100:
            print(f"       {label:<7} {name}  too few comparable pixels")
            continue
        a, b = ref[good].astype(np.float64), mine[good].astype(np.float64)
        r = float(np.corrcoef(a, b)[0, 1])
        rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))
        if tol is None:
            verdict = "identical" if rel < 1e-5 else f"DIFFERS, max {rel*100:.3g}%"
            ok &= rel < 1e-5
        else:
            verdict = "same sky" if r > tol else f"r = {r:.4f}, TOO LOW"
            ok &= r > tol
        print(f"       {label:<7} {name}  r = {r:.6f}  "
              f"max diff {rel*100:.3g}% of peak   {verdict}")
    if ok:
        print("[vrfy] indexing confirmed")
    else:
        print("[vrfy] INDEXING NOT CONFIRMED -- do not trust this crop")
    return ok


# ==========================================================================
# main
# ==========================================================================

def main():
    a = parse_args()
    variants = [v.strip() for v in a.variants.split(',') if v.strip()]
    S = a.size
    store, patch = {}, None

    # ---- geometry ----------------------------------------------------------
    if a.val_split:
        if not a.act_fits:
            sys.exit("--val-split needs --act-fits, the raw ACT coadd")
        if not a.healpix:
            sys.exit("--val-split re-samples Planck rather than cropping a "
                     "stored array, so it needs --healpix (Peter's folder)")
        vpatch = val_split_patch(a.act_fits, a.split_frac, a.val_bbox)
        full_shape = tuple(vpatch.shape)
        wcs = vpatch.wcs
        H, W = full_shape
    else:
        if not a.planck:
            sys.exit("give --planck, or use --val-split")
        planck, ph = open_array(a.planck)
        print(f"[data] planck {planck.shape} from {os.path.basename(a.planck)}")
        # act_planck_reproject.py stamps its provenance, so which Planck release
        # a stored array came from is a fact in the file rather than a guess.
        if ph is not None and ph.get('SRCFILE'):
            print(f"[data]   built from {ph.get('SRCFILE')}  "
                  f"[{ph.get('INTERP', '?')}, {ph.get('BUNIT', '?')}, "
                  f"scale {ph.get('UNITSCL', '?')}]")
            print("[data]   check that this is the map from Peter's folder; if it "
                  "is not, re-run act_planck_reproject.py against that folder")
        elif ph is not None:
            print("[data]   no SRCFILE header, so the Planck release behind this "
                  "array cannot be confirmed from the file")
        wcs = get_wcs(ph, a.planck)
        if wcs is None and a.wcs_from:
            from astropy.io import fits
            wcs = get_wcs(fits.getheader(a.wcs_from), a.wcs_from)
        act = None
        if a.act:
            act, aph = open_array(a.act)
            print(f"[data] act    {act.shape} from {os.path.basename(a.act)}")
            if act.shape[-2:] != planck.shape[-2:]:
                sys.exit(f"the two arrays are not on the same grid: "
                         f"{act.shape} vs {planck.shape}; they must come from "
                         f"the same reprojection run")
        H, W = planck.shape[-2:]
        full_shape = (H, W)

    # ---- resolve the corner -------------------------------------------------
    if a.ra is not None and a.dec is not None:
        if wcs is None:
            sys.exit("--ra/--dec needs a WCS; pass the .fits, --wcs-from, or --val-split")
        cx, cy = wcs.wcs_world2pix(a.ra, a.dec, 0)
        x0, y0 = int(round(float(cx) - S / 2)), int(round(float(cy) - S / 2))
        print(f"[crop] RA {a.ra} Dec {a.dec} -> corner ({x0}, {y0})")
    elif a.crop_x is not None and a.crop_y is not None:
        x0, y0 = a.crop_x, a.crop_y
        print(f"[crop] corner ({x0}, {y0}) as given")
    else:
        sys.exit("give either --crop-x/--crop-y or --ra/--dec")

    if not (0 <= x0 and x0 + S <= W and 0 <= y0 and y0 + S <= H):
        sys.exit(f"the crop [{y0}:{y0+S}, {x0}:{x0+S}] falls outside the "
                 f"{H} x {W} array")
    sl = (slice(y0, y0 + S), slice(x0, x0 + S))

    # ---- the crop -----------------------------------------------------------
    if a.val_split:
        patch = vpatch[sl]                  # enmap slicing carries the WCS along
        sub = patch.wcs
        store['act'] = np.asarray(patch, dtype=np.float32)
    else:
        sub = crop_wcs(wcs, x0, y0) if wcs is not None else None
        store['planck'] = np.asarray(
            planck[sl] if planck.ndim == 2 else planck[0][sl], dtype=np.float32)
        if act is not None:
            store['act'] = np.asarray(
                act[sl] if act.ndim == 2 else act[0][sl], dtype=np.float32)

    if 'act' in store:
        fin = np.isfinite(store['act']) & (store['act'] != 0.0)
        frac = float(fin.mean())
        print(f"[crop] ACT coverage in the crop: {frac*100:.1f}% live pixels")
        if frac < 0.999:
            # astrodsb_evaluate apodizes on the valid-pixel mask, and a zero is
            # not a missing value to it -- an off-footprint edge inside the crop
            # is read as real data and drags P(k) down at every scale.
            print(f"[crop] WARNING: {(1-frac)*100:.1f}% of this crop is off the "
                  f"ACT footprint (exactly 0.0).")
            print("       Those zeros are treated as real data by the P(k) "
                  "apodisation and will bias every")
            print("       spectrum in the analysis. Move the crop, or shrink "
                  "--size, until this reads 100%.")

    # ---- the Planck side ----------------------------------------------------
    hp_path = scale = None
    if a.healpix:
        import healpy as hp
        hp_path = resolve_healpix(a.healpix, a.band)
        scale, why = healpix_unit_scale(hp_path, a.band, a.unit_scale)
        print(f"[hpx ] {os.path.basename(hp_path)}")
        print(f"[hpx ] unit scale {scale:g}  ({why})")
        m = np.asarray(hp.read_map(hp_path, field=0, nest=False,
                                   dtype=np.float32), dtype=np.float32)
        bad = ~np.isfinite(m) | (m < UNSEEN_LIMIT)
        if bad.any():
            m[bad] = np.nan                      # never interpolate an UNSEEN
            print(f"[hpx ] {int(bad.sum())} UNSEEN pixels -> NaN")
        nside = hp.get_nside(m)
        print(f"[hpx ] nside {nside}")

        if a.val_split:
            # posmap is what extract_val_region.py used, so the primary array is
            # produced by exactly the production path.
            store['planck'] = sample_healpix_posmap(m, nside, patch,
                                                    'bilinear', scale)
            # the variants need an astropy WCS; confirm the two agree before
            # trusting anything derived from it
            viaw = sample_healpix(m, nside, sub, (S, S), 'bilinear', scale)
            d = np.nanmax(np.abs(viaw - store['planck']))
            rel = d / (np.nanstd(store['planck']) + 1e-30)
            print(f"[wcs ] posmap vs astropy WCS: max difference {d:.4g} "
                  f"({rel*100:.2g}% of the map rms)"
                  + ("" if rel < 1e-3 else "   <-- THEY DISAGREE, "
                                           "the variants are not trustworthy"))
    elif a.val_split:
        sys.exit("internal: --val-split without --healpix should already have exited")

    # ---- cross-check the indexing ------------------------------------------
    if a.verify_dir:
        if not verify(a.verify_dir, full_shape, sl,
                      store.get('act'), store.get('planck')):
            if not a.ignore_verify:
                sys.exit("\nRefusing to write a crop whose indexing did not "
                         "check out. A crop on the wrong sky produces figures\n"
                         "that look plausible and are wrong, which is worse "
                         "than no figures. Fix the geometry, or pass\n"
                         "--ignore-verify if you are deliberately cropping "
                         "something the stored arrays do not cover.")
            print("[vrfy] --ignore-verify given, writing anyway")

    p_crop = store['planck']
    fin_p = np.isfinite(p_crop)
    print(f"[crop] Planck {p_crop.shape}, {fin_p.mean()*100:.1f}% finite, "
          f"median {np.nanmedian(p_crop):.4g}")
    if sub is not None:
        ra_c, dec_c = sub.wcs_pix2world(S / 2, S / 2, 0)
        print(f"[crop] centre RA {float(ra_c):.4f} Dec {float(dec_c):.4f}")

    # ---- optional kernel variants ------------------------------------------
    if variants:
        if hp_path is None:
            sys.exit("--variants needs --healpix")
        if sub is None:
            sys.exit("--variants needs a WCS; pass the .fits, --wcs-from, or --val-split")
        if a.val_split and 'hpx_bilinear' in variants:
            print("[var ] note: in --val-split mode hpx_bilinear is already the "
                  "primary array, so this variant is a duplicate")
        for v in variants:
            if v == 'hpx_exact':
                try:
                    arr = sample_exact(m, sub, (S, S), scale, a.exact_iter)
                except Exception as e:                       # noqa: BLE001
                    print(f"[var ] hpx_exact failed ({e.__class__.__name__}: {e})")
                    print("       pixell could not run an SHT onto this "
                          "geometry; skipping. The other variants are unaffected.")
                    continue
            elif v in ('hpx_bilinear', 'hpx_nearest'):
                arr = sample_healpix(m, nside, sub, (S, S), v.split('_')[1], scale)
            elif v.startswith('zoom'):
                order = int(v[4:])
                f = a.zoom_factor
                # The coarse grid is rounded UP and the lift cropped back, so
                # --size need not be a multiple of --zoom-factor.  The coarse
                # WCS is derived from the fine one, so the extra row and column
                # sit outside the crop and the alignment is unaffected.
                nc = -(-S // f)
                cw = coarse_wcs(sub, f)
                coarse = sample_healpix(m, nside, cw, (nc, nc), 'bilinear', scale)
                arr = zoom_up(coarse, f, order)[:S, :S]
                print(f"       {v}: coarse grid {nc} x {nc} at "
                      f"{abs(cw.wcs.cdelt[1])*60:.2f} arcmin/px")
            else:
                sys.exit(f"unknown variant {v!r}")
            store[f'var_{v}'] = arr.astype(np.float32)
            d = arr - p_crop
            pct = float(np.nanstd(d) / (np.nanstd(p_crop) + 1e-12) * 100)
            print(f"[var ] {v:<13} median {np.nanmedian(arr):.4g}   "
                  f"rms difference from the primary array {pct:.2f}%")
            if v == 'hpx_exact' and pct > 25.0:
                # Planck 217 is sampled at ~2.9 pixels per beam, where bilinear
                # is within about 1% of exact.  A large number here does not
                # mean bilinear is bad, it means the SHT did not reproduce the
                # map: a non-galactic input, a masked region, or power above the
                # band limit.
                print(f"[var ] WARNING: hpx_exact is {pct:.0f}% from the "
                      f"bilinear array. That is far too large to be a kernel")
                print("       difference. Treat this variant as broken rather "
                      "than as evidence about the kernel:")
                print("       check the map is galactic and beam-limited, and "
                      "that the crop is clear of masked sky.")

    store['meta'] = json.dumps(dict(
        mode='val_split' if a.val_split else 'stored_arrays',
        act_fits=os.path.abspath(a.act_fits) if a.act_fits else None,
        split_frac=float(a.split_frac) if a.val_split else None,
        val_patch_shape=list(full_shape) if a.val_split else None,
        planck=os.path.abspath(a.planck) if a.planck else None,
        act=os.path.abspath(a.act) if a.act else None,
        healpix=os.path.abspath(hp_path) if hp_path else None,
        band=int(a.band), unit_scale=(float(scale) if scale is not None else None),
        crop_x=int(x0), crop_y=int(y0), size=int(S),
        variants=variants, zoom_factor=int(a.zoom_factor),
        wcs=(dict(sub.to_header()) if sub is not None else None),
    ), default=str)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    np.savez_compressed(a.out, **store)
    print(f"[done] {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print("       feed it to dsb_ensemble_infer.py with --region")


if __name__ == '__main__':
    main()