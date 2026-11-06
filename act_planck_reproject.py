"""
act_planck_reproject.py
=======================

Project every Planck HFI band onto the ACT grid, in one pass, writing FITS with a
correct WCS (plus .npy for the existing loaders).

This replaces `extract_aligned_full_sky` + `act_planck_geometry.py`: the geometry
is written while the crop is still in scope, so there is no separate header
recovery step and no way for the arrays and their WCS to drift apart.

    ACT_FullSky_GroundTruth.fits        ACT DR6 coadd, cropped to its active bbox
    Planck_FullSky_Input_100.fits       Planck 100 GHz on that grid
    Planck_FullSky_Input_143.fits       ... etc for 143, 217, 353, 545, 857
    (matching .npy beside each, unless --no-npy)

Every file shares one geometry, so they stack pixel-for-pixel, and every one has
NaN outside the ACT footprint -- `np.isfinite` is the coverage mask.

What changed relative to the original script, and why
-----------------------------------------------------

**Units are per-band.** The old `* 1e6` assumed K_CMB. That is right for 100-353
but 545 and 857 are distributed in MJy/sr, so multiplying them by a million
produced numbers 1e6 too large in a unit that was never temperature. This reads
TUNIT from each file, converts K_CMB -> uK_CMB, leaves MJy/sr alone, and records
what it did in BUNIT. A band whose header disagrees with the table below is
flagged loudly rather than silently coerced.

**UNSEEN is not a number.** HEALPix marks missing pixels with -1.6375e30. Bilinear
interpolation happily averages that with its neighbours, so a single bad pixel
bleeds a ~1e30 crater into the output. Those are converted to NaN before
interpolation, which propagates honestly instead.

**Interpolation weights are computed once for all six bands.** The expensive parts
per chunk are the C->G rotation and the HEALPix neighbour search, and both depend
only on the ACT grid, not on which Planck map is being sampled. Computing them
once and reusing them across bands is most of the ~6x saving over running the old
script six times. Only pixels inside the ACT footprint are sampled at all.

**Zeros became NaN.** The old pipeline ended with `act_final[~isfinite] = 0.0`, so
0.0 meant both "outside the survey" and "genuinely zero" and no statistic could
tell them apart. Off-footprint pixels are NaN here, identically in all bands.

**Memory is bounded by the chunk, not by the map.** Output goes straight into a
memory-mapped FITS data block, so peak RAM does not scale with the footprint. The
floor is the ACT map itself (~1.8 GB for a 10232 x 43200 float32, x3 if the file
is I/Q/U) plus ~200 MB per HEALPix band held in float32.

Finished bands are skipped on a re-run, so an interrupted job resumes, and
`--freqs` lets you do one band first.

Two geometries
--------------
`--geometry bbox` (default) crops to the ACT active bounding box, as before.

`--geometry fullsky` instead covers the whole sphere -- RA 0-360, Dec -90 to +90
-- at the ACT pixel scale, padding the ACT map with NaN (or `--act-pad zero`)
where the survey has no data. Planck is sampled everywhere, since Planck is
all-sky. The new grid is *aligned* to the ACT grid: its rows are a superset of
the ACT rows, so the ACT data is pasted in, never resampled or shifted.

Full sky at 0.5' is 21600 x 43200 = 933 M pixels, so **every output file is
3.7 GB**: seven files is 26 GB, doubled to 52 GB if .npy is written too. Check
the free space the script prints before letting it run; `--no-npy` halves it.

Time is dominated by that write, not by the arithmetic. Sampling the whole sphere
for six bands is a few minutes of CPU (measured ~3 M points/s for rotation +
neighbour search + a six-band gather); reading the ACT map and writing 52 GB is
what you will actually wait for.

Both geometries write the same filenames, so use separate `--outdir`s. The
geometry is recorded as GEOM in every header and a mismatch in an existing output
directory is refused rather than silently mixed.

One caveat worth knowing about
------------------------------
If a Planck band has missing input pixels, its NaN region is a *superset* of the
ACT footprint (bbox) or is non-empty at all (fullsky). That is honest -- there is
genuinely no data there -- but it means you cannot assume all bands share one
mask. The count goes in NBADPIX/FBADPIX and is printed at the end of the run. For
the standard full-mission HFI intensity maps it should come out zero.

Usage
-----
    python act_planck_reproject.py \
        --act    /scratch/xuduo117/Real_Obs_Data/Planck_Data/act-planck_dr6.02_coadd_AA_night_f220_map.fits \
        --planck /scratch/xuduo117/Real_Obs_Data/Planck_Data/Planck_Peter \
        --outdir ./act_inference_data/Full_Footprint_Inference/

    # whole sphere, ACT padded with NaN outside its footprint
    python act_planck_reproject.py --act ... --planck ... \
        --outdir ./act_inference_data/All_Sky/ --geometry fullsky

    # one band, smaller chunks, then check the result
    python act_planck_reproject.py --act ... --planck ... --outdir ... \
        --freqs 217 --chunk 1024 --verify 2000

Reading the output back with act_planck_cutout.py: point its FILES dict at the
band you want, e.g.

    FILES = {'planck': 'Planck_FullSky_Input_217.fits',
             'act':    'ACT_FullSky_GroundTruth.fits',
             'dsb':    'Recon_Planck_FullSky_Input.fits'}

Beam values
-----------
From the Planck PLA effective-beams table (2013 / R1.xx, FEBeCoP, BS Mars12).
`mean(fwhm)` is the Gaussian fit to the effective beam -- the one to use for
source identification and for beam matching. `FWHM_eff` is derived from the beam
solid angle and is the one to use for flux. Both are recorded in the header,
since which one is "the" beam depends on what you are doing with it.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time

import numpy as np
from astropy.io import fits


# --------------------------------------------------------------------------
# Band table.  fwhm_fit / fwhm_eff / omega from the PLA effective-beams page;
# `unit` is what the R1.xx maps are distributed in and is cross-checked against
# each file's TUNIT at run time.
# --------------------------------------------------------------------------
PLANCK_BANDS = {
    100: dict(fwhm_fit=9.651, fwhm_eff=9.66, omega=105.778, unit='K_CMB'),
    143: dict(fwhm_fit=7.248, fwhm_eff=7.27, omega=59.954,  unit='K_CMB'),
    217: dict(fwhm_fit=4.990, fwhm_eff=5.01, omega=28.447,  unit='K_CMB'),
    353: dict(fwhm_fit=4.818, fwhm_eff=4.86, omega=26.714,  unit='K_CMB'),
    545: dict(fwhm_fit=4.682, fwhm_eff=4.84, omega=26.535,  unit='MJy/sr'),
    857: dict(fwhm_fit=4.325, fwhm_eff=4.63, omega=24.244,  unit='MJy/sr'),
}

# K_CMB -> uK_CMB; MJy/sr is left alone.  Converting MJy/sr to uK_CMB needs the
# bandpass coefficients from the RIMO and is only meaningful for a CMB-like SED,
# so it is deliberately not done here.
UNIT_RULES = {
    'K_CMB':  (1e6, 'uK_CMB'),
    'KCMB':   (1e6, 'uK_CMB'),
    'K':      (1e6, 'uK_CMB'),
    'MJY/SR': (1.0, 'MJy/sr'),
}

ACT_BUNIT = 'uK_CMB'
ACT_BEAM_ARCMIN = {'f090': 2.0, 'f150': 1.4, 'f220': 1.0}   # nominal DR6 values
UNSEEN_LIMIT = -1e30            # HEALPix bad value is -1.63750e30


# ==========================================================================
# geometry
# ==========================================================================

def active_bbox(arr):
    """(y0, y1, x0, x1) of the finite, non-zero pixels; upper bounds exclusive.

    Same rule as the original pipeline, but via `any` along each axis rather than
    `np.where` over the whole map: `np.where` on a half-covered 10k x 43k array
    materialises two ~1.8 GB index arrays for a result that is four integers.
    """
    active = np.isfinite(arr) & (np.asarray(arr) != 0.0)
    rows = np.flatnonzero(active.any(axis=1))
    cols = np.flatnonzero(active.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise RuntimeError('No valid data found in the input ACT map!')
    return (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)


def read_act(act_path, verbose=True):
    """Load the ACT map, crop to its active footprint, return (patch, mask, bbox)."""
    from pixell import enmap

    if verbose:
        print(f'[act] reading {act_path}')
    act = enmap.read_map(act_path)
    if act.ndim > 2:
        act = act[0]                        # Stokes I, as in the original
    if verbose:
        print(f'[act] full map {act.shape[0]} x {act.shape[1]}')

    y0, y1, x0, x1 = active_bbox(act)
    patch = act[y0:y1, x0:x1]               # enmap slicing carries the WCS across
    mask = np.isfinite(patch) & (np.asarray(patch) != 0.0)
    if verbose:
        print(f'[act] active bbox  y {y0}:{y1}  x {x0}:{x1}')
        print(f'[act] cropped to   {patch.shape[0]} x {patch.shape[1]}   '
              f'({mask.mean()*100:.1f}% inside the footprint)')
    return patch, np.asarray(mask), (y0, y1, x0, x1)


def base_header(wcs, shape, bbox, geom='bbox', offset=(0, 0)):
    """Shared WCS header. Every output file starts from this."""
    hdr = wcs.to_header()
    y0, y1, x0, x1 = bbox
    hdr['GEOM'] = (geom, 'bbox = ACT active box; fullsky = whole sphere')
    hdr['CROP_Y0'] = (y0, 'active-bbox start row in the original ACT map')
    hdr['CROP_Y1'] = (y1, 'active-bbox end row (exclusive)')
    hdr['CROP_X0'] = (x0, 'active-bbox start column')
    hdr['CROP_X1'] = (x1, 'active-bbox end column (exclusive)')
    hdr['ACT_DY'] = (offset[0], 'row of this grid holding ACT bbox row 0')
    hdr['ACT_DX'] = (offset[1], 'column of this grid holding ACT bbox column 0')
    hdr['ORIGIN'] = 'act_planck_reproject.py'
    return hdr


def fullsky_geometry(wcs, shape):
    """Whole-sphere CAR grid at the same pixel scale, aligned to the input grid.

    Returns (wcs, (ny, nx), (dy, dx)): the input array's pixel (0, 0) lands at
    (dy, dx) in the new grid. Alignment is exact -- the new rows are a superset
    of the old ones -- so the ACT data is pasted, never resampled or shifted by
    a fraction of a pixel. Rows are kept while |Dec| <= 90 and the column count
    is whatever covers 360 degrees, both at the input CDELT.
    """
    if not str(wcs.wcs.ctype[0]).endswith('CAR') or float(wcs.wcs.crval[1]) != 0.0:
        raise ValueError('fullsky mode needs a CAR grid with CRVAL2 = 0; this one '
                         f'is {wcs.wcs.ctype[0]} with CRVAL2 = {wcs.wcs.crval[1]}')
    cd1, cd2 = float(wcs.wcs.cdelt[0]), float(wcs.wcs.cdelt[1])
    p1, p2 = float(wcs.wcs.crpix[0]), float(wcs.wcs.crpix[1])
    nx = int(shape[1])

    # Dec(y) = (y + 1 - CRPIX2) * CDELT2, y being a 0-based row index
    y_lo = int(np.ceil(p2 - 1 - 90.0 / cd2))
    y_hi = int(np.floor(p2 - 1 + 90.0 / cd2))
    nx_full = int(round(360.0 / abs(cd1)))
    if nx > nx_full:
        raise ValueError(f'the input is {nx} columns wide but 360 degrees is only '
                         f'{nx_full} at CDELT1 = {cd1}')

    w = wcs.deepcopy()
    w.wcs.crpix = [p1, p2 - y_lo]
    w._naxis = [nx_full, y_hi - y_lo + 1]
    return w, (y_hi - y_lo + 1, nx_full), (-y_lo, 0)


def grid_vectors(wcs, ny, nx):
    """(ra[nx], dec[ny], separable) in degrees.

    In CAR with CRVAL2 = 0 a pixel's RA depends only on its column and its Dec
    only on its row, so the whole grid is two 1-D vectors rather than two
    (ny, nx) arrays -- worth a lot at 933 M pixels. That is checked against
    astropy on a scatter of pixels rather than assumed; if it does not hold,
    `separable` is False and the caller falls back to a 2-D transform.
    """
    ra = np.asarray(wcs.wcs_pix2world(np.arange(nx, dtype=float),
                                      np.zeros(nx), 0)[0])
    dec = np.asarray(wcs.wcs_pix2world(np.zeros(ny),
                                       np.arange(ny, dtype=float), 0)[1])
    rng = np.random.default_rng(0)
    xs, ys = rng.integers(0, nx, 512), rng.integers(0, ny, 512)
    ra_c, dec_c = wcs.wcs_pix2world(xs.astype(float), ys.astype(float), 0)
    sep = bool(np.allclose((ra_c - ra[xs] + 180.0) % 360.0 - 180.0, 0.0, atol=1e-7)
               and np.allclose(dec_c, dec[ys], atol=1e-9))
    return ra, dec, sep


def chunk_angles(wcs, ra, dec, sep, y_lo, y_hi, x_lo, x_hi):
    """(theta, phi) in radians for every pixel of a chunk, row-major."""
    h, w = y_hi - y_lo, x_hi - x_lo
    if sep:
        th = np.repeat(np.deg2rad(90.0 - dec[y_lo:y_hi]), w)
        ph = np.tile(np.deg2rad(ra[x_lo:x_hi]), h)
    else:                                   # rotated or non-CAR: do it properly
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        r, d = wcs.wcs_pix2world(xx.astype(float).ravel(),
                                 yy.astype(float).ravel(), 0)
        th, ph = np.deg2rad(90.0 - d), np.deg2rad(r)
    return th, ph


# ==========================================================================
# large-file FITS
# ==========================================================================

def create_blank_fits(path, shape, header, dtype=np.float32):
    """Write header + an unwritten data block, so the file can then be memmapped.

    The astropy recipe for arrays too large to hold twice: lay down a correctly
    sized file, reopen it in update mode, and fill it through the memmap.
    """
    hdu = fits.PrimaryHDU(data=np.zeros((1, 1), dtype=dtype), header=header)
    hdu.header['NAXIS1'] = int(shape[1])
    hdu.header['NAXIS2'] = int(shape[0])
    head = hdu.header.tostring().encode('ascii')        # already 2880-padded
    nbytes = int(shape[0]) * int(shape[1]) * np.dtype(dtype).itemsize
    pad = (-nbytes) % 2880
    with open(path, 'wb') as f:
        f.write(head)
        f.truncate(len(head) + nbytes + pad)
    return path


def open_fits_memmap(path):
    """(hdulist, array) for an existing file, opened for in-place update."""
    hdul = fits.open(path, mode='update', memmap=True)
    return hdul, hdul[0].data


def fill_value(arr, value=np.nan, chunk=4096):
    """arr[:] = value, a band of rows at a time (arr may be a multi-GB memmap)."""
    for i in range(0, arr.shape[0], chunk):
        arr[i:i + chunk] = value


def check_geom(outdir, geom):
    """Refuse to mix bbox and fullsky files: the filenames are identical."""
    for p in glob.glob(os.path.join(outdir, '*FullSky*.fits')):
        try:
            found = str(fits.getheader(p).get('GEOM', 'bbox')).strip()
        except Exception:
            continue
        if found != geom:
            raise SystemExit(
                f'{outdir} already holds {found!r}-geometry output '
                f'({os.path.basename(p)}) but you asked for {geom!r}. The two use '
                f'the same filenames and are not interchangeable — write to a '
                f'different --outdir.')


def dump_npy(fits_path, npy_path, chunk=4096, verbose=True):
    """Stream a written FITS image out to .npy without loading it whole."""
    with fits.open(fits_path, memmap=True) as h:
        src = h[0].data
        out = np.lib.format.open_memmap(npy_path, mode='w+', dtype=np.float32,
                                        shape=src.shape)
        for i in range(0, src.shape[0], chunk):
            out[i:i + chunk] = src[i:i + chunk]
        out.flush()
        del out
    if verbose:
        print(f'       + {os.path.basename(npy_path)}')


# ==========================================================================
# Planck side
# ==========================================================================

def band_filename(freq, tag=''):
    """Output name for a band. `tag` separates two runs of the same frequency."""
    return f'Planck_FullSky_Input_{freq}{"_" + tag if tag else ""}.fits'


def check_provenance(path, src_name, what):
    """Refuse to skip a file that was built from a different input.

    Without this, pointing --planck at a different release of the same frequency
    finds an output of the same name, prints 'exists - skipped', and leaves you
    believing you have data you never made.
    """
    if not os.path.exists(path):
        return
    try:
        got = str(fits.getheader(path).get('SRCFILE', '')).strip()
    except Exception:
        return
    if got and got != src_name:
        raise SystemExit(
            f'\n{os.path.basename(path)} already exists, but it was built from\n'
            f'    {got}\n'
            f'and you are now passing\n'
            f'    {src_name}\n'
            f'Skipping would leave you with {what} from the wrong input. Use '
            f'--tag NAME to write a separate set, --overwrite to replace it, or '
            f'a different --outdir.')


def find_planck_files(spec):
    """{freq: path} from a directory, a glob, or an explicit list of files."""
    if isinstance(spec, str):
        spec = [spec]
    paths = []
    for s in spec:
        if os.path.isdir(s):
            paths += glob.glob(os.path.join(s, 'HFI_SkyMap_*.fits'))
        else:
            paths += glob.glob(s)
    found = {}
    for p in sorted(paths):
        m = re.search(r'HFI_SkyMap_(\d{3})_', os.path.basename(p))
        if not m:
            print(f'  ? cannot parse a frequency from {os.path.basename(p)} — skipped')
            continue
        f = int(m.group(1))
        if f not in PLANCK_BANDS:
            print(f'  ? {f} GHz is not an HFI band — skipped')
            continue
        if f in found:
            raise ValueError(f'two files for {f} GHz:\n  {found[f]}\n  {p}')
        found[f] = p
    return found


def unit_conversion(path, freq, field=0):
    """(scale, out_unit, native_unit) read from the file, cross-checked vs the table."""
    try:
        native = str(fits.getheader(path, 1).get(f'TUNIT{field + 1}', '')).strip()
    except Exception:
        native = ''
    expected = PLANCK_BANDS[freq]['unit']
    if not native:
        print(f'  ! {freq} GHz: no TUNIT in the header, assuming {expected}')
        native = expected
    elif native.upper().replace(' ', '') != expected.upper().replace(' ', ''):
        print(f'  ! {freq} GHz: header says {native!r}, the table expects '
              f'{expected!r} — trusting the header')

    key = native.upper().replace(' ', '')
    if key not in UNIT_RULES:
        raise ValueError(
            f'{freq} GHz has unit {native!r}, which this script does not know how '
            f'to convert. Add it to UNIT_RULES if that is right.')
    scale, out_unit = UNIT_RULES[key]
    return scale, out_unit, native


def read_healpix(path, field=0, verbose=True):
    """Read one HEALPix map as float32 in RING order, with UNSEEN turned into NaN."""
    import healpy as hp

    # nest=False makes healpy reorder a NESTED file to RING, which the Planck
    # maps are; `verbose` is deprecated since healpy 1.15, so it is not passed.
    m = np.asarray(hp.read_map(path, field=field, nest=False, dtype=np.float32),
                   dtype=np.float32)
    bad = ~np.isfinite(m) | (m < UNSEEN_LIMIT)
    nbad = int(bad.sum())
    if nbad:
        m[bad] = np.nan
        if verbose:
            print(f'       {nbad} UNSEEN/bad pixels -> NaN '
                  f'({nbad / m.size * 100:.3g}% of the sphere)')
    return m


# ==========================================================================
# the pass
# ==========================================================================

def reproject(act_path, planck_spec, outdir, freqs=None, chunk=2048,
              write_npy=True, overwrite=False, act_band=None, geometry='bbox',
              act_pad='nan', act_only=False, tag='', verbose=True):
    """One pass over the output grid, sampling every requested Planck band."""
    import healpy as hp

    if geometry not in ('bbox', 'fullsky'):
        raise ValueError("geometry must be 'bbox' or 'fullsky'")
    os.makedirs(outdir, exist_ok=True)
    check_geom(outdir, geometry)

    files = find_planck_files(planck_spec)
    if freqs:
        missing = sorted(set(freqs) - set(files))
        if missing:
            raise FileNotFoundError(f'no file found for {missing} in {planck_spec}')
        files = {f: files[f] for f in sorted(freqs)}
    if not files:
        raise FileNotFoundError(f'no HFI_SkyMap_*.fits found in {planck_spec}')

    print(f'\nPlanck bands: {", ".join(f"{f}" for f in sorted(files))}')

    # --- output geometry ----------------------------------------------------
    patch, mask, bbox = read_act(act_path, verbose=verbose)
    ny_act, nx_act = patch.shape
    if geometry == 'fullsky':
        wcs_out, (h, w), (dy, dx) = fullsky_geometry(patch.wcs, patch.shape)
        d0 = float(wcs_out.wcs_pix2world(0, 0, 0)[1])
        d1 = float(wcs_out.wcs_pix2world(0, h - 1, 0)[1])
        print(f'[geo] fullsky {h} x {w}   Dec {d0:+.4f} to {d1:+.4f}   '
              f'ACT block pasted at row {dy}, column {dx}')
    else:
        wcs_out, (h, w), (dy, dx) = patch.wcs, patch.shape, (0, 0)
        print(f'[geo] bbox {h} x {w}')
    hdr0 = base_header(wcs_out, (h, w), bbox, geometry, (dy, dx))

    nfiles = len(files) + 1
    gb = h * w * 4 / 1e9
    print(f'[geo] {gb:.2f} GB per file, {nfiles} files -> '
          f'{gb * nfiles * (2 if write_npy else 1):.1f} GB total'
          f'{" (FITS + npy)" if write_npy else " (FITS only)"}')

    if act_band is None:
        m = re.search(r'_(f\d{3})_', os.path.basename(act_path))
        act_band = m.group(1) if m else 'f220'
    pad = np.nan if act_pad == 'nan' else 0.0

    # --- ACT ground truth ---------------------------------------------------
    act_fits = os.path.join(outdir, 'ACT_FullSky_GroundTruth.fits')
    check_provenance(act_fits, os.path.basename(act_path), 'an ACT truth')
    if overwrite or not os.path.exists(act_fits):
        ah = hdr0.copy()
        ah['BUNIT'] = ACT_BUNIT
        ah['CONTENT'] = f'ACT DR6 {act_band} coadd (ground truth)'
        ah['BAND'] = (act_band, 'ACT frequency band')
        ah['BEAM'] = (ACT_BEAM_ARCMIN.get(act_band, 1.0),
                      'arcmin FWHM, nominal DR6 value')
        ah['ACTPAD'] = (act_pad, 'value where ACT has no coverage')
        ah['FCOVER'] = (round(mask.sum() / float(h * w), 6),
                        'fraction of this grid with ACT data')
        # no inline comment: a long filename plus a comment overruns the 80-column
        # card and astropy drops the comment. astropy will use CONTINUE if needed.
        ah['SRCFILE'] = os.path.basename(act_path)
        ah['COMMENT'] = f'{act_pad} = no ACT coverage'
        create_blank_fits(act_fits, (h, w), ah)
        hdul, arr = open_fits_memmap(act_fits)
        if geometry == 'fullsky':
            fill_value(arr, pad)                # everything outside the ACT block
        for i in range(0, ny_act, chunk):
            block = np.asarray(patch[i:i + chunk], dtype=np.float32)
            block[~mask[i:i + chunk]] = pad
            arr[dy + i:dy + i + block.shape[0], dx:dx + nx_act] = block
        hdul.close()
        print(f'[act] wrote {os.path.basename(act_fits)}  '
              f'({mask.sum()/float(h*w)*100:.1f}% of the grid has ACT data)')
        if write_npy:
            dump_npy(act_fits, os.path.join(outdir, 'ACT_FullSky_GroundTruth.npy'))
    else:
        print(f'[act] {os.path.basename(act_fits)} exists — skipped (use --overwrite)')

    # --- which bands still need doing ---------------------------------------
    todo = {}
    if act_only:
        print('\n--act-only: Planck bands left alone.')
        return {}
    for f, p in files.items():
        out = os.path.join(outdir, band_filename(f, tag))
        check_provenance(out, os.path.basename(p), f'a {f} GHz map')
        if os.path.exists(out) and not overwrite:
            print(f'[{f:3d}] {os.path.basename(out)} exists, same input — '
                  f'skipped (use --overwrite)')
            continue
        todo[f] = (p, out)
    if not todo:
        print('\nnothing left to do.')
        return {}

    # --- load the spheres ---------------------------------------------------
    print(f'\nloading {len(todo)} HEALPix map(s)...')
    maps, scales, units, natives = {}, {}, {}, {}
    for f, (src, _) in todo.items():
        scales[f], units[f], natives[f] = unit_conversion(src, f)
        print(f'  {f:3d} GHz  {os.path.basename(src)}')
        print(f'       {natives[f]} x {scales[f]:g} -> {units[f]}')
        maps[f] = read_healpix(src)
    nside = hp.npix2nside(maps[next(iter(maps))].size)
    for f, m in maps.items():
        if hp.npix2nside(m.size) != nside:
            raise ValueError(f'{f} GHz is nside {hp.npix2nside(m.size)}, '
                             f'others are {nside}')
    print(f'  nside {nside}')

    # --- output files -------------------------------------------------------
    hduls, arrs = {}, {}
    for f, (src, out) in todo.items():
        b = PLANCK_BANDS[f]
        ph = hdr0.copy()
        ph['BUNIT'] = units[f]
        ph['CONTENT'] = f'Planck {f} GHz on the ACT grid (input)'
        ph['FREQ'] = (f, 'GHz')
        ph['BEAM'] = (b['fwhm_fit'], 'arcmin, Gaussian fit to the effective beam')
        ph['BEAMEFF'] = (b['fwhm_eff'], 'arcmin, FWHM from the beam solid angle')
        ph['BEAMOMEG'] = (b['omega'], 'arcmin^2, effective beam solid angle')
        ph['SRCFILE'] = os.path.basename(src)          # see the note above
        ph['SRCUNIT'] = (natives[f], 'unit of the input map')
        ph['UNITSCL'] = (scales[f], 'factor applied to reach BUNIT')
        ph['NSIDE'] = (nside, 'HEALPix nside of the input')
        ph['INTERP'] = ('bilinear', 'healpy 4-point interpolation')
        ph['COORDSRC'] = ('GALACTIC', 'input frame; rotated C->G per pixel')
        # reserved now, filled before close: adding cards later would reflow the
        # header block and force astropy to rewrite the whole (multi-GB) file
        ph['NBADPIX'] = (0, 'sampled pixels NaN from missing input data')
        ph['FBADPIX'] = (0.0, 'NBADPIX as a fraction of the sampled area')
        ph['COMMENT'] = ('NaN = missing input (NBADPIX)' if geometry == 'fullsky'
                         else 'NaN = outside the ACT footprint, or missing input')
        ph['COMMENT'] = 'beams from the Planck PLA effective-beams table'
        create_blank_fits(out, (h, w), ph)
        hduls[f], arrs[f] = open_fits_memmap(out)
        fill_value(arrs[f], np.nan)

    # --- the chunk loop -----------------------------------------------------
    rot = hp.Rotator(coord=['C', 'G'])
    ra_vec, dec_vec, sep = grid_vectors(wcs_out, h, w)
    if not sep:
        print('  note: this grid is not separable in RA/Dec; falling back to a '
              'full 2-D coordinate transform per chunk (slower)')
    ny_c, nx_c = (h + chunk - 1) // chunk, (w + chunk - 1) // chunk
    print(f'\nsampling {ny_c} x {nx_c} chunks of {chunk}^2 for {len(todo)} '
          f'band(s){"" if geometry == "fullsky" else ", ACT footprint only"}...')

    t0 = time.time()
    done = skipped = 0
    nbad = {f: 0 for f in todo}          # sampled pixels with no input data
    ncov = h * w if geometry == 'fullsky' else int(mask.sum())
    for iy, y_lo in enumerate(range(0, h, chunk)):
        y_hi = min(y_lo + chunk, h)
        for x_lo in range(0, w, chunk):
            x_hi = min(x_lo + chunk, w)
            shp = (y_hi - y_lo, x_hi - x_lo)

            if geometry == 'fullsky':
                keep = None                        # sample the whole sphere
            else:
                keep = np.flatnonzero(mask[y_lo:y_hi, x_lo:x_hi].ravel())
                if keep.size == 0:                 # nothing to sample here
                    skipped += 1
                    continue

            theta, phi = chunk_angles(wcs_out, ra_vec, dec_vec, sep,
                                      y_lo, y_hi, x_lo, x_hi)
            if keep is not None:
                theta, phi = theta[keep], phi[keep]

            # equatorial -> galactic, then one neighbour search for all bands
            theta, phi = rot(theta, phi)
            pix, wgt = hp.get_interp_weights(nside, theta, phi, nest=False)

            for f, m in maps.items():
                vals = np.einsum('ij,ij->j', m[pix], wgt) * scales[f]
                nbad[f] += int(np.count_nonzero(~np.isfinite(vals)))
                if keep is None:
                    arrs[f][y_lo:y_hi, x_lo:x_hi] = vals.reshape(shp)
                else:
                    buf = np.full(shp[0] * shp[1], np.nan, dtype=np.float32)
                    buf[keep] = vals
                    arrs[f][y_lo:y_hi, x_lo:x_hi] = buf.reshape(shp)
            done += 1

        el = time.time() - t0
        frac = (iy + 1) / ny_c
        print(f'  row {iy+1:3d}/{ny_c}  {done:5d} sampled, {skipped:5d} empty  '
              f'{el/60:6.1f} min elapsed, ~{el*(1-frac)/frac/60:5.1f} min left',
              flush=True)

    for f in list(hduls):
        # same card count as reserved, so the header block does not reflow
        hduls[f][0].header['NBADPIX'] = nbad[f]
        hduls[f][0].header['FBADPIX'] = round(nbad[f] / max(ncov, 1), 8)
        hduls[f].close()
        if nbad[f]:
            print(f'  ! {f} GHz: {nbad[f]} pixels ({nbad[f]/ncov*100:.4g}% of the '
                  f'sampled area) are NaN because the input HEALPix map has no '
                  f'data there. Recorded as NBADPIX; this band therefore does not '
                  f'share a NaN pattern with the others.')
    print(f'\nsampled {done} chunks, skipped {skipped} empty ones, '
          f'in {(time.time()-t0)/60:.1f} min')

    written = {}
    for f, (src, out) in todo.items():
        written[f] = out
        print(f'[{f:3d}] wrote {os.path.basename(out)}  ({units[f]})')
        if write_npy:
            dump_npy(out, out[:-5] + '.npy')     # not .replace: a tag may repeat
    return written


# ==========================================================================
# verification
# ==========================================================================

def verify(outdir, planck_spec, n=2000, seed=0, freqs=None, tag='', verbose=True):
    """Re-sample random covered pixels by an independent route and compare.

    The pipeline gets its sky coordinates from pixell (`posmap`) and its values
    from precomputed interpolation weights. This goes the other way: astropy WCS
    on the header that was actually written, then `hp.get_interp_val`. Agreement
    means the data, the header and the coordinate rotation are mutually
    consistent, which is the thing that silently breaks.
    """
    import healpy as hp
    from astropy.wcs import WCS

    # Pair each *input* with its output. Globbing the output directory instead
    # picks up bands from earlier runs that this --planck does not cover.
    files = find_planck_files(planck_spec)
    if freqs:
        files = {f: p for f, p in files.items() if f in freqs}
    pairs, outs = [], []
    for f, src in sorted(files.items()):
        out = os.path.join(outdir, band_filename(f, tag))
        if os.path.exists(out):
            pairs.append((f, src, out))
            outs.append(out)
        else:
            print(f'  -- {f} GHz: {os.path.basename(out)} not found, skipped')
    if not pairs:
        print('nothing to verify')
        return False

    rng = np.random.default_rng(seed)
    ok_all = True
    # Sample where a Planck output is finite, not where ACT is: in fullsky
    # geometry most of the grid has no ACT data but every Planck pixel is real,
    # and it is the Planck sampling that is being checked.
    with fits.open(outs[0], memmap=True) as ph:
        ref, (h, w) = ph[0].data, ph[0].data.shape
        ys, xs, tries = [], [], 0
        while len(ys) < n and tries < 50:
            yy, xx = rng.integers(0, h, 4 * n), rng.integers(0, w, 4 * n)
            good = np.isfinite(ref[yy, xx])
            ys += list(yy[good][:n - len(ys)])
            xs += list(xx[good][:n - len(xs)])
            tries += 1
    if not ys:
        print('no finite pixels found to verify')
        return False
    ys, xs = np.array(ys[:n]), np.array(xs[:n])
    n = len(ys)

    rot = hp.Rotator(coord=['C', 'G'])
    for freq, src, out in pairs:
        with fits.open(out, memmap=True) as h_:
            hdr = h_[0].header
            got = np.asarray(h_[0].data[ys, xs], dtype=np.float64)
        wcs = WCS(hdr).celestial
        sky = wcs.pixel_to_world(xs, ys)

        m = read_healpix(src, verbose=False)
        th, ph = rot(np.pi / 2 - sky.dec.rad, sky.ra.rad)
        want = hp.get_interp_val(m, th, ph, nest=False) * hdr['UNITSCL']

        fin = np.isfinite(got) & np.isfinite(want)
        scale = np.nanpercentile(np.abs(want[fin]), 90) or 1.0
        dev = np.abs(got[fin] - want[fin]) / scale
        ok = bool(fin.mean() > 0.99 and np.nanmax(dev) < 1e-4)
        ok_all &= ok
        print(f'  {"OK  " if ok else "FAIL"} {freq:3d} GHz  '
              f'{fin.sum()}/{n} comparable, max relative deviation '
              f'{np.nanmax(dev):.2e}  [{hdr["BUNIT"]}]')
    return ok_all


# ==========================================================================
# CLI
# ==========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('Usage')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--act', required=True, help='ACT DR6 coadd map (FITS)')
    ap.add_argument('--planck', required=True, nargs='+',
                    help='directory, glob, or list of HFI_SkyMap_*.fits')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--freqs', type=int, nargs='+', default=None,
                    help='subset of 100 143 217 353 545 857 (default: all found)')
    ap.add_argument('--chunk', type=int, default=2048,
                    help='chunk edge in pixels (default 2048; lower it if RAM is tight)')
    ap.add_argument('--act-band', default=None,
                    help='f090/f150/f220, for the header (default: from the filename)')
    ap.add_argument('--geometry', choices=('bbox', 'fullsky'), default='bbox',
                    help='bbox: crop to the ACT active box (default). fullsky: '
                         'the whole sphere at the ACT pixel scale, ACT padded '
                         'where it has no coverage. Use a separate --outdir.')
    ap.add_argument('--act-pad', choices=('nan', 'zero'), default='nan',
                    help='what to put where ACT has no coverage (default nan)')
    ap.add_argument('--tag', default='',
                    help='suffix for the Planck outputs, e.g. --tag R4.00 gives '
                         'Planck_FullSky_Input_217_R4.00.fits. Use it to keep two '
                         'releases of the same frequency side by side.')
    ap.add_argument('--act-only', action='store_true',
                    help='write just ACT_FullSky_GroundTruth on the chosen '
                         'geometry and stop; use when the Planck bands are '
                         'already done and only the ACT truth needs regenerating')
    ap.add_argument('--no-npy', action='store_true', help='FITS only')
    ap.add_argument('--overwrite', action='store_true',
                    help='redo bands whose output already exists')
    ap.add_argument('--verify', type=int, nargs='?', const=2000, default=0,
                    metavar='N', help='after writing, re-sample N random pixels '
                                      'by an independent route and compare')
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args(argv)

    if a.verify_only:
        return 0 if verify(a.outdir, a.planck, n=a.verify or 2000,
                           freqs=a.freqs, tag=a.tag) else 1

    reproject(a.act, a.planck, a.outdir, freqs=a.freqs, chunk=a.chunk,
              write_npy=not a.no_npy, overwrite=a.overwrite, act_band=a.act_band,
              geometry=a.geometry, act_pad=a.act_pad, act_only=a.act_only,
              tag=a.tag)
    if a.verify:
        print('\nverifying...')
        return 0 if verify(a.outdir, a.planck, n=a.verify, freqs=a.freqs,
                           tag=a.tag) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())