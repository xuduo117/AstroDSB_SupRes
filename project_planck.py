import os
import numpy as np
import healpy as hp
from pixell import enmap, reproject
import time

def process_planck_to_act_grid():
    # 1. Define Paths
    input_fits = '/scratch/xuduo117/Real_Obs_Data/Planck_Data/HFI_SkyMap_217_2048_R4.00_full.fits'
    
    # Create a clean output directory in your scratch space
    output_dir = '/scratch/xuduo117/Real_Obs_Data/Planck_Data/ACT_Resolution_Maps'
    os.makedirs(output_dir, exist_ok=True)
    
    output_fits = os.path.join(output_dir, 'Planck_217GHz_0.5arcmin_CAR.fits')
    output_npy = os.path.join(output_dir, 'Planck_217GHz_0.5arcmin_CAR.npy')

    print(f"--- Starting Full-Sky Projection Pipeline ---")
    start_time = time.time()

    # 2. Load the HEALPix Map
    print(f"\n1. Loading Planck HEALPix map from:\n   {input_fits}")
    # field=0 loads the Temperature (Intensity) map. 
    # If you need polarization (Q, U), you would use field=(0,1,2)
    planck_map = hp.read_map(input_fits, field=0)
    print(f"   -> Map loaded. Nside: {hp.npix2nside(len(planck_map))}")

    # 3. Define the Target Geometry (ACT 0.5 arcmin resolution)
    print("\n2. Defining target CAR geometry (0.5 arcmin resolution)...")
    res_rad = np.deg2rad(0.5 / 60.0)
    shape_fs, wcs_fs = enmap.fullsky_geometry(res=res_rad)
    
    # A 0.5 arcmin full-sky map will be massive (approx 21600 x 43200 pixels)
    print(f"   -> Target matrix shape: {shape_fs}")
    estimated_ram = (shape_fs[0] * shape_fs[1] * 8) / (1024**3)
    print(f"   -> Estimated raw array size: ~{estimated_ram:.2f} GB")

    # 4. Perform the Reprojection
    print("\n3. Reprojecting HEALPix to CAR...")
    print("   (This will take several minutes and use significant RAM. Please wait...)")
    
    # reproject.healpix2map handles the interpolation from spherical to flat pixels
    projected_map = reproject.healpix2map(planck_map, shape_fs, wcs_fs)

    # 5. Save the FITS file (Preserves WCS coordinates)
    print(f"\n4. Saving WCS-aware FITS file...")
    enmap.write_map(output_fits, projected_map)
    print(f"   -> Saved: {output_fits}")

    # 6. Save the ML-ready Numpy Array
    print(f"5. Saving raw Numpy array for Deep Learning...")
    # Convert the enmap object back to a pure numpy array to ensure compatibility
    np.save(output_npy, np.asarray(projected_map))
    print(f"   -> Saved: {output_npy}")

    elapsed = (time.time() - start_time) / 60.0
    print(f"\n✅ Pipeline complete in {elapsed:.1f} minutes!")

if __name__ == "__main__":
    process_planck_to_act_grid()