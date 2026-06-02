import xarray as xr
import numpy as np
import h5py
import sys
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/hycom/hycom_ssh_tonkin_2023.h5", help="Output file path")
    parser.add_argument("--url", default="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0", help="OPeNDAP URL")
    args = parser.parse_args()

    print(f"Connecting to HYCOM OPeNDAP: {args.url}")
    # GLBy0.08 expt_93.0 has data from 1994 to ~present.
    # We will fetch 'surf_el' (Sea Surface Height)
    try:
        ds = xr.open_dataset(args.url, decode_times=False)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    # Skip decoding time to avoid "hours since analysis" errors
    # ds = xr.decode_cf(ds)
    # Gulf of Tonkin / South China Sea region
    lon_min, lon_max = 105.0, 110.0
    lat_min, lat_max = 15.0, 22.0
    
    # We will simply take 500 timesteps (which is ~60 days at 3-hourly) from the dataset
    
    print(f"Subsetting data... Lon: {lon_min}-{lon_max}, Lat: {lat_min}-{lat_max}")
    sys.stdout.flush()
    
    subset = ds['surf_el'].sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max)
    ).isel(time=slice(10000, 10500))  # Take 500 steps to download quickly
    
    print(f"Subset shape: {subset.shape} (time, lat, lon)")
    sys.stdout.flush()
    
    print("Downloading data to memory... (this may take a few minutes)")
    sys.stdout.flush()
    t0 = time.time()
    data = subset.compute()
    t1 = time.time()
    print(f"Download complete in {t1-t0:.1f} seconds.")
    sys.stdout.flush()

    # Fill NaNs (land area) with 0 or a large negative number
    # For now, let's just leave it or fill with 0 to make it clean
    data = data.fillna(0.0)

    val = data.values
    t_val = data.time.values
    lat_val = data.lat.values
    lon_val = data.lon.values

    # We need to construct a dataset similar to PDEBench:
    # A set of N samples, each with shape (T, Nx, Ny)
    # The PDEBench had N independent simulations. 
    # Here we have 1 continuous simulation. We will split it into rolling windows.
    # Let's say T=100 (which is 100 * 3 hours = 300 hours = 12.5 days)
    
    T_window = 100
    N_total = len(val)
    if N_total <= T_window:
        print("Not enough timesteps.")
        sys.exit(1)
        
    stride = 2 # Slide window by 2 timesteps (6 hours) to generate samples
    
    windows = []
    
    for i in range(0, N_total - T_window + 1, stride):
        win = val[i:i+T_window]
        windows.append(win)
        
    windows = np.array(windows, dtype=np.float32)
    print(f"Generated {len(windows)} samples of shape {windows.shape[1:]}")
    
    # Save to h5py format similar to PDEBench
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    print(f"Saving to {args.out}...")
    with h5py.File(args.out, "w") as f:
        # Save a reference grid (the models expect it)
        g0 = f.create_group("0000")
        grid = g0.create_group("grid")
        # PDEBench stores coordinates as abstract indices or physical values.
        grid.create_dataset("x", data=lon_val.astype(np.float32))
        grid.create_dataset("y", data=lat_val.astype(np.float32))
        # Abstract time from 0.0 to 1.0
        t_seq = np.linspace(0, 1, T_window, dtype=np.float32)
        grid.create_dataset("t", data=t_seq)
        
        # Save data
        for i in range(len(windows)):
            grp_name = f"{i:04d}"
            if grp_name not in f:
                grp = f.create_group(grp_name)
            else:
                grp = f[grp_name]
            
            # PDEBench format expects shape (T, Nx, Ny, 1) or similar.
            # `val` is (time, lat, lon). Let's save as (T, lon, lat, 1) to match (T, x, y, c)
            # Transpose from (time, lat, lon) to (time, lon, lat)
            sample_data = np.transpose(windows[i], (0, 2, 1))[..., np.newaxis]
            grp.create_dataset("data", data=sample_data)
            
    print("Done!")

if __name__ == "__main__":
    main()
