"""
Download HYCOM GOFS 3.1 GLBy0.08 SSH for Gulf of Tonkin
Domain: 105.5-110.5E, 16.5-22.5N
Period: Jan-May 2024 (for train/val/test split)
Temporal: 3-hourly
"""
import xarray as xr
import numpy as np
import os

# HYCOM OPeNDAP endpoint
URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh/2024"

# Domain bounds
LON_MIN, LON_MAX = 105.5, 110.5
LAT_MIN, LAT_MAX = 16.5, 22.5

OUT_DIR = "../data/hycom_data"
os.makedirs(OUT_DIR, exist_ok=True)

print("Opening HYCOM dataset (remote OPeNDAP)...")
ds = xr.open_dataset(URL, decode_times=False)

# Convert time
import pandas as pd
time_hours = ds['time'].values
base = pd.Timestamp('2000-01-01')
times = pd.to_timedelta(time_hours, unit='h') + base
print(f"Full time range: {times[0]} to {times[-1]}")
print(f"Total timestamps: {len(times)}")

# Find indices for Jan-May 2024
t_start = pd.Timestamp('2024-01-01')
t_end = pd.Timestamp('2024-06-01')  # exclusive
mask_t = (times >= t_start) & (times < t_end)
t_indices = np.where(mask_t)[0]
print(f"Jan-May 2024: {len(t_indices)} timestamps ({times[t_indices[0]]} to {times[t_indices[-1]]})")

# Spatial crop indices
lat = ds.coords['lat'].values
lon = ds.coords['lon'].values

lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)

lat_idx = np.where(lat_mask)[0]
lon_idx = np.where(lon_mask)[0]

print(f"Lat crop: indices [{lat_idx[0]}:{lat_idx[-1]+1}], "
      f"range [{lat[lat_idx[0]]:.3f}, {lat[lat_idx[-1]]:.3f}], N={len(lat_idx)}")
print(f"Lon crop: indices [{lon_idx[0]}:{lon_idx[-1]+1}], "
      f"range [{lon[lon_idx[0]]:.3f}, {lon[lon_idx[-1]]:.3f}], N={len(lon_idx)}")

# Download in monthly chunks to avoid timeout
months = [
    ('2024-01', '2024-02'),
    ('2024-02', '2024-03'),
    ('2024-03', '2024-04'),
    ('2024-04', '2024-05'),
    ('2024-05', '2024-06'),
]

all_data = []
all_times_decoded = []

for m_start, m_end in months:
    ts = pd.Timestamp(m_start)
    te = pd.Timestamp(m_end)
    m_mask = (times >= ts) & (times < te)
    m_idx = np.where(m_mask)[0]
    
    if len(m_idx) == 0:
        print(f"  {m_start}: no data, skipping")
        continue
    
    print(f"  Downloading {m_start}: {len(m_idx)} timesteps "
          f"[{times[m_idx[0]]} to {times[m_idx[-1]]}]...")
    
    # Slice: surf_el[time, lat, lon]
    chunk = ds['surf_el'].isel(
        time=m_idx,
        lat=lat_idx,
        lon=lon_idx
    ).values  # shape: (T_month, Nlat, Nlon)
    
    all_data.append(chunk)
    all_times_decoded.extend(times[m_idx])
    print(f"    Downloaded: shape={chunk.shape}, "
          f"NaN%={np.isnan(chunk).mean()*100:.1f}%")

# Concatenate
ssh_full = np.concatenate(all_data, axis=0)
times_full = np.array(all_times_decoded)

print(f"\n=== Final Dataset ===")
print(f"Shape: {ssh_full.shape} (T, Nlat, Nlon)")
print(f"Time range: {times_full[0]} to {times_full[-1]}")
print(f"NaN fraction: {np.isnan(ssh_full).mean()*100:.1f}%")
print(f"Value range (excl NaN): [{np.nanmin(ssh_full):.4f}, {np.nanmax(ssh_full):.4f}] m")

# Save as NetCDF
lat_crop = lat[lat_idx]
lon_crop = lon[lon_idx]

out_ds = xr.Dataset(
    data_vars={
        'surf_el': (['time', 'latitude', 'longitude'], ssh_full),
    },
    coords={
        'time': times_full,
        'latitude': lat_crop,
        'longitude': lon_crop,
    },
    attrs={
        'source': 'HYCOM GOFS 3.1 GLBy0.08 expt_93.0',
        'variable': 'sea_surface_height (surf_el)',
        'domain': f'{LON_MIN}-{LON_MAX}E, {LAT_MIN}-{LAT_MAX}N',
        'temporal_resolution': '3-hourly',
        'units': 'meters',
    }
)

out_path = os.path.join(OUT_DIR, "hycom_ssh_tonkin_jan_may_2024.nc")
out_ds.to_netcdf(out_path)
print(f"\nSaved to: {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

ds.close()
