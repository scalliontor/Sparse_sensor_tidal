"""
Download HYCOM GOFS 3.1 SSH for Gulf of Tonkin — EXTENDED
Domain: 105.5-110.5E, 16.5-22.5N
Period: Jan-Sep 2024 (all available)
Split: Jan-Jun train, Jul val, Aug-Sep test
"""
import xarray as xr
import numpy as np
import os
import pandas as pd

URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh/2024"
LON_MIN, LON_MAX = 105.5, 110.5
LAT_MIN, LAT_MAX = 16.5, 22.5
OUT_DIR = "../data/hycom_data"
os.makedirs(OUT_DIR, exist_ok=True)

print("Opening HYCOM dataset (remote OPeNDAP)...")
ds = xr.open_dataset(URL, decode_times=False)

time_hours = ds['time'].values
base = pd.Timestamp('2000-01-01')
times = pd.to_timedelta(time_hours, unit='h') + base
print(f"Full time range: {times[0]} to {times[-1]}")

# Spatial crop
lat = ds.coords['lat'].values
lon = ds.coords['lon'].values
lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
lat_idx = np.where(lat_mask)[0]
lon_idx = np.where(lon_mask)[0]
print(f"Lat: [{lat[lat_idx[0]]:.3f}, {lat[lat_idx[-1]]:.3f}], N={len(lat_idx)}")
print(f"Lon: [{lon[lon_idx[0]]:.3f}, {lon[lon_idx[-1]]:.3f}], N={len(lon_idx)}")

# Download all available 2024 data in monthly chunks
months = [
    ('2024-01', '2024-02'),
    ('2024-02', '2024-03'),
    ('2024-03', '2024-04'),
    ('2024-04', '2024-05'),
    ('2024-05', '2024-06'),
    ('2024-06', '2024-07'),
    ('2024-07', '2024-08'),
    ('2024-08', '2024-09'),
    ('2024-09', '2024-10'),
]

all_data = []
all_times = []

for m_start, m_end in months:
    ts = pd.Timestamp(m_start)
    te = pd.Timestamp(m_end)
    m_mask = (times >= ts) & (times < te)
    m_indices = np.where(m_mask)[0]
    
    if len(m_indices) == 0:
        print(f"  {m_start}: no data, skipping")
        continue
    
    print(f"  Downloading {m_start}: {len(m_indices)} timesteps "
          f"[{times[m_indices[0]]} to {times[m_indices[-1]]}]...")
    
    chunk = ds['surf_el'].isel(
        time=m_indices, lat=lat_idx, lon=lon_idx
    ).values
    
    all_data.append(chunk)
    all_times.extend(times[m_indices])
    print(f"    shape={chunk.shape}, NaN%={np.isnan(chunk).mean()*100:.1f}%")

ssh_full = np.concatenate(all_data, axis=0)
times_full = np.array(all_times)

print(f"\n=== Final Dataset ===")
print(f"Shape: {ssh_full.shape}")
print(f"Time: {times_full[0]} to {times_full[-1]}")
print(f"NaN%: {np.isnan(ssh_full).mean()*100:.1f}%")
print(f"SSH range: [{np.nanmin(ssh_full):.4f}, {np.nanmax(ssh_full):.4f}] m")

# Save
lat_crop = lat[lat_idx]
lon_crop = lon[lon_idx]

out_ds = xr.Dataset(
    data_vars={'surf_el': (['time', 'latitude', 'longitude'], ssh_full)},
    coords={'time': times_full, 'latitude': lat_crop, 'longitude': lon_crop},
    attrs={
        'source': 'HYCOM GOFS 3.1 GLBy0.08 expt_93.0',
        'variable': 'sea_surface_height (surf_el)',
        'domain': f'{LON_MIN}-{LON_MAX}E, {LAT_MIN}-{LAT_MAX}N',
        'temporal_resolution': '3-hourly',
        'units': 'meters',
    }
)

out_path = os.path.join(OUT_DIR, "hycom_ssh_tonkin_jan_sep_2024.nc")
out_ds.to_netcdf(out_path)
print(f"\nSaved: {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1e6:.1f} MB")

# Print month-by-month counts for split configuration
print("\n=== Monthly step counts ===")
for m in range(1, 10):
    ms = pd.Timestamp(f'2024-{m:02d}-01')
    me = pd.Timestamp(f'2024-{m+1:02d}-01') if m < 12 else pd.Timestamp('2025-01-01')
    cnt = np.sum((times_full >= ms) & (times_full < me))
    print(f"  2024-{m:02d}: {cnt} steps")

ds.close()
