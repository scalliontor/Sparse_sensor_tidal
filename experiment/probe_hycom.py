import xarray as xr
import numpy as np

url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh/2024"
print(f"Probing: {url}")
ds = xr.open_dataset(url, decode_times=False)

print("=== Variables ===")
for v in ds.data_vars:
    print(f"  {v}: dims={ds[v].dims}, shape={ds[v].shape}, dtype={ds[v].dtype}")

print("=== Coordinates ===")
for c in ds.coords:
    arr = ds[c].values
    print(f"  {c}: shape={arr.shape}, dtype={arr.dtype}, range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")

print("=== Time ===")
time_var = ds["time"]
units = time_var.attrs.get('units', 'N/A')
cal = time_var.attrs.get('calendar', 'N/A')
print(f"  units: {units}")
print(f"  calendar: {cal}")
print(f"  N timestamps: {len(time_var)}")
print(f"  First 5 values: {time_var.values[:5]}")
print(f"  Last 5 values: {time_var.values[-5:]}")
if len(time_var) > 1:
    dt = np.diff(time_var.values[:20])
    print(f"  First 19 intervals (hours): {dt}")

# Check lat/lon ranges
lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
lat = ds.coords[lat_name].values
lon = ds.coords[lon_name].values
print("=== Spatial ===")
print(f"  lat: [{lat.min():.3f}, {lat.max():.3f}], N={len(lat)}")
print(f"  lon: [{lon.min():.3f}, {lon.max():.3f}], N={len(lon)}")
print(f"  lat spacing: {np.diff(lat[:5])}")
print(f"  lon spacing: {np.diff(lon[:5])}")

ds.close()
