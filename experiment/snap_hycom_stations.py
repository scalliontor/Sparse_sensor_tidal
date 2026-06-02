"""
HYCOM Station Snapping + Data Inspection
1. Create HYCOM ocean mask
2. Snap Real-K12 stations to nearest HYCOM ocean cell
3. Save hycom_real_k12_stations.json
4. Print summary comparison with Copernicus
"""
import xarray as xr
import numpy as np
import json
import os

# Load HYCOM data
NC_PATH = "../data/hycom_data/hycom_ssh_tonkin_jan_may_2024.nc"
ds = xr.open_dataset(NC_PATH)
ssh = ds['surf_el'].values  # (T, Nlat, Nlon)
lats = ds['latitude'].values
lons = ds['longitude'].values
T, Nlat, Nlon = ssh.shape

print(f"HYCOM Grid: {Nlat} x {Nlon} = {Nlat*Nlon} total cells")
print(f"Lat range: [{lats.min():.3f}, {lats.max():.3f}]")
print(f"Lon range: [{lons.min():.3f}, {lons.max():.3f}]")
print(f"Lat spacing: {np.diff(lats[:3])}")
print(f"Lon spacing: {np.diff(lons[:3])}")

# Ocean mask: a cell is ocean if it has valid (non-NaN) SSH at any timestep
ocean_mask = ~np.all(np.isnan(ssh), axis=0)  # (Nlat, Nlon)
n_ocean = ocean_mask.sum()
print(f"Ocean cells: {n_ocean} / {Nlat*Nlon} ({n_ocean/(Nlat*Nlon)*100:.1f}%)")

# Load Real-K12 stations from Copernicus OSSE
with open("sensors_real_stations.json") as f:
    cop_data = json.load(f)

stations = cop_data['stations']
valid_stations = [s for s in stations if s.get('status', '').startswith('keep')]
print(f"\nReal-K12 stations to snap: {len(valid_stations)}")

# Haversine distance
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# Snap each station
hycom_stations = []
print(f"\n{'Station':<16} {'OrigLat':>8} {'OrigLon':>8} | {'SnapLat':>8} {'SnapLon':>8} {'i':>4} {'j':>4} {'dist_km':>8} {'ocean':>6}")
print("-" * 90)

for s in valid_stations:
    name = s['name']
    slat, slon = s['lat'], s['lon']
    
    # Find nearest ocean cell
    lat_diffs = np.abs(lats - slat)
    lon_diffs = np.abs(lons - slon)
    
    # Search in a local window
    best_dist = 999
    best_i, best_j = None, None
    
    for i in range(Nlat):
        for j in range(Nlon):
            if not ocean_mask[i, j]:
                continue
            d = haversine_km(slat, slon, lats[i], lons[j])
            if d < best_dist:
                best_dist = d
                best_i, best_j = i, j
    
    snap_lat = float(lats[best_i]) if best_i is not None else None
    snap_lon = float(lons[best_j]) if best_j is not None else None
    is_valid = bool(best_dist < 30.0)
    
    status = "keep" if best_dist < 10 else ("keep_snapped" if best_dist < 30 else "excluded")
    
    print(f"{name:<16} {slat:>8.3f} {slon:>8.3f} | {snap_lat:>8.3f} {snap_lon:>8.3f} {best_i:>4} {best_j:>4} {best_dist:>8.2f} {'YES' if is_valid else 'NO':>6}")
    
    hycom_stations.append({
        "name": name,
        "lat": slat,
        "lon": slon,
        "country": s.get('country', ''),
        "status": status,
        "snapped_lat": snap_lat,
        "snapped_lon": snap_lon,
        "i": int(best_i) if best_i is not None else None,
        "j": int(best_j) if best_j is not None else None,
        "snap_distance_km": round(best_dist, 2),
        "valid_ocean": is_valid
    })

# Save
out = {
    "dataset": "HYCOM_GOFS_3.1_GLBy0.08",
    "domain": {
        "lon_min": 105.5, "lon_max": 110.5,
        "lat_min": 16.5, "lat_max": 22.5
    },
    "grid": {"Nlat": Nlat, "Nlon": Nlon, "n_ocean": int(n_ocean)},
    "temporal_resolution": "3-hourly",
    "K": len([s for s in hycom_stations if s['valid_ocean']]),
    "stations": hycom_stations
}

out_path = "hycom_real_k12_stations.json"
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Valid stations: {out['K']} / {len(hycom_stations)}")

# Compare with Copernicus
print("\n=== Comparison: Copernicus vs HYCOM ===")
print(f"{'':>20} {'Copernicus':>12} {'HYCOM':>12}")
print(f"{'Grid':>20} {'73 x 61':>12} {f'{Nlat} x {Nlon}':>12}")
print(f"{'Ocean points':>20} {'~2520':>12} {n_ocean:>12}")
print(f"{'Temporal res':>20} {'1-hourly':>12} {'3-hourly':>12}")
print(f"{'SSH range (m)':>20} {'anomaly':>12} {f'{np.nanmin(ssh):.3f}-{np.nanmax(ssh):.3f}':>12}")

ds.close()
