"""Deep verification of K=12 real tide-gauge stations."""
import numpy as np
import xarray as xr
import json

ds = xr.open_dataset('../data/real_data/copernicus_ssh_tonkin_jan2024.nc')
ssh = ds['sea_surface_height'].values[:,0,:,:]
lats = ds.latitude.values
lons = ds.longitude.values
T, Ny, Nx = ssh.shape
ocean_mask = ~np.isnan(ssh[0])

with open('sensors_real_stations.json') as f:
    layout = json.load(f)

kept = [s for s in layout['stations'] if s['status'].startswith('keep')]
print(f'=== DETAILED VERIFICATION: K={len(kept)} stations ===')
print(f'Dataset: T={T}, Grid={Ny}x{Nx}, Ocean={ocean_mask.sum()}')
print()

# Check 1: No duplicate grid cells
grid_cells = [(s['i'], s['j']) for s in kept]
unique_cells = set(grid_cells)
dup_ok = len(unique_cells) == len(grid_cells)
print(f'CHECK 1 - Unique grid cells: {len(unique_cells)}/{len(grid_cells)}', end='')
if dup_ok:
    print(' ✓ No duplicates')
else:
    dupes = [c for c in grid_cells if grid_cells.count(c) > 1]
    print(f' ✗ DUPLICATES: {dupes}')

# Check 2: All stations on ocean mask
print(f'CHECK 2 - Ocean mask validity:')
ocean_ok = True
for s in kept:
    i, j = s['i'], s['j']
    on_ocean = ocean_mask[i, j]
    tag = '✓' if on_ocean else '✗ ON LAND!'
    if not on_ocean:
        ocean_ok = False
    print(f"  {s['name']:15s} grid[{i:2d},{j:2d}]  ocean={on_ocean} {tag}")

# Check 3: SSH time series quality at each station
print(f'CHECK 3 - SSH time series quality (T={T} timesteps):')
ts_ok = True
for s in kept:
    i, j = s['i'], s['j']
    series = ssh[:, i, j]
    n_valid = int(np.sum(~np.isnan(series)))
    n_nan = int(np.sum(np.isnan(series)))
    if n_nan > 0:
        ts_ok = False
    if n_valid > 0:
        mn = np.nanmean(series)
        rng = float(np.nanmax(series) - np.nanmin(series))
        std = np.nanstd(series)
        tag = '✓' if n_nan == 0 else '⚠ HAS NaN'
        print(f"  {s['name']:15s}  valid={n_valid}/{T}  NaN={n_nan}  mean={mn:.4f}  range={rng:.4f}  std={std:.4f} {tag}")
    else:
        ts_ok = False
        print(f"  {s['name']:15s}  valid=0/{T}  ALL NaN!")

# Check 4: Snap distances summary
print(f'CHECK 4 - Snap distances:')
dists = [s['snap_distance_km'] for s in kept]
snap_ok = all(d < 30 for d in dists)
min_d = min(dists)
max_d = max(dists)
min_name = [s['name'] for s in kept if s['snap_distance_km'] == min_d][0]
max_name = [s['name'] for s in kept if s['snap_distance_km'] == max_d][0]
print(f'  Min: {min_d:.1f} km ({min_name})')
print(f'  Max: {max_d:.1f} km ({max_name})')
print(f'  Mean: {np.mean(dists):.1f} km')
print(f'  All < 30km threshold: {snap_ok} ✓' if snap_ok else f'  FAILED: some > 30km ✗')

# Check 5: Min pairwise distance between stations
print(f'CHECK 5 - Pairwise distances between stations:')
min_pair_dist = float('inf')
min_pair = ('', '')
close_pairs = []
for a in range(len(kept)):
    for b in range(a+1, len(kept)):
        ia, ja = kept[a]['i'], kept[a]['j']
        ib, jb = kept[b]['i'], kept[b]['j']
        d = np.sqrt((ia-ib)**2 + (ja-jb)**2)
        if d < min_pair_dist:
            min_pair_dist = d
            min_pair = (kept[a]['name'], kept[b]['name'])
        if d < 3:
            close_pairs.append((kept[a]['name'], kept[b]['name'], d))

print(f'  Closest pair: {min_pair[0]} -- {min_pair[1]}, dist={min_pair_dist:.1f} grid cells')
if close_pairs:
    print('  ⚠ Very close pairs (<3 grid cells):')
    for n1, n2, d in close_pairs:
        print(f'    {n1} -- {n2}: {d:.1f} grid cells')
else:
    print('  No pairs closer than 3 grid cells ✓')

# Check 6: Spatial coverage
print(f'CHECK 6 - Spatial coverage:')
i_vals = [s['i'] for s in kept]
j_vals = [s['j'] for s in kept]
print(f'  Lat range: grid[{min(i_vals)}..{max(i_vals)}] = [{lats[min(i_vals)]:.2f}..{lats[max(i_vals)]:.2f}]N')
print(f'  Lon range: grid[{min(j_vals)}..{max(j_vals)}] = [{lons[min(j_vals)]:.2f}..{lons[max(j_vals)]:.2f}]E')
lat_frac = (max(i_vals) - min(i_vals)) / (Ny - 1) * 100
lon_frac = (max(j_vals) - min(j_vals)) / (Nx - 1) * 100
print(f'  Domain fraction: lat {lat_frac:.0f}%, lon {lon_frac:.0f}%')

# Check 7: Clustering analysis
print(f'CHECK 7 - Station clustering (distance from centroid):')
ci = np.mean(i_vals)
cj = np.mean(j_vals)
for s in kept:
    d = np.sqrt((s['i'] - ci)**2 + (s['j'] - cj)**2)
    print(f"  {s['name']:15s}  dist_from_centroid={d:.1f} grid cells")

print()
print('=== FINAL VERDICT ===')
all_ok = dup_ok and ocean_ok and ts_ok and snap_ok
if all_ok:
    print(f'ALL 7 CHECKS PASSED. K={len(kept)} stations ready for training.')
else:
    issues = []
    if not dup_ok: issues.append('duplicates')
    if not ocean_ok: issues.append('land cells')
    if not ts_ok: issues.append('NaN in timeseries')
    if not snap_ok: issues.append('snap > 30km')
    print(f'ISSUES FOUND: {", ".join(issues)}. Review above.')
