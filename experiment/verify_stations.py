"""
Verify real tide-gauge station coordinates against the Copernicus SSH grid.
Snap to nearest ocean cell, report distances, generate station map.
"""
import numpy as np
import xarray as xr
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Real tide-gauge stations in / near Gulf of Tonkin domain ──
# Domain: 16.5°N–22.5°N, 105.5°E–110.5°E
STATIONS = [
    {"name": "Hon Dau",       "lat": 20.67, "lon": 106.80, "country": "Vietnam"},
    {"name": "Hon Gai",       "lat": 20.95, "lon": 107.07, "country": "Vietnam"},
    {"name": "Cua Ong",       "lat": 21.03, "lon": 107.37, "country": "Vietnam"},
    {"name": "Co To",         "lat": 20.97, "lon": 107.77, "country": "Vietnam"},
    {"name": "Bach Long Vi",  "lat": 20.13, "lon": 107.72, "country": "Vietnam"},
    {"name": "Dong Hoi",      "lat": 17.70, "lon": 106.47, "country": "Vietnam"},
    {"name": "Cua Lo",        "lat": 18.80, "lon": 105.77, "country": "Vietnam"},
    {"name": "Son Tra",       "lat": 16.10, "lon": 108.22, "country": "Vietnam"},  # likely out of domain
    {"name": "Beihai",        "lat": 21.48, "lon": 109.10, "country": "China"},
    {"name": "Dongfang",      "lat": 19.10, "lon": 108.62, "country": "China"},
    {"name": "Haikou",        "lat": 20.02, "lon": 110.28, "country": "China"},
    {"name": "Bac Hai",       "lat": 21.42, "lon": 107.68, "country": "Vietnam"},  # needs verification
    {"name": "Vinh",          "lat": 18.68, "lon": 105.67, "country": "Vietnam"},
]

SNAP_THRESHOLD_KM = 30.0

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def main():
    # Load dataset
    ds = xr.open_dataset('../data/real_data/copernicus_ssh_tonkin_jan2024.nc')
    ssh = ds['sea_surface_height'].values[:, 0, :, :]  # (T, Ny, Nx)
    lats = ds.latitude.values   # (Ny,)
    lons = ds.longitude.values  # (Nx,)
    
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    print(f"Domain: [{lat_min:.2f}, {lat_max:.2f}]N x [{lon_min:.2f}, {lon_max:.2f}]E")
    print(f"Grid: {len(lats)} x {len(lons)}")
    
    ocean_mask = ~np.isnan(ssh[0])  # (Ny, Nx)
    print(f"Ocean points: {ocean_mask.sum()}")
    
    # Ocean cell coordinates
    ocean_ij = np.argwhere(ocean_mask)  # (N_ocean, 2) -> [i_lat, j_lon]
    ocean_lats = lats[ocean_ij[:, 0]]
    ocean_lons = lons[ocean_ij[:, 1]]
    
    results = []
    kept = []
    excluded = []
    
    for s in STATIONS:
        name = s["name"]
        slat, slon = s["lat"], s["lon"]
        
        # Check domain bounds
        in_domain = (lat_min <= slat <= lat_max) and (lon_min <= slon <= lon_max)
        
        if not in_domain:
            s_result = {**s, "status": "excluded_domain", "reason": "outside domain bounds",
                       "snapped_lat": None, "snapped_lon": None, "i": None, "j": None,
                       "snap_distance_km": None}
            results.append(s_result)
            excluded.append(s_result)
            print(f"  ✗ {name}: lat={slat}, lon={slon} — OUTSIDE DOMAIN")
            continue
        
        # Find nearest grid cell
        i_lat = np.argmin(np.abs(lats - slat))
        j_lon = np.argmin(np.abs(lons - slon))
        
        # Check if this cell is ocean
        if ocean_mask[i_lat, j_lon]:
            snap_lat = float(lats[i_lat])
            snap_lon = float(lons[j_lon])
            dist_km = haversine_km(slat, slon, snap_lat, snap_lon)
            s_result = {**s, "status": "keep", "reason": "direct ocean cell",
                       "snapped_lat": snap_lat, "snapped_lon": snap_lon,
                       "i": int(i_lat), "j": int(j_lon), "snap_distance_km": round(dist_km, 2)}
            results.append(s_result)
            kept.append(s_result)
            print(f"  ✓ {name}: ({slat},{slon}) → grid[{i_lat},{j_lon}] = ({snap_lat:.3f},{snap_lon:.3f}), snap={dist_km:.1f}km")
        else:
            # Snap to nearest ocean cell
            dists = np.array([haversine_km(slat, slon, olat, olon) 
                            for olat, olon in zip(ocean_lats, ocean_lons)])
            best_idx = np.argmin(dists)
            best_dist = dists[best_idx]
            best_i, best_j = ocean_ij[best_idx]
            snap_lat = float(lats[best_i])
            snap_lon = float(lons[best_j])
            
            if best_dist <= SNAP_THRESHOLD_KM:
                s_result = {**s, "status": "keep_snapped", "reason": f"snapped to ocean ({best_dist:.1f}km)",
                           "snapped_lat": snap_lat, "snapped_lon": snap_lon,
                           "i": int(best_i), "j": int(best_j), "snap_distance_km": round(best_dist, 2)}
                results.append(s_result)
                kept.append(s_result)
                print(f"  ~ {name}: ({slat},{slon}) → LAND, snapped to ({snap_lat:.3f},{snap_lon:.3f}), dist={best_dist:.1f}km")
            else:
                s_result = {**s, "status": "excluded_snap", "reason": f"snap distance {best_dist:.1f}km > {SNAP_THRESHOLD_KM}km",
                           "snapped_lat": snap_lat, "snapped_lon": snap_lon,
                           "i": int(best_i), "j": int(best_j), "snap_distance_km": round(best_dist, 2)}
                results.append(s_result)
                excluded.append(s_result)
                print(f"  ✗ {name}: ({slat},{slon}) → nearest ocean at {best_dist:.1f}km — EXCLUDED (>{SNAP_THRESHOLD_KM}km)")
    
    K = len(kept)
    print(f"\n{'='*60}")
    print(f"KEPT: {K} stations")
    for s in kept:
        print(f"  {s['name']:15s}  grid[{s['i']:2d},{s['j']:2d}]  snap={s['snap_distance_km']:.1f}km")
    print(f"EXCLUDED: {len(excluded)} stations")
    for s in excluded:
        print(f"  {s['name']:15s}  reason: {s['reason']}")
    
    # Save JSON
    layout = {
        "layout_name": f"real_station_k{K}",
        "level": "Level-1 Realistic OSSE",
        "note": "Real station coordinates, model-sampled sensor values",
        "domain": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        "K": K,
        "stations": results
    }
    with open("sensors_real_stations.json", "w") as f:
        json.dump(layout, f, indent=2)
    print(f"\nSaved sensors_real_stations.json")
    
    # ── Generate station map ──
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ocean mask
    extent = [lon_min, lon_max, lat_min, lat_max]
    ax.imshow(ocean_mask, origin='lower', extent=extent, cmap='Blues', alpha=0.3, aspect='auto')
    
    # Plot ocean boundary 
    from scipy import ndimage
    boundary = ndimage.binary_dilation(ocean_mask, iterations=1) & ~ocean_mask
    # Plot land as gray
    land_mask = ~ocean_mask
    ax.imshow(np.where(land_mask, 1.0, np.nan), origin='lower', extent=extent, 
              cmap='Greys', alpha=0.15, aspect='auto', vmin=0, vmax=1)
    
    # Plot kept stations
    for s in kept:
        ax.plot(s['lon'], s['lat'], 'r^', markersize=12, markeredgecolor='darkred', markeredgewidth=1.5, zorder=5)
        ax.annotate(s['name'], (s['lon'], s['lat']), fontsize=8, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
        # Draw snap line if snapped
        if s.get('snapped_lat') and s['snap_distance_km'] > 1:
            ax.plot([s['lon'], s['snapped_lon']], [s['lat'], s['snapped_lat']], 
                   'r--', alpha=0.5, linewidth=0.8)
    
    # Plot excluded stations
    for s in excluded:
        ax.plot(s['lon'], s['lat'], 'kx', markersize=10, markeredgewidth=2, zorder=5)
        ax.annotate(f"{s['name']} (excl.)", (s['lon'], s['lat']), fontsize=7, color='gray',
                   xytext=(5, -10), textcoords='offset points')
    
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(f'Real Tide-Gauge Stations in Gulf of Tonkin Domain (K={K} kept)', fontsize=13)
    ax.set_xlim(lon_min - 0.2, lon_max + 0.2)
    ax.set_ylim(lat_min - 0.2, lat_max + 0.2)
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markeredgecolor='darkred', markersize=10, label=f'Kept stations (K={K})'),
        Line2D([0], [0], marker='x', color='black', markersize=8, linestyle='None', label=f'Excluded ({len(excluded)})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('station_map.png', dpi=150, bbox_inches='tight')
    print(f"Saved station_map.png")
    
    # Also generate equispaced-K and random-K layouts for comparison
    # Equispaced: pick K ocean cells evenly spaced along the boundary
    boundary_cells = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            if ocean_mask[i, j]:
                # Check if it's a boundary cell (has at least one land/edge neighbor)
                is_boundary = False
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if ni < 0 or ni >= len(lats) or nj < 0 or nj >= len(lons) or not ocean_mask[ni, nj]:
                        is_boundary = True
                        break
                if is_boundary:
                    boundary_cells.append((i, j))
    
    boundary_cells = np.array(boundary_cells)
    print(f"\nBoundary ocean cells: {len(boundary_cells)}")
    
    # Equispaced: select K evenly from boundary
    indices = np.linspace(0, len(boundary_cells)-1, K, dtype=int)
    eq_cells = boundary_cells[indices]
    eq_layout = {
        "layout_name": f"equispaced_k{K}",
        "K": K,
        "stations": [{"name": f"EQ_{idx}", "i": int(i), "j": int(j),
                      "lat": float(lats[i]), "lon": float(lons[j])}
                     for idx, (i, j) in enumerate(eq_cells)]
    }
    with open("sensors_equispaced.json", "w") as f:
        json.dump(eq_layout, f, indent=2)
    print(f"Saved sensors_equispaced.json (K={K})")
    
    # Random: 10 random layouts
    rng = np.random.default_rng(2024)
    for seed in range(10):
        rng_s = np.random.default_rng(seed)
        rand_idx = rng_s.choice(len(boundary_cells), K, replace=False)
        rand_cells = boundary_cells[rand_idx]
        rand_layout = {
            "layout_name": f"random_k{K}_seed{seed}",
            "K": K,
            "seed": seed,
            "stations": [{"name": f"RND_{idx}", "i": int(i), "j": int(j),
                          "lat": float(lats[i]), "lon": float(lons[j])}
                         for idx, (i, j) in enumerate(rand_cells)]
        }
        with open(f"sensors_random_seed{seed}.json", "w") as f:
            json.dump(rand_layout, f, indent=2)
    print(f"Saved 10 random layouts")

if __name__ == "__main__":
    main()
