"""
Cartopy-based layout comparison: Real-K12 / Equispaced-K12 / Random-K12
Professional ocean figure with real coastlines, land shading, and gridlines.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import json
import xarray as xr

# ─── Config ───
EXTENT = [105.5, 110.5, 16.5, 22.5]  # lon_min, lon_max, lat_min, lat_max
NC_PATH = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"

REAL_JSON = "sensors_real_stations.json"
EQ_JSON   = "sensors_equispaced.json"
RAND_JSON = "sensors_random_seed0.json"

# ─── Load SSH for background ───
ds = xr.open_dataset(NC_PATH)
ssh = ds['sea_surface_height'].values[0, 0]  # first timestep, squeeze depth
lats = ds['latitude'].values
lons = ds['longitude'].values

# Create land mask from NaN
land_mask = np.isnan(ssh)

def load_stations(path):
    with open(path) as f:
        data = json.load(f)
    stations = data.get('stations', data.get('sensors', []))
    valid = [s for s in stations if s.get('i') is not None and s.get('lat') is not None]
    return valid

real_st = load_stations(REAL_JSON)
eq_st   = load_stations(EQ_JSON)
rand_st = load_stations(RAND_JSON)

layouts = [
    ("(a)  Real-K12", real_st, '#D32F2F', '^', True),
    ("(b)  Equispaced-K12", eq_st, '#1565C0', 's', False),
    ("(c)  Random-K12", rand_st, '#2E7D32', 'o', False),
]

# ─── Figure ───
fig, axes = plt.subplots(
    1, 3, figsize=(14, 5.5),
    subplot_kw={'projection': ccrs.PlateCarree()},
)
plt.subplots_adjust(wspace=0.08)

for ax, (title, stations, color, marker, label_names) in zip(axes, layouts):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Ocean background: plot SSH as subtle blue shading
    ssh_plot = np.where(land_mask, np.nan, ssh)
    ax.pcolormesh(
        lons, lats, ssh_plot,
        cmap='Blues', alpha=0.25,
        transform=ccrs.PlateCarree(), zorder=0,
        shading='auto'
    )

    # Land and coastlines
    ax.add_feature(cfeature.LAND, facecolor='#E8E0D8', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#5D4037', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, color='#8D6E63', linestyle='--', zorder=2)

    # Ocean fill for NaN areas already handled by pcolormesh + land feature
    # Plot ocean mask boundary lightly
    ax.contour(
        lons, lats, land_mask.astype(float),
        levels=[0.5], colors=['#795548'], linewidths=0.4,
        transform=ccrs.PlateCarree(), zorder=2
    )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, alpha=0.4,
        color='gray', linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': '#616161'}
    gl.ylabel_style = {'size': 8, 'color': '#616161'}

    # Only show y-labels on leftmost panel
    if ax != axes[0]:
        gl.left_labels = False

    # Plot stations
    slons = [s.get('snapped_lon', s['lon']) for s in stations]
    slats = [s.get('snapped_lat', s['lat']) for s in stations]

    ax.scatter(
        slons, slats,
        marker=marker, s=70, c=color,
        edgecolors='white', linewidths=0.8,
        transform=ccrs.PlateCarree(), zorder=5,
        label=f'K={len(stations)}'
    )

    # Label real station names
    if label_names:
        for s in stations:
            slon = s.get('snapped_lon', s['lon'])
            slat = s.get('snapped_lat', s['lat'])
            name = s.get('name', '')
            if name:
                txt = ax.text(
                    slon + 0.12, slat + 0.12, name,
                    fontsize=6, fontfamily='sans-serif', color='#212121',
                    transform=ccrs.PlateCarree(), zorder=6
                )
                txt.set_path_effects([
                    pe.withStroke(linewidth=2, foreground='white')
                ])

    # Title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8, color='#212121',
                 fontfamily='sans-serif')

    # K count legend
    ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=True,
              framealpha=0.85, edgecolor='#BDBDBD')

plt.savefig("layout_comparison_cartopy.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("layout_comparison_cartopy.pdf", bbox_inches='tight', facecolor='white')
print("Saved layout_comparison_cartopy.png and .pdf")
