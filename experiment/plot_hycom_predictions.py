"""
HYCOM Prediction Visualization with Cartopy
Generates a multi-panel figure showing:
  (a) Ground truth SSH
  (b) OVCNO predicted mean
  (c) Absolute error  
  (d) Predicted uncertainty (std)
  (e) VCO predicted mean
  (f) VCO absolute error
Uses trained checkpoints from hycom_full run.
"""
import os
import sys
import numpy as np
import torch
import json
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Try cartopy, fallback to plain matplotlib
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not installed, using plain matplotlib")

from dataset_hycom import HYCOMOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE

# ─── Config ───
NC_PATH = "../data/hycom_data/hycom_ssh_tonkin_jan_sep_2024.nc"
STATION_JSON = "hycom_real_k12_stations.json"
CKPT_DIR = "hycom_full"
SEED = 42
T_OBS = 8
SAVE_PATH = "hycom_smoke_v2/hycom_prediction_maps.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load data ───
ds = xr.open_dataset(NC_PATH)
ssh_full = ds['surf_el'].values
lat = ds.coords['latitude'].values
lon = ds.coords['longitude'].values
ds.close()

T_total = ssh_full.shape[0]
train_end = int(T_total * 0.74)
val_end = int(T_total * 0.87)

# Train mean for normalization
train_mean = np.nanmean(ssh_full[:train_end], axis=0)
ocean_mask = ~np.isnan(ssh_full[0])

# Test data (Aug-Sep)
ssh_test = ssh_full[val_end:]
ssh_test_anom = ssh_test - train_mean

print(f"Test data: {ssh_test.shape}, lat=[{lat[0]:.2f},{lat[-1]:.2f}], lon=[{lon[0]:.2f},{lon[-1]:.2f}]")

# ─── Load stations ───
with open(STATION_JSON) as f:
    sdata = json.load(f)
stations = [s for s in sdata['stations'] if s.get('valid_ocean', False)]
station_coords = np.array([[s['i'], s['j']] for s in stations])
station_names = [s['name'] for s in stations]
station_lats = [s['lat'] for s in stations]
station_lons = [s['lon'] for s in stations]
n_sensors = len(stations)

# ─── Load models ───
ovcno = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
ovcno.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"hycom_ovcno_s{SEED}.pt"), map_location=device))
ovcno.eval()

vco = ForecastDeepONetVAE(n_sensors=n_sensors, lstm_hidden=256, latent_dim=256, width=256).to(device)
vco.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"hycom_vco_s{SEED}.pt"), map_location=device))
vco.eval()

# ─── Select test sample ───
# Pick a time in the test period with interesting dynamics
t_start = 20  # ~2.5 days into Aug
t_target = t_start + T_OBS + 4  # +12h forecast

# Build sensor history
x_norm = np.linspace(-1, 1, ssh_test.shape[2], dtype=np.float32)
y_norm = np.linspace(-1, 1, ssh_test.shape[1], dtype=np.float32)
t_norm = np.linspace(-1, 1, ssh_test.shape[0], dtype=np.float32)

sensor_hist = np.zeros((1, T_OBS, n_sensors), dtype=np.float32)
sensor_pts = np.zeros((1, n_sensors, 2), dtype=np.float32)

for i, (yi, xi) in enumerate(station_coords):
    sensor_hist[0, :, i] = ssh_test_anom[t_start:t_start+T_OBS, yi, xi]
    sensor_pts[0, i, 0] = x_norm[xi]
    sensor_pts[0, i, 1] = y_norm[yi]

# Build full-grid query points
ocean_ij = np.argwhere(ocean_mask)  # (N_ocean, 2)
N_ocean = len(ocean_ij)

q_y = y_norm[ocean_ij[:, 0]]
q_x = x_norm[ocean_ij[:, 1]]
q_t = np.full(N_ocean, t_norm[t_target], dtype=np.float32)

# Distance to nearest sensor
dx = q_x[:, None] - sensor_pts[0, :, 0]
dy = q_y[:, None] - sensor_pts[0, :, 1]
dists = np.sqrt(dx**2 + dy**2)
min_dists = np.min(dists, axis=1)

trunk = np.stack([q_x, q_y, q_t, min_dists], axis=-1).astype(np.float32)

# ─── Inference ───
hist_t = torch.tensor(sensor_hist, dtype=torch.float32).to(device)
pts_t = torch.tensor(sensor_pts, dtype=torch.float32).to(device)
trunk_t = torch.tensor(trunk, dtype=torch.float32).to(device)

with torch.no_grad():
    # OVCNO
    ovcno_mu, ovcno_logvar, _, _, ovcno_obs = ovcno(hist_t, pts_t, trunk_t)
    ovcno_mu = ovcno_mu.cpu().numpy().flatten()
    ovcno_std = np.exp(0.5 * ovcno_logvar.cpu().numpy().flatten())
    ovcno_obs_val = torch.sigmoid(ovcno_obs).cpu().numpy().flatten()
    
    # VCO
    vco_mu, vco_logvar, _, _ = vco(hist_t, trunk_t[:, :3])
    vco_mu = vco_mu.cpu().numpy().flatten()
    vco_std = np.exp(0.5 * vco_logvar.cpu().numpy().flatten())

# Ground truth
gt_anom = ssh_test_anom[t_target]

# ─── Reconstruct full-grid maps ───
def scatter_to_grid(values, ocean_ij, shape):
    grid = np.full(shape, np.nan)
    for idx, (yi, xi) in enumerate(ocean_ij):
        grid[yi, xi] = values[idx]
    return grid

Ny, Nx = ssh_test.shape[1], ssh_test.shape[2]
gt_grid = gt_anom
ovcno_mu_grid = scatter_to_grid(ovcno_mu, ocean_ij, (Ny, Nx))
ovcno_std_grid = scatter_to_grid(ovcno_std, ocean_ij, (Ny, Nx))
ovcno_err_grid = scatter_to_grid(np.abs(ovcno_mu - gt_anom[ocean_ij[:, 0], ocean_ij[:, 1]]), ocean_ij, (Ny, Nx))
ovcno_obs_grid = scatter_to_grid(ovcno_obs_val, ocean_ij, (Ny, Nx))
vco_mu_grid = scatter_to_grid(vco_mu, ocean_ij, (Ny, Nx))
vco_err_grid = scatter_to_grid(np.abs(vco_mu - gt_anom[ocean_ij[:, 0], ocean_ij[:, 1]]), ocean_ij, (Ny, Nx))
vco_std_grid = scatter_to_grid(vco_std, ocean_ij, (Ny, Nx))

# ─── Plot ───
lon2d, lat2d = np.meshgrid(lon, lat)

if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 4, figsize=(22, 10),
                              subplot_kw={'projection': proj})
else:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

# Color ranges
ssh_vmin, ssh_vmax = np.nanpercentile(gt_anom[ocean_mask], [2, 98])
err_vmax = np.nanpercentile(np.abs(gt_anom[ocean_mask]), 95) * 0.5
std_vmax = max(np.nanpercentile(ovcno_std_grid[ocean_mask], 95),
               np.nanpercentile(vco_std_grid[ocean_mask], 95))

panels = [
    # Row 1: OVCNO
    (axes[0, 0], gt_grid, 'RdBu_r', ssh_vmin, ssh_vmax, '(a) Ground Truth SSH anomaly'),
    (axes[0, 1], ovcno_mu_grid, 'RdBu_r', ssh_vmin, ssh_vmax, '(b) OVCNO Predicted Mean'),
    (axes[0, 2], ovcno_err_grid, 'hot_r', 0, err_vmax, '(c) OVCNO |Error|'),
    (axes[0, 3], ovcno_std_grid, 'YlOrRd', 0, std_vmax, '(d) OVCNO Pred. Std (σ)'),
    # Row 2: VCO
    (axes[1, 0], ovcno_obs_grid, 'viridis', 0, 1, '(e) OVCNO Observability o(x,y)'),
    (axes[1, 1], vco_mu_grid, 'RdBu_r', ssh_vmin, ssh_vmax, '(f) VCO Predicted Mean'),
    (axes[1, 2], vco_err_grid, 'hot_r', 0, err_vmax, '(g) VCO |Error|'),
    (axes[1, 3], vco_std_grid, 'YlOrRd', 0, std_vmax, '(h) VCO Pred. Std (σ)'),
]

for ax, data, cmap, vmin, vmax, title in panels:
    if HAS_CARTOPY:
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=proj)
        ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='#666666', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#333333')
        im = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=proj, shading='auto')
        # Plot stations
        ax.scatter(station_lons, station_lats, c='lime', s=25, edgecolors='black',
                   linewidths=0.8, zorder=10, transform=proj, marker='^')
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
    else:
        im = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.scatter(station_lons, station_lats, c='lime', s=25, edgecolors='black',
                   linewidths=0.8, zorder=10, marker='^')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

# Compute stats for annotation
ovcno_rmse = np.sqrt(np.nanmean(ovcno_err_grid[ocean_mask]**2))
vco_rmse = np.sqrt(np.nanmean(vco_err_grid[ocean_mask]**2))
ovcno_mean_std = np.nanmean(ovcno_std_grid[ocean_mask])
vco_mean_std = np.nanmean(vco_std_grid[ocean_mask])

fig.suptitle(f'HYCOM SSH Gulf of Tonkin — Test Sample (Aug 2024, +12h forecast)\n'
             f'OVCNO: RMSE={ovcno_rmse:.4f}m, mean σ={ovcno_mean_std:.4f}m  |  '
             f'VCO: RMSE={vco_rmse:.4f}m, mean σ={vco_mean_std:.4f}m',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {SAVE_PATH}")

# Also save high-res for paper
plt.savefig("hycom_smoke_v2/hycom_prediction_maps_hires.png", dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: hycom_smoke_v2/hycom_prediction_maps_hires.png")
