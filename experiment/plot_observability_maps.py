import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
import matplotlib
matplotlib.use('Agg')

def plot_observability_maps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    
    ds_train = CopernicusOVCNODataset(nc_path, n_sensors=16, pts_per_sample=10, split="train", variable_sensors=False)
    
    ds = CopernicusOVCNODataset(nc_path, n_sensors=16, pts_per_sample=4000, 
                                split="val", variable_sensors=False,
                                train_mean=ds_train.train_mean)
    ds.pts = len(ds.ocean_coords)
    
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    model.load_state_dict(torch.load("ovcno_checkpoint.pt", map_location=device))
    model.eval()

    hist, pts, trunk, labels = next(iter(dl))
    B, P = trunk.shape[0], trunk.shape[1]
    
    with torch.no_grad():
        # latent_sample=False to plot the mean without sampling noise for visualization
        y_mu, y_logvar, mz, lz, o_i = model(hist.to(device), pts.to(device), trunk.view(B*P, 4).to(device), sample_z=False)
        
    y_mu = y_mu.cpu().numpy().flatten()
    y_var = torch.exp(y_logvar).cpu().numpy().flatten()
    o_i = o_i.cpu().numpy().flatten()
    labels = labels.numpy().flatten()
    
    err = np.abs(y_mu - labels)
    
    import xarray as xr
    try:
        raw_ds = xr.open_dataset(nc_path)
    except:
        raw_ds = xr.open_dataset("../data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_105.5-110.5E_16.5-22.5N.nc")
    
    if 'zos' in raw_ds:
        ssh = raw_ds['zos'].values
    elif 'sea_surface_height' in raw_ds:
        ssh = raw_ds['sea_surface_height'].values
    else:
        ssh = raw_ds['sla'].values
        
    Ny, Nx = 73, 61
    mask = (~np.isnan(ssh[0])).squeeze()
    if mask.ndim > 2:
        mask = mask[0]
    
    grid_gt = np.full((Ny, Nx), np.nan)
    grid_mu = np.full((Ny, Nx), np.nan)
    grid_err = np.full((Ny, Nx), np.nan)
    grid_o = np.full((Ny, Nx), np.nan)
    grid_unc = np.full((Ny, Nx), np.nan)
    
    trunk_np = trunk.view(P, 4).numpy()
    x_norm = np.linspace(-1, 1, Nx, dtype=np.float32)
    y_norm = np.linspace(-1, 1, Ny, dtype=np.float32)
    
    for i in range(P):
        qx, qy = trunk_np[i, 0], trunk_np[i, 1]
        x_idx = np.argmin(np.abs(x_norm - qx))
        y_idx = np.argmin(np.abs(y_norm - qy))
        
        grid_gt[y_idx, x_idx] = labels[i]
        grid_mu[y_idx, x_idx] = y_mu[i]
        grid_err[y_idx, x_idx] = err[i]
        grid_o[y_idx, x_idx] = o_i[i]
        grid_unc[y_idx, x_idx] = y_var[i]
        
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)
    
    # Pre-draw land mask
    land_cmap = mcolors.ListedColormap(['#e0e0e0', '#00000000'])
    for ax in axes:
        ax.imshow(mask, origin='lower', cmap=land_cmap)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # 1. Ground Truth
    im0 = axes[0].imshow(grid_gt, origin='lower', cmap='seismic', vmin=-0.6, vmax=0.6)
    axes[0].set_title("Ground Truth SSH", fontsize=14, fontweight='bold', pad=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, shrink=0.8)
    
    # 2. Predicted
    im1 = axes[1].imshow(grid_mu, origin='lower', cmap='seismic', vmin=-0.6, vmax=0.6)
    axes[1].set_title("Predicted Mean", fontsize=14, fontweight='bold', pad=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, shrink=0.8)
    
    # 3. Absolute Error
    im2 = axes[2].imshow(grid_err, origin='lower', cmap='Reds', vmin=0, vmax=0.15)
    axes[2].set_title("Absolute Error", fontsize=14, fontweight='bold', pad=10)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, shrink=0.8)
    
    # 4. Observability
    im3 = axes[3].imshow(grid_o, origin='lower', cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title("Learned Observability $o_\psi$", fontsize=14, fontweight='bold', pad=10)
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, shrink=0.8)
    
    # 5. Uncertainty
    im4 = axes[4].imshow(grid_unc, origin='lower', cmap='plasma', vmin=0)
    axes[4].set_title("Predictive Variance $\sigma^2_\eta$", fontsize=14, fontweight='bold', pad=10)
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04, shrink=0.8)
    
    # Plot sensors
    pts_np = pts[0].numpy()
    for ax in axes:
        for k in range(pts_np.shape[0]):
            kx = np.argmin(np.abs(x_norm - pts_np[k, 0]))
            ky = np.argmin(np.abs(y_norm - pts_np[k, 1]))
            ax.plot(kx, ky, marker='^', color='black', markersize=8, markeredgecolor='white', markeredgewidth=1)
            
    plt.savefig("observability_maps.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved observability_maps.png")

if __name__ == "__main__":
    plot_observability_maps()

