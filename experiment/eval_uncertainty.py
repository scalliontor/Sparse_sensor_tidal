import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

from model_vae import ForecastDeepONetVAE
from dataset_vae import CopernicusVAEDataset

def evaluate_uncertainty():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating uncertainty on device:", device)

    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    val_ds = CopernicusVAEDataset(nc_path, split="val", pts_per_sample=1024, seed=99)
    # We don't really need a large dataloader, just a few samples to plot uncertainty field
    
    # Pre-calculate spatial distance from all ocean points to nearest sensor
    sensor_pts = val_ds.sensor_coords  # (16, 2)
    ocean_pts = val_ds.ocean_coords    # (N_ocean, 2)
    
    print("Computing baseline distances...")
    # Distance in grid cells (or approximate euclidian)
    distances = cdist(ocean_pts, sensor_pts, metric='euclidean')
    min_dist_per_ocean_pt = distances.min(axis=1) # (N_ocean,)

    model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=128, latent_dim=128).to(device)
    ckpt_path = "vae_checkpoint.pt"
    
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded checkpoint successfully.")
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {ckpt_path}. Using untrained model. ({e})")
        # Proceed with untrained for code test if missing

    model.eval()

    # Take one sequence and evaluate uncertainty across ALL ocean points for a future T
    # Start idx = 0 in validation set
    start_idx = 0
    T_obs = 48
    
    sensor_hist = val_ds.sensor_data[start_idx : start_idx + T_obs]
    sensor_hist = torch.tensor(sensor_hist, dtype=torch.float32).unsqueeze(0).to(device) # (1, T_obs, 16)
    
    future_t_idx = start_idx + T_obs + 6 # +6 hours ahead
    t_val = val_ds.t_norm[future_t_idx]
    
    # Query all ocean points
    x_vals = val_ds.x_norm[ocean_pts[:, 1]]
    y_vals = val_ds.y_norm[ocean_pts[:, 0]]
    
    trunk = np.stack([
        x_vals, y_vals, np.full_like(x_vals, t_val)
    ], axis=-1)
    trunk = torch.tensor(trunk, dtype=torch.float32).view(1 * len(ocean_pts), 3).to(device) # (B*P, 3)
    
    with torch.no_grad():
        _, y_logvar, _, _ = model(sensor_hist, trunk)
    
    y_var = torch.exp(y_logvar).cpu().numpy().flatten() # Predictive variance
    
    # Plot Variance vs Distance
    plt.figure(figsize=(8, 6))
    plt.scatter(min_dist_per_ocean_pt, y_var, alpha=0.5, s=2, c=y_var, cmap='turbo')
    plt.xlabel("Distance to nearest sensor (Grid Units)")
    plt.ylabel("Predictive Variance ($\sigma^2$)")
    plt.title("Spatial Uncertainty Degradation\n(Demonstrating Information Blindspot)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.colorbar(label="Variance Level")
    plt.tight_layout()
    plt.savefig("uncertainty_vs_distance.png", dpi=300)
    print("Saved plot to experiment/uncertainty_vs_distance.png")

if __name__ == "__main__":
    evaluate_uncertainty()
