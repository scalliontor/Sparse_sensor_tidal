import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import xarray as xr
from torch.utils.data import DataLoader
import sys

# Import dataset and model from the remote environment
sys.path.append('.') # Assuming run from experiment folder
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO

def load_layout(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('stations', data.get('sensors'))

def plot_forecast_trajectory():
    # 1. Load data and setup dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    layout_path = "sensors_real_stations.json"
    checkpoint_path = "ovcno_checkpoint.pt"

    # For OVCNO, we might not need layout_path directly in the dataset if we pass variable_sensors=False.
    # We will just load layout manually later.
    ds_train = CopernicusOVCNODataset(nc_path, n_sensors=12, pts_per_sample=10, split="train")
    
    ds = CopernicusOVCNODataset(nc_path, n_sensors=12, pts_per_sample=-1, split="val",
                                 train_mean=ds_train.train_mean)
    # We want ALL ocean points, or just specific points. 
    # Let's extract specific 3 points from the full grid.
    mask = ds.data[0]
    if hasattr(mask, 'values'): mask = mask.values
    while mask.ndim > 2: mask = mask[0]
    mask = ~np.isnan(mask)
    Ny, Nx = mask.shape
    
    # 2. Select 3 representative points
    real_sensors = [s for s in load_layout(layout_path) if s.get('i') is not None]
    
    # a. High observability (near sensor)
    # Let's pick a point right next to the first sensor
    s1 = real_sensors[0]
    p1_j, p1_i = s1['j'], s1['i']
    # If exactly on sensor, it might be trivial or perfect, let's pick it or slightly off (offset by 1)
    # Let's pick exactly the sensor location if it's ocean, which it is.
    
    # b. Mid-domain
    p2_j, p2_i = 30, 30
    if not mask[p2_i, p2_j]:
        p2_j, p2_i = 35, 30
        
    # c. Far / Low observability (bottom right, far from coast)
    p3_j, p3_i = 50, 5 # j is x, i is y
    if not mask[p3_i, p3_j]:
        p3_j, p3_i = 45, 10
        
    points = [(p1_j, p1_i, "Near Station"), 
              (p2_j, p2_i, "Mid-Domain"), 
              (p3_j, p3_i, "Weakly Observed Interior")]
    
    # 3. Setup Model
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 4. Pick a specific time slice in validation set
    # ds.val uses the later part of the dataset. 
    # Let's pick a sequence in the validation set.
    val_T = ds.data.shape[0]
    start_idx = 10 # 10 hours into the validation set
    T_obs = 24
    lead_times = np.arange(1, 25) # 1 to 24 hours
    
    # We will build the sensor history manually for this start_idx
    k = 12
    sensor_hist = np.zeros((T_obs, k), dtype=np.float32)
    sensor_pts = np.zeros((k, 2), dtype=np.float32)
    
    for idx_s, s in enumerate(real_sensors):
        # We need to make sure we index correctly into ds.data
        hist_slice = ds.data[start_idx : start_idx + T_obs]
        if hasattr(hist_slice, 'values'): hist_slice = hist_slice.values
        while hist_slice.ndim > 3:
            hist_slice = hist_slice[:, 0]
        sensor_hist[:, idx_s] = hist_slice[:, s['i'], s['j']]
        sensor_pts[idx_s, 0] = ds.x_norm[s['j']]
        sensor_pts[idx_s, 1] = ds.y_norm[s['i']]
        
    sensor_hist_t = torch.tensor(sensor_hist, dtype=torch.float32).unsqueeze(0).to(device)
    sensor_pts_t = torch.tensor(sensor_pts, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Prepare plots
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 5. Evaluate for each point over 24 hours
    for idx_p, (px, py, title) in enumerate(points):
        truths = []
        means = []
        stds = []
        
        for t_lead in lead_times:
            eval_idx = start_idx + T_obs + t_lead - 1
            if eval_idx >= val_T:
                break
                
            eval_slice = ds.data[eval_idx]
            if hasattr(eval_slice, 'values'): eval_slice = eval_slice.values
            while eval_slice.ndim > 2:
                eval_slice = eval_slice[0]
                
            y_true = eval_slice[py, px]
            y_mean = ds_train.train_mean[py, px]
            
            # Predict
            trunk = np.array([
                ds.x_norm[px],
                ds.y_norm[py],
                ds.t_norm[eval_idx],
                0.0 # dummy distance for plotting ? Wait, the trunk requires distance!
            ]).astype(np.float32)
            
            # Recompute real distance
            # For OVCNO, distance is to nearest sensor
            dists = np.sqrt((ds.x_norm[px] - sensor_pts[:, 0])**2 + (ds.y_norm[py] - sensor_pts[:, 1])**2)
            trunk[3] = np.min(dists)
            
            trunk_t = torch.tensor(trunk).view(1, 4).to(device)
            
            with torch.no_grad():
                # Note: plot_observability_maps used sample_z=False to plot the clean mean
                mu, logvar, _, _, _ = model(sensor_hist_t, sensor_pts_t, trunk_t, sample_z=False)
                sigma = torch.exp(0.5 * logvar)
            
            mu_val = mu.item() * 1.0 # Values are already anomaly scaled
            sigma_val = sigma.item() * 1.0 # Standard deviation scale
            
            truths.append(y_true)
            means.append(mu_val)
            stds.append(sigma_val)
            
        ax = axes[idx_p]
        x_axis = np.arange(1, len(truths) + 1)
        
        ax.plot(x_axis, truths, 'k-o', label='Ground Truth', markersize=4, linewidth=2)
        ax.plot(x_axis, means, color=colors[0], marker='s', label='OVCNO Predictive Mean', markersize=4, linewidth=2)
        
        means = np.array(means)
        stds = np.array(stds)
        ax.fill_between(x_axis, means - 1.96*stds, means + 1.96*stds, color=colors[0], alpha=0.2, label='95% Interval')
        
        ax.set_title(f"({chr(97+idx_p)}) {title}", loc='left', fontweight='bold', fontsize=12)
        ax.set_ylabel("SSH Anomaly (m)", fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        
        if idx_p == 0:
            ax.legend(loc='upper right', fontsize=10)
            
    axes[2].set_xlabel("Forecast Horizon / Lead Time (Hours)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("forecast_trajectories.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved forecast_trajectories.png")

if __name__ == "__main__":
    plot_forecast_trajectory()
