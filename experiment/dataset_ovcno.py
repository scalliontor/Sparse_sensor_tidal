import numpy as np
import xarray as xr
import torch
import math
from torch.utils.data import Dataset

class CopernicusOVCNODataset(Dataset):
    """
    Observability-Aware formatting.
    Returns sensor values AND their geometrical coordinates.
    Also returns query points with pre-computed distance to the nearest sensor (d_s).
    Supports variable sensor sets natively.
    """
    def __init__(self, nc_path: str, n_sensors: int = 16, pts_per_sample: int = 512,
                 T_obs_min: int = 24, T_obs_max: int = 72, seed: int = 42,
                 split: str = "train", variable_sensors: bool = False,
                 train_mean: "np.ndarray | None" = None):
        ds = xr.open_dataset(nc_path)
        ssh_full = ds['sea_surface_height'].values[:, 0, :, :]
        
        split_idx = 600
        if split == "train":
            ssh = ssh_full[:split_idx]
        else:
            ssh = ssh_full[split_idx:]
            
        self.T, self.Ny, self.Nx = ssh.shape
        self.pts = pts_per_sample
        self.T_obs_min = T_obs_min
        self.T_obs_max = min(T_obs_max, self.T - 24)
        self.variable_sensors = variable_sensors
        self.max_sensors = n_sensors
        
        # Normalize: subtract train-set temporal mean ONLY
        if train_mean is not None:
            self.train_mean = train_mean
        elif split == "train":
            self.train_mean = np.nanmean(ssh, axis=0)  # (Ny, Nx)
        else:
            raise ValueError(
                "val/test split requires train_mean to avoid normalization leakage."
            )
        ssh = ssh - self.train_mean[np.newaxis, :, :]
        self.data = ssh
        
        self.ocean_mask = ~np.isnan(ssh[0])
        self.ocean_coords = np.argwhere(self.ocean_mask)  # (N_ocean, 2) [y, x]
        
        # We always pick a master set of sensors. If variable_sensors is True, 
        # we subsample from this master set during __getitem__.
        np.random.seed(42)  # global fixed seed for reproducible master sensors
        sensor_indices = np.random.choice(len(self.ocean_coords), 64, replace=False) # max pool
        self.master_sensor_coords = self.ocean_coords[sensor_indices] # (64, 2)
        
        self.x_norm = np.linspace(-1, 1, self.Nx, dtype=np.float32)
        self.y_norm = np.linspace(-1, 1, self.Ny, dtype=np.float32)
        self.t_norm = np.linspace(-1, 1, self.T, dtype=np.float32)
        
        self.rng = np.random.default_rng(seed)
        print(f"OVCNO Copernicus {split}: T={self.T}, Grid={self.Ny}x{self.Nx}, Variable Sensors={variable_sensors}")

    def __len__(self):
        return self.T - self.T_obs_max - 1

    def __getitem__(self, start_idx):
        T_obs = self.rng.integers(self.T_obs_min, self.T_obs_max + 1)
        
        if self.variable_sensors:
            # 8 to max_sensors
            k = self.rng.integers(8, self.max_sensors + 1)
        else:
            k = self.max_sensors
            
        # Draw k sensors from master
        active_sensor_idx = self.rng.choice(len(self.master_sensor_coords), k, replace=False)
        active_coords = self.master_sensor_coords[active_sensor_idx]
        
        sensor_hist = np.zeros((T_obs, k), dtype=np.float32)
        sensor_pts = np.zeros((k, 2), dtype=np.float32)
        
        for i, (y, x) in enumerate(active_coords):
            sensor_hist[:, i] = self.data[start_idx : start_idx + T_obs, y, x]
            sensor_pts[i, 0] = self.x_norm[x]
            sensor_pts[i, 1] = self.y_norm[y]
            
        future_start = start_idx + T_obs
        future_end = min(future_start + 24, self.T)
        t_idx = self.rng.integers(future_start, future_end, size=self.pts)
        
        pt_indices = self.rng.integers(0, len(self.ocean_coords), size=self.pts)
        y_idx = self.ocean_coords[pt_indices, 0]
        x_idx = self.ocean_coords[pt_indices, 1]
        
        # Calculate distance d_s(x,y) from query to nearest active sensor
        q_x = self.x_norm[x_idx]
        q_y = self.y_norm[y_idx]
        
        # Broadcasting to find min distance:
        # q_x: (P, 1), sensor_pts[:, 0]: (1, K)
        dx = q_x[:, None] - sensor_pts[None, :, 0]
        dy = q_y[:, None] - sensor_pts[None, :, 1]
        dists = np.sqrt(dx**2 + dy**2) # (P, K)
        min_dists = np.min(dists, axis=1) # (P,)
        
        trunk = np.stack([
            q_x, q_y, self.t_norm[t_idx], min_dists
        ], axis=-1).astype(np.float32)
        
        labels = self.data[t_idx, y_idx, x_idx].astype(np.float32)
        
        return (torch.tensor(sensor_hist, dtype=torch.float32),
                torch.tensor(sensor_pts, dtype=torch.float32),
                torch.tensor(trunk, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32))
