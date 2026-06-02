import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

class CopernicusVAEDataset(Dataset):
    """
    Causal Information Bottleneck forecasting on Copernicus SSH.
    Extracts sensor histories and future spatial queries from ocean (non-NaN) coordinates.
    """
    def __init__(self, nc_path: str, n_sensors: int = 16, pts_per_sample: int = 512,
                 T_obs_min: int = 24, T_obs_max: int = 72, seed: int = 42,
                 split: str = "train"):
        ds = xr.open_dataset(nc_path)
        # Shape: (T, Lat, Lon) -> Remove depth
        ssh = ds['sea_surface_height'].values[:, 0, :, :]
        
        # Temporal split (744 hours total ~ 31 days)
        # Train: first 25 days (600 hours). Val: remaining 6 days (144 hours)
        split_idx = 600
        if split == "train":
            ssh = ssh[:split_idx]
        else:
            ssh = ssh[split_idx:]
            
        self.T, self.Ny, self.Nx = ssh.shape
        self.pts = pts_per_sample
        self.T_obs_min = T_obs_min
        self.T_obs_max = min(T_obs_max, self.T - 24)
        
        # Subtract temporal mean (anomaly prediction)
        mean_field = np.nanmean(ssh, axis=0, keepdims=True)
        ssh = ssh - mean_field
        
        self.data = ssh
        
        # Identify ocean mask
        self.ocean_mask = ~np.isnan(ssh[0])
        self.ocean_coords = np.argwhere(self.ocean_mask)  # (N_ocean, 2)
        
        # Define 16 boundary sensors exactly as specified (South and East boundary)
        # We find the furthest South ocean points and furthest East ocean points
        sx, sy = [], []
        # Let's just pick 16 random ocean coordinates for simplicity and robustness in this experiment
        # OR deterministic boundary sensors:
        np.random.seed(42)
        sensor_indices = np.random.choice(len(self.ocean_coords), n_sensors, replace=False)
        self.sensor_coords = self.ocean_coords[sensor_indices]
        
        # Pre-extract sensor entire history
        # (T, n_sensors)
        self.sensor_data = np.zeros((self.T, n_sensors), dtype=np.float32)
        for i, (y, x) in enumerate(self.sensor_coords):
            self.sensor_data[:, i] = self.data[:, y, x]
            
        # Normalize coordinates
        self.x_norm = np.linspace(-1, 1, self.Nx, dtype=np.float32)
        self.y_norm = np.linspace(-1, 1, self.Ny, dtype=np.float32)
        self.t_norm = np.linspace(-1, 1, self.T, dtype=np.float32)
        
        self.rng = np.random.default_rng(seed)
        print(f"Copernicus {split}: T={self.T}, Grid={self.Ny}x{self.Nx}, Ocean Pts={len(self.ocean_coords)}")

    def __len__(self):
        # Treat each trajectory as a starting point. We can start a trajectory anywhere.
        return self.T - self.T_obs_max - 1

    def __getitem__(self, start_idx):
        T_obs = self.rng.integers(self.T_obs_min, self.T_obs_max + 1)
        
        # History string
        sensor_hist = self.sensor_data[start_idx : start_idx + T_obs].copy()
        
        # Sample points in the future horizon (up to 24 hours ahead)
        future_start = start_idx + T_obs
        future_end = min(future_start + 24, self.T)
        
        t_idx = self.rng.integers(future_start, future_end, size=self.pts)
        
        # Sample random ocean coordinates
        pt_indices = self.rng.integers(0, len(self.ocean_coords), size=self.pts)
        y_idx = self.ocean_coords[pt_indices, 0]
        x_idx = self.ocean_coords[pt_indices, 1]
        
        trunk = np.stack([
            self.x_norm[x_idx],
            self.y_norm[y_idx],
            self.t_norm[t_idx]
        ], axis=-1).astype(np.float32)
        
        labels = self.data[t_idx, y_idx, x_idx].astype(np.float32)
        
        return torch.tensor(sensor_hist, dtype=torch.float32), \
               torch.tensor(trunk, dtype=torch.float32), \
               torch.tensor(labels, dtype=torch.float32)
