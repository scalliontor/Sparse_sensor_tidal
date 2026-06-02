"""
Dataset for OVCNO with configurable sensor layouts (real/equispaced/random).
Loads sensor positions from a JSON layout file rather than random selection.
"""
import numpy as np
import xarray as xr
import torch
import json
from torch.utils.data import Dataset

class OVCNOLayoutDataset(Dataset):
    """
    OVCNO dataset that reads sensor positions from a JSON layout file.
    Supports: real tide-gauge positions, equispaced, random layouts.
    
    Layout JSON format:
        {"K": 12, "stations": [{"i": 50, "j": 16, "name": "Hon Dau", ...}, ...]}
    """
    def __init__(self, nc_path: str, layout_path: str, pts_per_sample: int = 512,
                 T_obs: int = 24, seed: int = 42, split: str = "train",
                 sensor_dropout: float = 0.0, train_mean: "np.ndarray | None" = None):
        """
        Args:
            nc_path: Path to the Copernicus SSH NetCDF file.
            layout_path: Path to JSON file with sensor layout.
            pts_per_sample: Number of query points per sample.
            T_obs: Fixed observation history length (timesteps).
            seed: Random seed for reproducibility.
            split: "train" or "val".
            sensor_dropout: Fraction of sensors to randomly drop at each sample (0.0 = no dropout).
            train_mean: Pre-computed train-set temporal mean field (Ny, Nx).
                        If None and split=="train", computed from train data and stored.
                        If None and split!="train", raises ValueError.
        """
        ds = xr.open_dataset(nc_path)
        ssh_full = ds['sea_surface_height'].values[:, 0, :, :]
        
        split_idx = 600
        if split == "train":
            ssh = ssh_full[:split_idx]
        else:
            ssh = ssh_full[split_idx:]
        
        self.T, self.Ny, self.Nx = ssh.shape
        self.pts = pts_per_sample
        self.T_obs = min(T_obs, self.T - 24)
        self.sensor_dropout = sensor_dropout
        
        # Normalize: subtract train-set temporal mean ONLY
        if train_mean is not None:
            self.train_mean = train_mean
        elif split == "train":
            self.train_mean = np.nanmean(ssh, axis=0)  # (Ny, Nx)
        else:
            raise ValueError(
                "val/test split requires train_mean to be passed explicitly "
                "to avoid normalization leakage. Compute it from the train split first."
            )
        ssh = ssh - self.train_mean[np.newaxis, :, :]
        self.data = ssh
        
        self.ocean_mask = ~np.isnan(ssh[0])
        self.ocean_coords = np.argwhere(self.ocean_mask)  # (N_ocean, 2) [y, x]
        
        # Load sensor layout from JSON
        with open(layout_path) as f:
            layout = json.load(f)
        
        stations = layout['stations']
        # Filter to only kept stations
        if 'status' in stations[0]:
            stations = [s for s in stations if s['status'].startswith('keep')]
        
        self.K = len(stations)
        self.station_names = [s.get('name', f'S{idx}') for idx, s in enumerate(stations)]
        self.sensor_grid_ij = np.array([(s['i'], s['j']) for s in stations])  # (K, 2)
        
        # Verify all sensors are on ocean
        for idx, (i, j) in enumerate(self.sensor_grid_ij):
            assert self.ocean_mask[i, j], f"Station {self.station_names[idx]} at grid[{i},{j}] is on land!"
        
        # Precompute normalized coordinates
        self.x_norm = np.linspace(-1, 1, self.Nx, dtype=np.float32)
        self.y_norm = np.linspace(-1, 1, self.Ny, dtype=np.float32)
        self.t_norm = np.linspace(-1, 1, self.T, dtype=np.float32)
        
        # Precompute sensor normalized positions (K, 2) -> [x_norm, y_norm]
        self.sensor_pts_norm = np.zeros((self.K, 2), dtype=np.float32)
        for idx, (i, j) in enumerate(self.sensor_grid_ij):
            self.sensor_pts_norm[idx, 0] = self.x_norm[j]
            self.sensor_pts_norm[idx, 1] = self.y_norm[i]
        
        self.rng = np.random.default_rng(seed)
        
        # Full-grid mode: use all ocean points deterministically
        self.full_grid = (pts_per_sample == -1)
        if self.full_grid:
            self.pts = len(self.ocean_coords)
            print(f"  Full-grid eval mode: {self.pts} ocean points per sample")
        
        layout_name = layout.get('layout_name', 'unknown')
        print(f"OVCNO {split}: layout={layout_name}, K={self.K}, T={self.T}, "
              f"Grid={self.Ny}x{self.Nx}, dropout={sensor_dropout}")

    def __len__(self):
        return self.T - self.T_obs - 1

    def __getitem__(self, start_idx):
        # Determine active sensors (apply dropout if enabled)
        if self.sensor_dropout > 0 and self.training_mode:
            n_keep = max(4, int(self.K * (1 - self.sensor_dropout)))
            active_idx = self.rng.choice(self.K, n_keep, replace=False)
            active_idx = np.sort(active_idx)
        else:
            active_idx = np.arange(self.K)
        
        k = len(active_idx)
        
        # Extract sensor histories
        sensor_hist = np.zeros((self.T_obs, k), dtype=np.float32)
        sensor_pts = np.zeros((k, 2), dtype=np.float32)
        
        for local_i, global_i in enumerate(active_idx):
            gi, gj = self.sensor_grid_ij[global_i]
            sensor_hist[:, local_i] = self.data[start_idx : start_idx + self.T_obs, gi, gj]
            sensor_pts[local_i] = self.sensor_pts_norm[global_i]
        
        # Sample query points in the future
        future_start = start_idx + self.T_obs
        future_end = min(future_start + 24, self.T)
        
        if self.full_grid:
            # Deterministic: all ocean points, single random future time
            t_val = self.rng.integers(future_start, future_end)
            t_idx = np.full(self.pts, t_val, dtype=np.intp)
            y_idx = self.ocean_coords[:, 0]
            x_idx = self.ocean_coords[:, 1]
        else:
            t_idx = self.rng.integers(future_start, future_end, size=self.pts)
            pt_indices = self.rng.integers(0, len(self.ocean_coords), size=self.pts)
            y_idx = self.ocean_coords[pt_indices, 0]
            x_idx = self.ocean_coords[pt_indices, 1]
        
        # Query coordinates
        q_x = self.x_norm[x_idx]
        q_y = self.y_norm[y_idx]
        
        # Distance to nearest active sensor
        dx = q_x[:, None] - sensor_pts[None, :, 0]
        dy = q_y[:, None] - sensor_pts[None, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        trunk = np.stack([q_x, q_y, self.t_norm[t_idx], min_dists], axis=-1).astype(np.float32)
        labels = self.data[t_idx, y_idx, x_idx].astype(np.float32)
        
        return (torch.tensor(sensor_hist, dtype=torch.float32),
                torch.tensor(sensor_pts, dtype=torch.float32),
                torch.tensor(trunk, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32))
    
    @property
    def training_mode(self):
        """Check if in training mode for dropout."""
        return self.sensor_dropout > 0
