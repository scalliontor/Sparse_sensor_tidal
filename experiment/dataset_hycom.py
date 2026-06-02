"""
HYCOM OVCNO Dataset — adapted from CopernicusOVCNODataset.
Key differences:
- Variable name: surf_el (not sea_surface_height)
- No depth dimension
- 3-hourly temporal resolution (8 steps = 24h history)
- Chronological train/val/test split for multi-month data
- Station coordinates from hycom_real_k12_stations.json
"""
import numpy as np
import xarray as xr
import torch
import json
from torch.utils.data import Dataset


class HYCOMOVCNODataset(Dataset):
    """
    HYCOM cross-product validation dataset for OVCNO.
    Uses real station coordinates snapped to HYCOM grid.
    3-hourly temporal resolution: 8 steps = 24h history.
    """
    def __init__(self, nc_path: str, station_json: str,
                 pts_per_sample: int = 512,
                 T_obs: int = 8,  # 8 steps × 3h = 24h history
                 seed: int = 42,
                 split: str = "train",
                 variable_sensors: bool = False,
                 train_mean: "np.ndarray | None" = None,
                 train_months: int = 3):
        """
        Args:
            nc_path: Path to HYCOM NetCDF (time, latitude, longitude)
            station_json: Path to hycom_real_k12_stations.json
            pts_per_sample: Number of query points per sample
            T_obs: Number of input history steps (8 = 24h at 3-hourly)
            seed: Random seed
            split: "train", "val", or "test"
            variable_sensors: Whether to randomly subsample sensors
            train_mean: Precomputed train mean (required for val/test)
            train_months: Number of months for training (default 3 = Jan-Mar)
        """
        ds = xr.open_dataset(nc_path)
        ssh_full = ds['surf_el'].values  # (T, Nlat, Nlon), no depth dimension

        # Chronological split based on data length
        # Detect if this is 5-month (Jan-May) or 9-month (Jan-Sep) data
        T_total = ssh_full.shape[0]
        
        if T_total > 1500:
            # Extended 9-month data: ~1984 timesteps
            # Jan-Jun train (~1464), Jul val (~248), Aug-Sep test (~272)
            # steps_per_month ≈ [244, 232, 248, 240, 248, 240, 248, 248, 36+]
            # Use ratio-based split for robustness
            train_end = int(T_total * 0.74)   # ~6/8.1 months
            val_end = int(T_total * 0.87)     # ~7/8.1 months
            split_desc = f"extended 9-month (train:{train_end}, val:{val_end-train_end}, test:{T_total-val_end})"
        else:
            # Original 5-month data: Jan-May ~1212 timesteps
            # Jan-Mar train (724), Apr val (240), May test (248)
            train_end = 724
            val_end = 964
            split_desc = f"5-month (train:{train_end}, val:{val_end-train_end}, test:{T_total-val_end})"
        
        if split == "train":
            ssh = ssh_full[:train_end]
        elif split == "val":
            ssh = ssh_full[train_end:val_end]
        elif split == "test":
            ssh = ssh_full[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"  Split config: {split_desc}")
        
        self.T, self.Ny, self.Nx = ssh.shape
        self.pts = pts_per_sample
        self.T_obs = T_obs
        self.variable_sensors = variable_sensors

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

        # Load station coordinates from JSON
        with open(station_json) as f:
            sdata = json.load(f)
        stations = [s for s in sdata['stations'] if s.get('valid_ocean', False)]
        
        self.n_stations = len(stations)
        self.max_sensors = self.n_stations
        self.station_coords = np.array([[s['i'], s['j']] for s in stations])  # (K, 2) [lat_idx, lon_idx]
        self.station_names = [s['name'] for s in stations]

        # Verify all station cells are ocean
        for idx, (i, j) in enumerate(self.station_coords):
            if not self.ocean_mask[i, j]:
                print(f"WARNING: Station {self.station_names[idx]} at ({i},{j}) is NOT ocean!")

        # Normalized coordinate arrays
        self.x_norm = np.linspace(-1, 1, self.Nx, dtype=np.float32)
        self.y_norm = np.linspace(-1, 1, self.Ny, dtype=np.float32)
        self.t_norm = np.linspace(-1, 1, self.T, dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        
        # Max lead time in steps (24h / 3h = 8 steps)
        self.max_lead_steps = 8
        
        print(f"OVCNO HYCOM {split}: T={self.T}, Grid={self.Ny}x{self.Nx}, "
              f"Stations={self.n_stations}, Variable={variable_sensors}")

    def __len__(self):
        return self.T - self.T_obs - self.max_lead_steps

    def __getitem__(self, start_idx):
        # Use fixed station set (or variable if enabled)
        if self.variable_sensors:
            k = self.rng.integers(8, self.max_sensors + 1)
            active_idx = self.rng.choice(self.n_stations, k, replace=False)
        else:
            k = self.max_sensors
            active_idx = np.arange(self.n_stations)

        active_coords = self.station_coords[active_idx]

        # Sensor history: (T_obs, K)
        sensor_hist = np.zeros((self.T_obs, k), dtype=np.float32)
        sensor_pts = np.zeros((k, 2), dtype=np.float32)

        for i, (yi, xi) in enumerate(active_coords):
            sensor_hist[:, i] = self.data[start_idx: start_idx + self.T_obs, yi, xi]
            sensor_pts[i, 0] = self.x_norm[xi]
            sensor_pts[i, 1] = self.y_norm[yi]

        # Future query points
        future_start = start_idx + self.T_obs
        future_end = min(future_start + self.max_lead_steps, self.T)
        t_idx = self.rng.integers(future_start, future_end, size=self.pts)

        pt_indices = self.rng.integers(0, len(self.ocean_coords), size=self.pts)
        y_idx = self.ocean_coords[pt_indices, 0]
        x_idx = self.ocean_coords[pt_indices, 1]

        # Calculate distance d_s(x,y) from query to nearest active sensor
        q_x = self.x_norm[x_idx]
        q_y = self.y_norm[y_idx]

        dx = q_x[:, None] - sensor_pts[None, :, 0]
        dy = q_y[:, None] - sensor_pts[None, :, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)  # (P, K)
        min_dists = np.min(dists, axis=1)  # (P,)

        trunk = np.stack([
            q_x, q_y, self.t_norm[t_idx], min_dists
        ], axis=-1).astype(np.float32)

        labels = self.data[t_idx, y_idx, x_idx].astype(np.float32)

        return (torch.tensor(sensor_hist, dtype=torch.float32),
                torch.tensor(sensor_pts, dtype=torch.float32),
                torch.tensor(trunk, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32))
