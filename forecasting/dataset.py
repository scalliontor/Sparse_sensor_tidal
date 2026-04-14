"""
PDEBench 2D SWE — Causal Forecasting Dataset.

Each sample:
  - Branch: sensor readings at boundary, t = 0..T_obs  shape (T_obs, n_sensors)
  - Trunk:  future query points (x, y, t) with t > T_obs   shape (P, 3)
  - Label:  eta at those future query points               shape (P,)

T_obs is sampled randomly per __getitem__ so the model learns
to forecast from any observation horizon.
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


def make_boundary_sensors(Nx: int, Ny: int, n_sensors: int = 16):
    """16 sensors evenly placed on 4 edges (4 per edge)."""
    per = max(n_sensors // 4, 1)
    sx, sy = [], []
    for j in np.linspace(0, Ny - 1, per, dtype=int):
        sx.append(0);      sy.append(j)       # top edge
    for j in np.linspace(0, Ny - 1, per, dtype=int):
        sx.append(Nx - 1); sy.append(j)       # bottom edge
    for i in np.linspace(0, Nx - 1, per + 2, dtype=int)[1:-1]:
        sx.append(i); sy.append(0)            # left edge
    for i in np.linspace(0, Nx - 1, per + 2, dtype=int)[1:-1]:
        sx.append(i); sy.append(Ny - 1)      # right edge
    return np.array(sx, dtype=int), np.array(sy, dtype=int)


def load_pdebench(path: str, n_samples: int = None):
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())[:n_samples]
        g0 = f[keys[0]]
        t_c = np.array(g0["grid/t"], dtype=np.float32)
        x_c = np.array(g0["grid/x"], dtype=np.float32)
        y_c = np.array(g0["grid/y"], dtype=np.float32)
        T, Nx, Ny = len(t_c), len(x_c), len(y_c)
        data = np.empty((len(keys), T, Nx, Ny), dtype=np.float32)
        for i, k in enumerate(keys):
            data[i] = f[k]["data"][:, :, :, 0]
    return data, t_c, x_c, y_c


class PDEBenchForecastDataset(Dataset):
    """
    Causal forecasting: observe t in [0, T_obs), predict t in [T_obs, T).
    T_obs is sampled randomly in [T_obs_min, T_obs_max] each call.
    """
    def __init__(self, data: np.ndarray, t_coord, x_coord, y_coord,
                 n_sensors: int = 16, pts_per_sample: int = 256,
                 T_obs_min: int = 10, T_obs_max: int = 80,
                 seed: int = None):
        self.data = data          # (N, T, Nx, Ny)
        self.N, self.T, self.Nx, self.Ny = data.shape
        self.pts = pts_per_sample
        self.T_obs_min = T_obs_min
        self.T_obs_max = min(T_obs_max, self.T - 2)

        # Normalized coordinates
        def norm(c): return (2.0*(c - c[0])/(c[-1] - c[0] + 1e-12) - 1.0).astype(np.float32)
        self.t_norm = norm(t_coord)
        self.x_norm = norm(x_coord)
        self.y_norm = norm(y_coord)

        # Sensor positions
        self.sx, self.sy = make_boundary_sensors(self.Nx, self.Ny, n_sensors)
        self.n_sensors = len(self.sx)

        self.rng = np.random.default_rng(seed)
        print(f"  Dataset: N={self.N}, T={self.T}, grid={self.Nx}x{self.Ny}")
        print(f"  Sensors: {self.n_sensors} boundary, T_obs in [{T_obs_min}, {self.T_obs_max}]")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        field = self.data[idx]  # (T, Nx, Ny)

        # Sample T_obs
        T_obs = self.rng.integers(self.T_obs_min, self.T_obs_max + 1)

        # Branch: sensor readings at t = 0..T_obs  → (T_obs, n_sensors)
        sensor_hist = field[:T_obs, self.sx, self.sy].astype(np.float32)  # (T_obs, n_sensors)

        # Sample future query points: t > T_obs
        t_idx = self.rng.integers(T_obs, self.T, size=self.pts)
        x_idx = self.rng.integers(0, self.Nx, size=self.pts)
        y_idx = self.rng.integers(0, self.Ny, size=self.pts)

        trunk = np.stack([
            self.x_norm[x_idx],
            self.y_norm[y_idx],
            self.t_norm[t_idx],
        ], axis=-1).astype(np.float32)  # (P, 3)

        labels = field[t_idx, x_idx, y_idx].astype(np.float32)  # (P,)

        return sensor_hist, T_obs, trunk, labels
