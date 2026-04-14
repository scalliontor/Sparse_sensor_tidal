#!/usr/bin/env python3
"""
Build dataset for 2D DeepONet from raw simulation npz files.

Raw sim schema (per window i):
- t: (T,)                (T=168 recommended)
- h: (T, Ny, Nx)         total water depth (or your field)
- ssh_input: (168, 10)   boundary forcing (time x sensors)

We will build dataset samples:
(branch_flat, trunk_xyz) -> label_eta

Where:
- branch_flat: flatten(ssh_input) -> (1680,)
- trunk_xyz: normalized (x, y, t) -> (3,)
- label_eta: h - mean_t(h) at the queried (x,y,t)
- window_id: integer window index (0..N-1)

IMPORTANT:
- split train/test MUST be by window_id, not by random sample.
"""

from __future__ import annotations
import argparse
import glob
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GridSpec:
    nx: int
    ny: int
    # normalized coordinates in [-1, 1]
    x_norm: np.ndarray  # (nx,)
    y_norm: np.ndarray  # (ny,)

    @staticmethod
    def from_shape(ny: int, nx: int) -> "GridSpec":
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
        return GridSpec(nx=nx, ny=ny, x_norm=x, y_norm=y)


def normalize_time_index(t_idx: np.ndarray, T: int) -> np.ndarray:
    """Map integer indices [0..T-1] -> [-1,1]."""
    if T <= 1:
        return np.zeros_like(t_idx, dtype=np.float32)
    return (2.0 * (t_idx.astype(np.float32) / (T - 1)) - 1.0).astype(np.float32)


def compute_eta_from_h(h: np.ndarray) -> np.ndarray:
    """
    Since out_h from our new solver is already (U[0] - base_h), it IS the anomaly.
    We just need to ensure no base offsets are present.
    """
    return h.astype(np.float32)


def zscore_fit(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mu.astype(np.float32), std.astype(np.float32)


def zscore_apply(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mu) / std).astype(np.float32)


def sample_points_per_window(
    eta: np.ndarray,  # (T, ny, nx)
    grid: GridSpec,
    n_points: int,
    rng: np.random.Generator,
    ocean_mask: np.ndarray = None,  # (ny, nx) bool, True = ocean
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample n_points points uniformly across (t, y, x).
    If ocean_mask provided, only samples ocean cells.
    Returns:
      trunk (n_points,3)  [x_norm, y_norm, t_norm]
      labels (n_points,)
      indices (n_points,3) [t_idx, y_idx, x_idx] (int)
    """
    T, ny, nx = eta.shape

    if ocean_mask is not None:
        # Get flat indices of ocean cells
        ocean_flat = np.where(ocean_mask.ravel())[0]
        flat_idx = rng.choice(ocean_flat, size=n_points, replace=True)
        y_idx = flat_idx // nx
        x_idx = flat_idx % nx
    else:
        y_idx = rng.integers(0, ny, size=(n_points,), endpoint=False)
        x_idx = rng.integers(0, nx, size=(n_points,), endpoint=False)

    t_idx = rng.integers(0, T, size=(n_points,), endpoint=False)
    t_norm = normalize_time_index(t_idx, T)
    x_norm = grid.x_norm[x_idx]
    y_norm = grid.y_norm[y_idx]

    trunk = np.stack([x_norm, y_norm, t_norm], axis=1).astype(np.float32)
    labels = eta[t_idx, y_idx, x_idx].astype(np.float32)
    indices = np.stack([t_idx, y_idx, x_idx], axis=1).astype(np.int32)
    return trunk, labels, indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims_dir", type=str, default="data/simulations_2d", help="Folder containing sim_2d_*.npz")
    ap.add_argument("--pattern", type=str, default="sim_2d_*.npz")
    ap.add_argument("--out", type=str, default="data/dataset_2d_v2.npz")
    ap.add_argument("--points_per_window", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train_windows",
        type=str,
        default="0-15",
        help="Range for training windows (inclusive), e.g. '0-15' for 16 windows.",
    )
    ap.add_argument(
        "--test_windows",
        type=str,
        default="16-19",
        help="Range for test windows (inclusive), e.g. '16-19' for 4 windows.",
    )
    args = ap.parse_args()

    sim_files = sorted(glob.glob(os.path.join(args.sims_dir, args.pattern)))
    if not sim_files:
        raise FileNotFoundError(f"No sim files found at {args.sims_dir}/{args.pattern}")

    rng = np.random.default_rng(args.seed)

    # Parse window ranges
    def parse_range(s: str) -> np.ndarray:
        if ',' in s:
            return np.array([int(x) for x in s.split(',')], dtype=np.int32)
        a, b = s.split('-')
        return np.arange(int(a), int(b) + 1, dtype=np.int32)

    train_w = set(parse_range(args.train_windows).tolist())
    test_w = set(parse_range(args.test_windows).tolist())

    # First pass: load all windows, build arrays (unnormalized)
    branch_list = []
    trunk_list = []
    labels_list = []
    window_id_list = []

    # We'll store per-sample split masks later by window_id
    # Load bathymetry for ocean masking
    bathymetry_path = os.path.join(os.path.dirname(args.sims_dir), "processed", "elev_grid.npy")
    ocean_mask = None
    if os.path.exists(bathymetry_path):
        bath = np.load(bathymetry_path)
        ocean_mask = bath < 0
        print(f"Ocean mask loaded: {ocean_mask.sum()} ocean cells ({100*ocean_mask.mean():.1f}%)")
    else:
        print("Warning: bathymetry not found, sampling all cells")

    for wid, f in enumerate(sim_files):
        data = np.load(f)
        h = data["h"]  # (T, ny, nx) ideally T=168
        ssh_in = data["ssh_input"]  # (168,10)

        # Validate expected shapes quickly
        # Validate expected time dimensions quickly
        if ssh_in.shape[0] != 168:
            raise ValueError(f"{os.path.basename(f)} ssh_input time shape {ssh_in.shape[0]} != 168")
        if h.ndim != 3:
            raise ValueError(f"{os.path.basename(f)} h must be 3D (T,ny,nx), got {h.shape}")

        T, ny, nx = h.shape
        if T < 10:
            raise ValueError(f"{os.path.basename(f)} T too small ({T}). Set save_every so T=168.")

        grid = GridSpec.from_shape(ny=ny, nx=nx)

        eta = compute_eta_from_h(h.astype(np.float32))  # (T,ny,nx)

        trunk, labels, _ = sample_points_per_window(
            eta=eta,
            grid=grid,
            n_points=args.points_per_window,
            rng=rng,
            ocean_mask=ocean_mask,
        )

        # Branch: replicate flattened boundary for each sampled point
        branch_flat = ssh_in.astype(np.float32).reshape(-1)  # dynamic sizing
        branch = np.repeat(branch_flat[None, :], repeats=args.points_per_window, axis=0)

        branch_list.append(branch)
        trunk_list.append(trunk)
        labels_list.append(labels)
        window_id_list.append(np.full((args.points_per_window,), wid, dtype=np.int32))

    branch_all = np.concatenate(branch_list, axis=0)  # (N,1680)
    trunk_all = np.concatenate(trunk_list, axis=0)    # (N,3)
    labels_all = np.concatenate(labels_list, axis=0)  # (N,)
    window_id_all = np.concatenate(window_id_list, axis=0)  # (N,)

    # Split by window_id
    train_mask = np.isin(window_id_all, list(train_w))
    test_mask = np.isin(window_id_all, list(test_w))
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("Train/Test masks empty. Check window ranges vs number of sim files.")

    # Fit normalization on TRAIN only (no leakage)
    branch_mu, branch_std = zscore_fit(branch_all[train_mask])
    trunk_mu, trunk_std = zscore_fit(trunk_all[train_mask])
    y_mu, y_std = zscore_fit(labels_all[train_mask, None])  # treat y as (N,1)

    branch_norm = zscore_apply(branch_all, branch_mu, branch_std)
    trunk_norm = zscore_apply(trunk_all, trunk_mu, trunk_std)
    y_norm = zscore_apply(labels_all[:, None], y_mu, y_std).reshape(-1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        branch=branch_norm.astype(np.float32),
        trunk=trunk_norm.astype(np.float32),
        labels=y_norm.astype(np.float32),
        window_id=window_id_all.astype(np.int32),
        # store normalization stats for evaluation/inference inverse-transform
        branch_mu=branch_mu,
        branch_std=branch_std,
        trunk_mu=trunk_mu,
        trunk_std=trunk_std,
        y_mu=y_mu.reshape(-1),
        y_std=y_std.reshape(-1),
        train_windows=np.array(sorted(list(train_w)), dtype=np.int32),
        test_windows=np.array(sorted(list(test_w)), dtype=np.int32),
    )

    print(f"[OK] Saved: {args.out}")
    print(f"Total samples: {branch_norm.shape[0]:,}")
    print(f"Train samples: {train_mask.sum():,} | Test samples: {test_mask.sum():,}")
    print("Note: labels are eta (h - mean_t(h)) normalized by TRAIN stats.")


if __name__ == "__main__":
    main()
