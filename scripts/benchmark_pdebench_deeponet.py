"""
benchmark_pdebench_deeponet.py

Skeleton Code to train and evaluate our PI-DeepONet on the PDEBench 2D Shallow Water Equations dataset.
This script sets up the exact Fair Comparison environment required for Section 3A of the paper.

Key Features:
- Designed to load standard PDEBench HDF5/NPZ formatting.
- Processes the 2D Field into Branch Net (Flattened Input) and Trunk Net (x, y, t coordinates).
- Training loop with PI-Loss (Physics-Informed PDE loss) configurable via arguments.
- Computes standard RMSE and Relative L2 Error metrics for Section 3A table.

Usage:
  python benchmark_pdebench_deeponet.py --data_path /path/to/pdebench_swe2d.h5 --epochs 50 --use_physics
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import our custom DeepONet architecture
from deeponet.model import DeepONet


class PDEBenchDataset(Dataset):
    """
    Placeholder Dataset for PDEBench 2D SWE.
    PDEBench typically provides data as HDF5 with shapes like [Num_Samples, Nt, Nx, Ny, 1] for h(t, x, y).
    """
    def __init__(self, data_path, split="train", subsample_points=1000):
        self.data_path = data_path
        self.subsample_points = subsample_points
        self.split = split
        
        print(f"Loading PDEBench 2D SWE data from {data_path} ({split} split)...")
        # TODO: Implement actual h5py or npz loading here depending on the Exact PDEBench download format.
        # Example pseudo-code for loading PDEBench:
        # with h5py.File(data_path, 'r') as f:
        #     self.h_data = f['tensor'][:]  # Expected shape (N_samples, T, X, Y)
        #     self.grid_x = f['x-coordinate'][:]
        #     self.grid_y = f['y-coordinate'][:]
        #     self.grid_t = f['t-coordinate'][:]
        
        # MOCK DATA FOR SKELETON (Remove when actual data is downloaded)
        self.N_samples = 100 if split == "train" else 20
        self.nx, self.ny, self.nt = 64, 64, 50 
        self.h_data = np.random.randn(self.N_samples, self.nt, self.nx, self.ny).astype(np.float32)
        
        # In this benchmark, the Branch Net input is usually the Initial Condition (IC) at t=0
        # or the boundary conditions. For PDEBench SWE, IC is standard.
        # Branch Input shape: Flattened (Nx * Ny)
        self.branch_dim = self.nx * self.ny
        
    def __len__(self):
        return self.N_samples

    def __getitem__(self, idx):
        # 1. Branch Input (Initial condition at t=0)
        u0 = self.h_data[idx, 0, :, :] # Shape: (Nx, Ny)
        branch_x = u0.flatten()        # Shape: (Nx * Ny,)
        
        # 2. Trunk Input (x, y, t) and Labels (eta)
        # To avoid OOM, randomly subsample continuous points across the grid for each epoch
        t_idxs = np.random.randint(0, self.nt, self.subsample_points)
        x_idxs = np.random.randint(0, self.nx, self.subsample_points)
        y_idxs = np.random.randint(0, self.ny, self.subsample_points)
        
        # Normalize coordinates between -1 and 1
        x_norm = (x_idxs / (self.nx - 1)) * 2 - 1
        y_norm = (y_idxs / (self.ny - 1)) * 2 - 1
        t_norm = (t_idxs / (self.nt - 1)) * 2 - 1
        
        trunk_x = np.stack([x_norm, y_norm, t_norm], axis=-1).astype(np.float32) # (subsample, 3)
        
        # Labels
        labels = self.h_data[idx, t_idxs, x_idxs, y_idxs] # (subsample,)
        
        return branch_x, trunk_x, labels


def calc_physics_loss(eta, trunk_x, c_wave=1.0):
    """
    Computes the 2D Wave Equation residual constraint using Automatic Differentiation.
    PDE: eta_tt - c^2 * (eta_xx + eta_yy) = 0
    """
    # Create computation graph for higher-order derivatives
    deta = torch.autograd.grad(eta, trunk_x, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    
    eta_x = deta[:, 0]
    eta_y = deta[:, 1]
    eta_t = deta[:, 2]
    
    eta_xx = torch.autograd.grad(eta_x, trunk_x, grad_outputs=torch.ones_like(eta_x), create_graph=True)[0][:, 0]
    eta_yy = torch.autograd.grad(eta_y, trunk_x, grad_outputs=torch.ones_like(eta_y), create_graph=True)[0][:, 1]
    eta_tt = torch.autograd.grad(eta_t, trunk_x, grad_outputs=torch.ones_like(eta_t), create_graph=True)[0][:, 2]
    
    residual = eta_tt - (c_wave**2) * (eta_xx + eta_yy)
    return torch.mean(residual**2)


def main():
    parser = argparse.ArgumentParser(description="PDEBench DeepONet Skeleton")
    parser.add_argument("--data_path", type=str, default="./data/pdebench_swe2d_dummy.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_physics", action="store_true", help="Enable PI-DeepONet PDE residuals")
    parser.add_argument("--lambda_phys", type=float, default=0.1, help="Physics loss weight")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset & DataLoader
    train_ds = PDEBenchDataset(args.data_path, split="train", subsample_points=1000)
    val_ds   = PDEBenchDataset(args.data_path, split="val", subsample_points=1000)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize DeepONet 
    # branch_in dimensions must match the flattened IC grid (Nx * Ny)
    model = DeepONet(
        branch_in=train_ds.branch_dim, 
        trunk_in=3, 
        width=256, 
        depth=4, 
        latent_dim=128, 
        activation="gelu"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"Starting Training! (use_physics={args.use_physics})")
    
    # 3. Training Loop
    start_time = time.time()
    best_val_l2 = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse, train_phys = 0.0, 0.0
        
        for b_b, t_b, y_b in train_loader:
            b_b = b_b.to(device)
            t_b = t_b.to(device)
            y_b = y_b.contiguous().view(-1, 1).to(device)
            
            # Require grad on Trunk input to compute physics residuals
            if args.use_physics:
                t_b.requires_grad_(True)
            
            # Expand branch to match trunk subsampling batch shape (B, subsample, branch_dim)
            B, P = t_b.shape[0], t_b.shape[1]
            b_exp = b_b.unsqueeze(1).expand(-1, P, -1).reshape(B*P, -1)
            t_exp = t_b.reshape(B*P, 3)
            
            # Forward pass
            y_pred = model(b_exp, t_exp)
            
            # Data Loss
            loss_data = criterion(y_pred, y_b)
            
            # Physics Loss
            loss_phys = torch.tensor(0.0).to(device)
            if args.use_physics:
                loss_phys = calc_physics_loss(y_pred, t_exp)
                
            loss = loss_data + args.lambda_phys * loss_phys
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_mse += loss_data.item()
            if args.use_physics:
                train_phys += loss_phys.item()

        # Validation Loop (Computing RelL2 exactly like the Paper requires)
        model.eval()
        val_rmse, val_rel_l2 = 0.0, 0.0
        with torch.no_grad():
            for b_b, t_b, y_b in val_loader:
                b_b = b_b.to(device)
                t_b = t_b.to(device)
                y_b = y_b.contiguous().view(-1, 1).to(device)
                
                B, P = t_b.shape[0], t_b.shape[1]
                b_exp = b_b.unsqueeze(1).expand(-1, P, -1).reshape(B*P, -1)
                t_exp = t_b.reshape(B*P, 3)
                
                y_pred = model(b_exp, t_exp)
                
                # Rel L2 Error calculation
                err_l2 = torch.norm(y_pred - y_b, p=2)
                norm_l2 = torch.norm(y_b, p=2)
                
                val_rmse += criterion(y_pred, y_b).item()
                val_rel_l2 += (err_l2 / norm_l2).item()
                
        val_rmse /= len(val_loader)
        val_rel_l2 /= len(val_loader)
        
        print(f"Epoch {epoch:03d} | Train MSE: {train_mse/len(train_loader):.4f} | "
              f"Phys: {train_phys/len(train_loader):.4f} || "
              f"Val RMSE: {val_rmse:.4f} | Val RelL2: {val_rel_l2*100:.2f}%")
              
        if val_rel_l2 < best_val_l2:
            best_val_l2 = val_rel_l2
            torch.save(model.state_dict(), "benchmark_pdebench_deeponet_best.pt")

    total_time = time.time() - start_time
    print(f"\n--- Benchmark Complete ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Val Relative L2 Error: {best_val_l2*100:.2f}%")
    print(f"Saved best model to 'benchmark_pdebench_deeponet_best.pt'.")
    print("Compare this RelL2 and Time against the FNO baseline in the PDEBench Paper!")

if __name__ == "__main__":
    main()
