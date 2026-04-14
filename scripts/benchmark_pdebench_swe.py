#!/usr/bin/env python3
"""
benchmark_pdebench_swe.py
=========================
Train & evaluate DeepONet variants on the PDEBench 2D SWE dataset.
Incorporates the "Partial Observation" contribution.

Usage:
  # Vanilla DeepONet (Full Field IC -> Future Field) [Like standard PDEBench]
  python benchmark_pdebench_swe.py --epochs 50

  # PI-DeepONet
  python benchmark_pdebench_swe.py --epochs 50 --physics

  # Partial Observation DeepONet (OUR CONTRIBUTION)
  # Uses only 16 sensors tracked over all timesteps instead of the 128x128 full initial field!
  python benchmark_pdebench_swe.py --epochs 50 --partial
"""

from __future__ import annotations
import argparse, os, sys, time, json
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from deeponet.model import DeepONet

def load_pdebench_swe(path: str, n_samples: int | None = None):
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        if n_samples is not None:
            keys = keys[:n_samples]
        N = len(keys)
        g0 = f[keys[0]]
        t_coord = np.array(g0["grid/t"], dtype=np.float32)
        x_coord = np.array(g0["grid/x"], dtype=np.float32)
        y_coord = np.array(g0["grid/y"], dtype=np.float32)
        T, Nx, Ny = len(t_coord), len(x_coord), len(y_coord)

        print(f"  Loading {N} samples, grid: T={T}, Nx={Nx}, Ny={Ny}")
        h_all = np.empty((N, T, Nx, Ny), dtype=np.float32)
        for i, k in enumerate(keys):
            h_all[i] = f[k]["data"][:, :, :, 0]
    return h_all, t_coord, x_coord, y_coord

class SWE2DDataset(Dataset):
    def __init__(self, h_field: np.ndarray, t_coord, x_coord, y_coord, 
                 pts_per_sample: int = 512, partial: bool = False, 
                 boundary: bool = False, num_sensors: int = 16):
        self.h = h_field
        self.N, self.T, self.Nx, self.Ny = self.h.shape
        self.pts = pts_per_sample
        self.partial = partial or boundary  # both use sensor time-series
        self.boundary = boundary
        
        if self.boundary:
            # --- BOUNDARY-ONLY sensors: evenly spaced on the 4 edges ---
            per_edge = max(num_sensors // 4, 1)
            sx_list, sy_list = [], []
            # Top edge (x=0, y varies)
            for j in np.linspace(0, self.Ny-1, per_edge, dtype=int):
                sx_list.append(0); sy_list.append(j)
            # Bottom edge (x=Nx-1, y varies)
            for j in np.linspace(0, self.Ny-1, per_edge, dtype=int):
                sx_list.append(self.Nx-1); sy_list.append(j)
            # Left edge (y=0, x varies) — skip corners already added
            for i in np.linspace(0, self.Nx-1, per_edge+2, dtype=int)[1:-1]:
                sx_list.append(i); sy_list.append(0)
            # Right edge (y=Ny-1, x varies) — skip corners
            for i in np.linspace(0, self.Nx-1, per_edge+2, dtype=int)[1:-1]:
                sx_list.append(i); sy_list.append(self.Ny-1)
            self.sensor_x = np.array(sx_list, dtype=int)
            self.sensor_y = np.array(sy_list, dtype=int)
            self.num_sensors = len(self.sensor_x)
            self.branch_dim = self.num_sensors * self.T
            print(f"  [BOUNDARY Mode] {self.num_sensors} sensors on 4 edges, T={self.T} -> branch_dim = {self.branch_dim}")
        elif self.partial:
            # --- INTERIOR sensors: 4x4 uniform grid ---
            step_x = self.Nx // int(np.sqrt(num_sensors))
            step_y = self.Ny // int(np.sqrt(num_sensors))
            sx = np.arange(step_x//2, self.Nx, step_x)
            sy = np.arange(step_y//2, self.Ny, step_y)
            self.sensor_x, self.sensor_y = np.meshgrid(sx, sy)
            self.sensor_x = self.sensor_x.flatten()
            self.sensor_y = self.sensor_y.flatten()
            self.num_sensors = len(self.sensor_x)
            self.branch_dim = self.num_sensors * self.T
            print(f"  [Partial Mode] {self.num_sensors} interior sensors, T={self.T} -> branch_dim = {self.branch_dim}")
        else:
            self.branch_dim = self.Nx * self.Ny
            print(f"  [Full Mode] Using Full IC {self.Nx}x{self.Ny} -> branch_dim = {self.branch_dim}")

        self.t_norm = (2.0 * (t_coord - t_coord[0]) / (t_coord[-1] - t_coord[0] + 1e-12) - 1.0).astype(np.float32)
        self.x_norm = (2.0 * (x_coord - x_coord[0]) / (x_coord[-1] - x_coord[0] + 1e-12) - 1.0).astype(np.float32)
        self.y_norm = (2.0 * (y_coord - y_coord[0]) / (y_coord[-1] - y_coord[0] + 1e-12) - 1.0).astype(np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        field = self.h[idx]
        
        if self.partial:
            # Branch: Sensor time-series across all time steps
            # Shape: (T, num_sensors)
            sensors_t = field[:, self.sensor_x, self.sensor_y]
            branch = sensors_t.flatten()
        else:
            # Branch: Full Initial Condition at t=0
            branch = field[0].flatten()

        ti = np.random.randint(1, self.T, self.pts)
        xi = np.random.randint(0, self.Nx, self.pts)
        yi = np.random.randint(0, self.Ny, self.pts)

        trunk = np.stack([
            self.x_norm[xi],
            self.y_norm[yi],
            self.t_norm[ti],
        ], axis=-1).astype(np.float32)

        labels = field[ti, xi, yi].astype(np.float32)
        return branch, trunk, labels

def physics_residual(eta: torch.Tensor, coords: torch.Tensor, g: float=1.0, H0: float=1.0):
    c2 = g * H0
    grad1 = torch.autograd.grad(eta, coords, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    eta_xx = torch.autograd.grad(grad1[:, 0], coords, grad_outputs=torch.ones_like(grad1[:,0]), create_graph=True)[0][:, 0]
    eta_yy = torch.autograd.grad(grad1[:, 1], coords, grad_outputs=torch.ones_like(grad1[:,1]), create_graph=True)[0][:, 1]
    eta_tt = torch.autograd.grad(grad1[:, 2], coords, grad_outputs=torch.ones_like(grad1[:,2]), create_graph=True)[0][:, 2]
    return torch.mean((eta_tt - c2 * (eta_xx + eta_yy)) ** 2)

@torch.no_grad()
def evaluate_full(model, dataset: SWE2DDataset, device, batch_pts=8192, max_samples=None):
    model.eval()
    total_err2, total_norm2 = 0.0, 0.0
    T, Nx, Ny = dataset.T, dataset.Nx, dataset.Ny
    ti, xi, yi = np.arange(1, T), np.arange(Nx), np.arange(Ny)
    TT, XX, YY = np.meshgrid(ti, xi, yi, indexing='ij')
    trunk_full = np.stack([dataset.x_norm[XX.ravel()], dataset.y_norm[YY.ravel()], dataset.t_norm[TT.ravel()]], axis=-1).astype(np.float32)

    n_eval = min(len(dataset), max_samples) if max_samples else len(dataset)
    for idx in range(n_eval):
        field = dataset.h[idx]
        
        if dataset.partial:
            branch_np = field[:, dataset.sensor_x, dataset.sensor_y].flatten()
        else:
            branch_np = field[0].flatten()
            
        labels_np = field[1:].ravel()
        br_t = torch.from_numpy(branch_np).to(device)
        N_pts = trunk_full.shape[0]
        preds = []

        for s in range(0, N_pts, batch_pts):
            e = min(s + batch_pts, N_pts)
            bsz = e - s
            b_in = br_t.unsqueeze(0).expand(bsz, -1)
            t_in = torch.from_numpy(trunk_full[s:e]).to(device)
            preds.append(model(b_in, t_in).view(-1).cpu().numpy())

        pred_all = np.concatenate(preds)
        err = pred_all - labels_np
        total_err2  += float(np.sum(err ** 2))
        total_norm2 += float(np.sum(labels_np ** 2))
    return float(np.sqrt(total_err2 / (total_norm2 + 1e-12)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     type=str, default="data/pdebench/2D_rdb_NA_NA.h5")
    ap.add_argument("--n_total",  type=int, default=1000)
    ap.add_argument("--pts",      type=int, default=512)
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--bs",       type=int, default=16)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--physics",  action="store_true")
    ap.add_argument("--lam",      type=float, default=0.1)
    ap.add_argument("--partial",  action="store_true", help="Partial Observation (interior 4x4 grid)")
    ap.add_argument("--boundary", action="store_true", help="Boundary-Only Observation (sensors on edges)")
    ap.add_argument("--num_sensors", type=int, default=16, help="Number of sensors for partial/boundary")
    ap.add_argument("--eval_max", type=int, default=100)
    ap.add_argument("--outdir",   type=str, default="outputs/benchmark")
    
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    h_all, t_coord, x_coord, y_coord = load_pdebench_swe(args.data, args.n_total)
    N = len(h_all)
    n_train = int(0.8 * N)
    
    train_ds = SWE2DDataset(h_all[:n_train], t_coord, x_coord, y_coord, args.pts, 
                            partial=args.partial, boundary=args.boundary, num_sensors=args.num_sensors)
    val_ds   = SWE2DDataset(h_all[n_train:], t_coord, x_coord, y_coord, args.pts, 
                            partial=args.partial, boundary=args.boundary, num_sensors=args.num_sensors)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    if args.boundary:
        tag = f"Boundary-DeepONet-{train_ds.num_sensors}s"
    elif args.partial:
        tag = "Partial-DeepONet"
    elif args.physics:
        tag = "PI-DeepONet"
    else:
        tag = "DeepONet"
    model = DeepONet(
        branch_in=train_ds.branch_dim, trunk_in=3,
        width=256, depth=4, latent_dim=128, activation="gelu",
    ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()

    best_rel_l2 = float("inf")
    ckpt_path = os.path.join(args.outdir, f"best_{tag.lower().replace('-','_')}.pt")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_d, ep_p, n_batch = 0.0, 0.0, 0
        for b_in, t_in, y_lb in train_dl:
            B, P = t_in.shape[0], t_in.shape[1]
            b_in, t_in, y_lb = b_in.to(device), t_in.to(device).reshape(B*P, 3), y_lb.to(device).reshape(B*P, 1)
            if args.physics: t_in.requires_grad_(True)
            
            b_exp = b_in.unsqueeze(1).expand(-1, P, -1).reshape(B*P, -1)
            y_hat = model(b_exp, t_in)
            loss_d = mse(y_hat, y_lb)
            
            loss_p = torch.tensor(0.0, device=device)
            if args.physics: loss_p = physics_residual(y_hat, t_in)
            
            loss = loss_d + args.lam * loss_p
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            ep_d += loss_d.item(); ep_p += loss_p.item(); n_batch += 1
            
        sched.step()
        
        model.eval()
        v_err, v_nrm = 0.0, 0.0
        with torch.no_grad():
            for b_in, t_in, y_lb in val_dl:
                B, P = t_in.shape[0], t_in.shape[1]
                b_in, t_in, y_lb = b_in.to(device), t_in.to(device).reshape(B*P, 3), y_lb.to(device).reshape(B*P, 1)
                b_exp = b_in.unsqueeze(1).expand(-1, P, -1).reshape(B*P, -1)
                y_hat = model(b_exp, t_in)
                v_err += torch.sum((y_hat - y_lb)**2).item()
                v_nrm += torch.sum(y_lb**2).item()
                
        val_rl2 = float(np.sqrt(v_err / (v_nrm + 1e-12)))
        star = " ★" if val_rl2 < best_rel_l2 else ""
        if star:
            best_rel_l2 = val_rl2
            torch.save({"model": model.state_dict(), "val_rel_l2": val_rl2}, ckpt_path)
            
        print(f"[{ep:03d}/{args.epochs}]  data={ep_d/n_batch:.4e}  phys={ep_p/n_batch:.4e}  val_relL2={val_rl2*100:.2f}%{star}")

    train_time = time.time() - t0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    
    print("\n── Full-field evaluation ──")
    t_inf = time.time()
    full_rl2 = evaluate_full(model, val_ds, device, max_samples=args.eval_max)
    inf_time = time.time() - t_inf
    
    print(f"\n╔══════════════════════════════════════════╗")
    print(f"║  {tag:^38}  ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Val Rel-L2 (full) : {full_rl2*100:>11.2f}%     ║")
    print(f"║  Train time        : {train_time:>11.1f}s     ║")
    print(f"╚══════════════════════════════════════════╝\n")

if __name__ == "__main__":
    main()
