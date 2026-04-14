"""
train_deeponet_2d_pi.py — Physics-Informed DeepONet training
=============================================================
Adds wave-equation residual loss (simplified mass conservation / linear SWE)
on top of the data MSE loss.

  L_total = L_data + lambda_phys * L_physics

Physics constraint (linear shallow-water wave equation):
  ∂²η/∂t² − c² (∂²η/∂x² + ∂²η/∂y²) = 0
  c = sqrt(g * H_mean)  [H_mean ≈ 50 m for Gulf of Tonkin → c ≈ 22 m/s]

All derivatives are taken w.r.t. the z-score-normalised trunk coordinates
(x_n, y_n, t_n).  The normalisation scales cancel in the residual, but we
must account for the coordinate-mapping Jacobians:

  c²_eff_x = c² · (Δt_phys / Δx_phys)²  ·  (trunk_std_t / trunk_std_x)²
  c²_eff_y = c² · (Δt_phys / Δy_phys)²  ·  (trunk_std_t / trunk_std_y)²

Defaults (Gulf of Tonkin grid, 167-h window, 10-sensor branch):
  Δx_phys ≈ 5° × 111 000 m = 555 000 m
  Δy_phys ≈ 6° × 111 000 m = 666 000 m
  Δt_phys = 166 h × 3600 s/h  = 597 600 s
  trunk_std ≈ 0.5808 (x), 0.5826 (y), 0.5809 (t)
  → c²_eff_x ≈ 484 × (597600/555000)² × (0.5809/0.5808)² ≈ 560
  → c²_eff_y ≈ 484 × (597600/666000)² × (0.5809/0.5826)² ≈ 389

lambda_phys starts small and is warmed up every 10 epochs.
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Allow running from either project root or deeponet/
sys.path.insert(0, os.path.dirname(__file__))

from model import DeepONet
from data import DeepONetNPZDataset
from metrics import rmse, mae
from utils import seed_all, get_device, save_checkpoint


# ---------------------------------------------------------------------------
# Physics loss
# ---------------------------------------------------------------------------

def wave_equation_residual(
    model: DeepONet,
    branch_colloc: torch.Tensor,   # (M, branch_in)  — detached, no grad
    trunk_colloc: torch.Tensor,    # (M, 3)           — will enable grad
    c2_eff_x: float = 560.0,
    c2_eff_y: float = 389.0,
) -> torch.Tensor:
    """
    Compute mean-square wave-equation residual over M collocation points.

    Returns a scalar tensor (with grad) suitable for .backward().
    """
    trunk_colloc = trunk_colloc.detach().requires_grad_(True)
    branch_colloc = branch_colloc.detach()

    eta = model(branch_colloc, trunk_colloc)          # (M, 1)

    # ---- first-order gradients ----
    ones = torch.ones_like(eta)
    g1 = torch.autograd.grad(
        eta, trunk_colloc,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]                                              # (M, 3): ∂η/∂x, ∂η/∂y, ∂η/∂t

    deta_dx = g1[:, 0]   # ∂η/∂x_n
    deta_dy = g1[:, 1]   # ∂η/∂y_n
    deta_dt = g1[:, 2]   # ∂η/∂t_n

    # ---- second-order ----
    ones_v = torch.ones_like(deta_dt)

    d2eta_dt2 = torch.autograd.grad(
        deta_dt, trunk_colloc,
        grad_outputs=ones_v,
        create_graph=True,
        retain_graph=True,
    )[0][:, 2]                                        # ∂²η/∂t_n²

    d2eta_dx2 = torch.autograd.grad(
        deta_dx, trunk_colloc,
        grad_outputs=torch.ones_like(deta_dx),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0]                                        # ∂²η/∂x_n²

    d2eta_dy2 = torch.autograd.grad(
        deta_dy, trunk_colloc,
        grad_outputs=torch.ones_like(deta_dy),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1]                                        # ∂²η/∂y_n²

    # Wave-equation residual: ∂²η/∂t² − c²_x ∂²η/∂x² − c²_y ∂²η/∂y² = 0
    residual = d2eta_dt2 - c2_eff_x * d2eta_dx2 - c2_eff_y * d2eta_dy2
    return torch.mean(residual ** 2)


def sample_collocation_points(
    n: int,
    device: torch.device,
    trunk_min: float = -2.0,
    trunk_max: float =  2.0,
) -> torch.Tensor:
    """
    Sample M random collocation points uniformly in the normalised trunk domain.
    Returns (n, 3) float32 tensor.
    """
    return (
        torch.rand(n, 3, device=device) * (trunk_max - trunk_min) + trunk_min
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch_pi(
    model: DeepONet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_phys: float,
    n_colloc: int,
    c2_eff_x: float,
    c2_eff_y: float,
) -> dict:
    model.train()
    total_data = total_phys = 0.0
    n = 0

    for branch_x, trunk_x, y in loader:
        branch_x = branch_x.to(device)
        trunk_x  = trunk_x.to(device)
        y        = y.to(device)

        # --- data loss ---
        pred  = model(branch_x, trunk_x)
        l_data = torch.mean((pred - y) ** 2)

        # --- physics loss ---
        # sample random collocation points (no data needed, physics only)
        trunk_colloc = sample_collocation_points(n_colloc, device)
        # use random branch vectors from current batch (reuse statistics)
        idx_branch = torch.randint(0, branch_x.shape[0], (n_colloc,), device=device)
        branch_colloc = branch_x[idx_branch].detach()

        l_phys = wave_equation_residual(
            model, branch_colloc, trunk_colloc, c2_eff_x, c2_eff_y
        )

        l_total = l_data + lambda_phys * l_phys

        optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = y.shape[0]
        total_data += l_data.item() * bs
        total_phys += l_phys.item() * bs
        n += bs

    return {
        "data_mse": total_data / max(n, 1),
        "phys_mse": total_phys / max(n, 1),
    }


@torch.no_grad()
def eval_model(model: DeepONet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, ys = [], []
    for branch_x, trunk_x, y in loader:
        branch_x = branch_x.to(device)
        trunk_x  = trunk_x.to(device)
        y        = y.to(device)
        preds.append(model(branch_x, trunk_x))
        ys.append(y)
    pred = torch.cat(preds, dim=0)
    y    = torch.cat(ys,    dim=0)
    return {"rmse": rmse(pred, y), "mae": mae(pred, y)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="PI-DeepONet: DeepONet + wave-equation physics loss"
    )
    p.add_argument("--data",         type=str,   default="../data/dataset_2d_v2.npz")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch",        type=int,   default=8192)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--val_frac",     type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)

    # Model hyper-parameters
    p.add_argument("--width",   type=int,   default=256)
    p.add_argument("--depth",   type=int,   default=4)
    p.add_argument("--latent",  type=int,   default=128)
    p.add_argument("--act",     type=str,   default="gelu")
    p.add_argument("--dropout", type=float, default=0.0)

    # Physics-loss hyper-parameters
    p.add_argument("--lambda_phys_start", type=float, default=0.01,
                   help="Initial weight for physics loss")
    p.add_argument("--lambda_phys_max",   type=float, default=1.0,
                   help="Maximum weight for physics loss")
    p.add_argument("--lambda_warmup",     type=int,   default=10,
                   help="Multiply lambda_phys by 1.5 every this many epochs")
    p.add_argument("--n_colloc",          type=int,   default=1000,
                   help="Collocation points per batch for physics loss")
    p.add_argument("--c2_eff_x",          type=float, default=560.0,
                   help="Effective c² in x (normalised coord space)")
    p.add_argument("--c2_eff_y",          type=float, default=389.0,
                   help="Effective c² in y (normalised coord space)")

    p.add_argument("--ckpt", type=str, default="../checkpoints/deeponet_2d_pi_best.pt")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Physics loss: lambda_start={args.lambda_phys_start}, "
          f"lambda_max={args.lambda_phys_max}, "
          f"c2_eff=(x={args.c2_eff_x}, y={args.c2_eff_y}), "
          f"n_colloc={args.n_colloc}")

    # Dataset
    ds = DeepONetNPZDataset(args.data)
    m = ds.m
    n_val   = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    model = DeepONet(
        branch_in=m,
        trunk_in=3,
        width=args.width,
        depth=args.depth,
        latent_dim=args.latent,
        activation=args.act,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    lambda_phys = args.lambda_phys_start
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch_pi(
            model, train_loader, optimizer, device,
            lambda_phys=lambda_phys,
            n_colloc=args.n_colloc,
            c2_eff_x=args.c2_eff_x,
            c2_eff_y=args.c2_eff_y,
        )
        val = eval_model(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"data_mse={tr['data_mse']:.4e} | "
            f"phys_mse={tr['phys_mse']:.4e} | "
            f"lambda={lambda_phys:.4f} | "
            f"val_rmse={val['rmse']:.4e} | "
            f"val_mae={val['mae']:.4e}"
        )

        if val["rmse"] < best:
            best = val["rmse"]
            save_checkpoint(args.ckpt, model, optimizer, epoch, best)
            print(f"  [Saved] best val RMSE → {best:.6e}")

        # Lambda warm-up schedule
        if epoch % args.lambda_warmup == 0 and epoch > 0:
            lambda_phys = min(lambda_phys * 1.5, args.lambda_phys_max)
            print(f"  [Schedule] lambda_phys updated → {lambda_phys:.4f}")

    print(f"\nTraining complete. Best val RMSE: {best:.6e}")
    print(f"Checkpoint saved to: {args.ckpt}")


if __name__ == "__main__":
    main()
