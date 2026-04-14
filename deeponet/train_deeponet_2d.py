# train_deeponet.py
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from model import DeepONet
import numpy as np
from data import DeepONetNPZDataset, DeepONetArrayDataset
from metrics import rmse, mae
from utils import seed_all, get_device, save_checkpoint


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for branch_x, trunk_x, y in loader:
        branch_x = branch_x.to(device)
        trunk_x = trunk_x.to(device)
        y = y.to(device)

        pred = model(branch_x, trunk_x)
        loss = torch.mean((pred - y) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * y.shape[0]
        n += y.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds = []
    ys = []
    for branch_x, trunk_x, y in loader:
        branch_x = branch_x.to(device)
        trunk_x = trunk_x.to(device)
        y = y.to(device)
        pred = model(branch_x, trunk_x)
        preds.append(pred)
        ys.append(y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return {
        "rmse": rmse(pred, y),
        "mae": mae(pred, y),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, default="../data/dataset_2d_train.npz", help="Path to compiled dataset .npz")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Model hyperparams
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--act", type=str, default="gelu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n_fourier_freqs", type=int, default=0, help="Fourier features for trunk (0=disabled)")

    p.add_argument("--ckpt", type=str, default="../checkpoints/deeponet_2d_best.pt")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device()
    print("Device:", device)

    # Load and filter to train_windows only (proper train/test split)
    raw = np.load(args.data)
    train_windows = raw['train_windows']
    window_id = raw['window_id']
    train_mask = np.isin(window_id, train_windows)
    print(f"Train windows: {train_windows}, using {train_mask.sum():,} samples")

    branch_all = raw['branch'][train_mask]
    trunk_all  = raw['trunk'][train_mask]
    labels_all = raw['labels'][train_mask]
    m = branch_all.shape[1]

    # Val split within train windows only
    n_total = len(labels_all)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    rng_idx = np.random.default_rng(args.seed)
    perm = rng_idx.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = DeepONetArrayDataset(branch_all[train_idx], trunk_all[train_idx], labels_all[train_idx])
    val_ds   = DeepONetArrayDataset(branch_all[val_idx],   trunk_all[val_idx],   labels_all[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = DeepONet(
        branch_in=m,
        trunk_in=3,
        width=args.width,
        depth=args.depth,
        latent_dim=args.latent,
        activation=args.act,
        dropout=args.dropout,
        n_fourier_freqs=args.n_fourier_freqs,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_model(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_mse={tr_loss:.6e} | val_rmse={val_metrics['rmse']:.6e} | val_mae={val_metrics['mae']:.6e}")

        if val_metrics["rmse"] < best:
            best = val_metrics["rmse"]
            save_checkpoint(args.ckpt, model, optimizer, epoch, best)

    print("Best val RMSE:", best)
    print("Saved best checkpoint to:", args.ckpt)


if __name__ == "__main__":
    main()
