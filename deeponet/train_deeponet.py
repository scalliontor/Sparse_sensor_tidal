# train_deeponet.py
from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from model import DeepONet
from data import DeepONetNPZDataset
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
    p.add_argument("--data", type=str, required=True, help="Path to compiled dataset .npz")
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

    p.add_argument("--ckpt", type=str, default="checkpoints/deeponet_best.pt")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device()
    print("Device:", device)

    ds = DeepONetNPZDataset(args.data)
    m = ds.m
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = DeepONet(
        branch_in=m,
        trunk_in=2,
        width=args.width,
        depth=args.depth,
        latent_dim=args.latent,
        activation=args.act,
        dropout=args.dropout,
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
