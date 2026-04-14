"""
Train ForecastDeepONet on PDEBench 2D SWE.

Variable T_obs per batch: we pad sensor histories to the max T_obs in the batch.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from model import ForecastDeepONet
from dataset import PDEBenchForecastDataset, load_pdebench


def collate_fn(batch):
    """Pad sensor histories to max T_obs in batch."""
    hists, t_obs_list, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    n_sensors = hists[0].shape[1]

    # Pad to T_max with zeros (LSTM will just read padding at end, doesn't matter)
    padded = np.zeros((len(hists), T_max, n_sensors), dtype=np.float32)
    lengths = []
    for i, h in enumerate(hists):
        padded[i, :h.shape[0]] = h
        lengths.append(h.shape[0])

    return (
        torch.tensor(padded),                                    # (B, T_max, n_sensors)
        torch.tensor(lengths, dtype=torch.long),                 # (B,)
        torch.tensor(np.stack(trunks)),                         # (B, P, 3)
        torch.tensor(np.stack(labels)),                         # (B, P)
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    err2, norm2 = 0.0, 0.0
    for hist, lengths, trunk, labels in loader:
        B, P = trunk.shape[0], trunk.shape[1]
        hist   = hist.to(device)
        trunk  = trunk.to(device).reshape(B * P, 3)
        labels = labels.to(device).reshape(B * P, 1)

        pred = model(hist, trunk)
        err2  += torch.sum((pred - labels) ** 2).item()
        norm2 += torch.sum(labels ** 2).item()
    return float(np.sqrt(err2 / (norm2 + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     default="../data/pdebench/2D_rdb_NA_NA.h5")
    ap.add_argument("--n_total",  type=int, default=1000)
    ap.add_argument("--epochs",   type=int, default=100)
    ap.add_argument("--bs",       type=int, default=8)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--pts",      type=int, default=512)
    ap.add_argument("--n_sensors",type=int, default=16)
    ap.add_argument("--T_obs_min",type=int, default=10)
    ap.add_argument("--T_obs_max",type=int, default=80)
    ap.add_argument("--lstm_hidden", type=int, default=256)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--latent",   type=int, default=256)
    ap.add_argument("--width",    type=int, default=256)
    ap.add_argument("--depth",    type=int, default=4)
    ap.add_argument("--n_fourier",type=int, default=8)
    ap.add_argument("--ckpt",     default="../checkpoints/forecast_best.pt")
    ap.add_argument("--outdir",   default="../outputs/forecasting")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load data
    print("Loading PDEBench data...")
    data, t_c, x_c, y_c = load_pdebench(args.data, args.n_total)
    N = len(data)
    n_train = int(0.8 * N)

    train_ds = PDEBenchForecastDataset(
        data[:n_train], t_c, x_c, y_c,
        n_sensors=args.n_sensors, pts_per_sample=args.pts,
        T_obs_min=args.T_obs_min, T_obs_max=args.T_obs_max, seed=42)
    val_ds = PDEBenchForecastDataset(
        data[n_train:], t_c, x_c, y_c,
        n_sensors=args.n_sensors, pts_per_sample=args.pts,
        T_obs_min=args.T_obs_min, T_obs_max=args.T_obs_max, seed=99)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # Model
    model = ForecastDeepONet(
        n_sensors=args.n_sensors,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        latent_dim=args.latent,
        width=args.width,
        depth=args.depth,
        n_fourier_freqs=args.n_fourier,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse   = nn.MSELoss()

    best_rl2 = float("inf")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, n_batch = 0.0, 0

        for hist, lengths, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist   = hist.to(device)                    # (B, T_max, n_sensors)
            trunk  = trunk.to(device).reshape(B*P, 3)   # (B*P, 3)
            labels = labels.to(device).reshape(B*P, 1)  # (B*P, 1)

            pred = model(hist, trunk)
            loss = mse(pred, labels)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            n_batch  += 1

        sched.step()
        val_rl2 = evaluate(model, val_dl, device)
        star = " ★" if val_rl2 < best_rl2 else ""
        if star:
            best_rl2 = val_rl2
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_rel_l2": val_rl2, "args": vars(args)}, args.ckpt)

        print(f"[{ep:03d}/{args.epochs}] loss={ep_loss/n_batch:.4e}  val_relL2={val_rl2*100:.2f}%{star}")

    elapsed = time.time() - t0
    print(f"\nBest val Rel L2: {best_rl2*100:.2f}%  |  Train time: {elapsed:.1f}s")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
