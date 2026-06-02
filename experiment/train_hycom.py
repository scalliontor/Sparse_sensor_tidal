import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_vae import ForecastDeepONetVAE
from loss import compute_vae_loss
from dataset_vae import CopernicusVAEDataset

def collate_fn(batch):
    """Pad sensor histories to max T_obs in batch."""
    hists, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    n_sensors = hists[0].shape[1]

    # Pad to T_max with zeros (LSTM ignores right padding safely usually or deals with it if causal)
    padded = np.zeros((len(hists), T_max, n_sensors), dtype=np.float32)
    for i, h in enumerate(hists):
        padded[i, :h.shape[0]] = h

    return (
        torch.tensor(padded),
        torch.stack(trunks),
        torch.stack(labels)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_path",  default="../data/real_data/copernicus_ssh_tonkin_jan2024.nc")
    ap.add_argument("--epochs",   type=int, default=50) # Reduced for quick iteration
    ap.add_argument("--bs",       type=int, default=8)
    ap.add_argument("--lr",       type=float, default=5e-4)
    ap.add_argument("--beta",     type=float, default=1e-3, help="KL Divergence weight")
    ap.add_argument("--ckpt",     default="vae_checkpoint.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading datasets...")
    train_ds = CopernicusVAEDataset(args.nc_path, split="train", pts_per_sample=512)
    val_ds = CopernicusVAEDataset(args.nc_path, split="val", pts_per_sample=512)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=128, latent_dim=128).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE Model params: {total_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_nll = float('inf')
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_nll, ep_kl, n_batch = 0.0, 0.0, 0.0, 0

        for hist, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist   = hist.to(device)
            trunk  = trunk.to(device).reshape(B*P, 3)
            labels = labels.to(device).reshape(B*P, 1)

            # Forward
            y_mu, y_logvar, mu_z, logvar_z = model(hist, trunk)
            
            # Loss
            loss, nll, kl = compute_vae_loss(y_mu, y_logvar, labels, mu_z, logvar_z, beta=args.beta)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            ep_nll += nll.item()
            ep_kl += kl.item()
            n_batch += 1

        sched.step()

        # Validation
        model.eval()
        v_loss, v_nll, v_kl, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for hist, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                pred_mu, pred_logvar, mz, lz = model(hist.to(device), trunk.to(device).reshape(B*P, 3))
                val_total, val_nll, val_kl = compute_vae_loss(pred_mu, pred_logvar, labels.to(device).reshape(B*P, 1), mz, lz, beta=args.beta)
                v_loss += val_total.item()
                v_nll += val_nll.item()
                v_kl += val_kl.item()
                vn += 1

        v_loss, v_nll, v_kl = v_loss/vn, v_nll/vn, v_kl/vn
        
        star = " *" if v_nll < best_nll else ""
        if star:
            best_nll = v_nll
            torch.save(model.state_dict(), args.ckpt)

        print(f"[{ep:03d}/{args.epochs}] Train Total: {ep_loss/n_batch:.4f} (NLL: {ep_nll/n_batch:.4f}, KL: {ep_kl/n_batch:.4f}) "
              f"| Val NLL: {v_nll:.4f} KL: {v_kl:.4f}{star}")

    print(f"Training Complete. Best Val NLL: {best_nll:.4f}. Time: {time.time()-t0:.1f}s")
    
if __name__ == "__main__":
    main()
