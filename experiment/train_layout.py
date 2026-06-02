"""
Train OVCNO with a specific sensor layout (real/equispaced/random).
Usage:
    python train_layout.py --layout sensors_real_stations.json --name real_k12
    python train_layout.py --layout sensors_equispaced.json --name equispaced_k12
    python train_layout.py --layout sensors_random_seed0.json --name random_k12_s0
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_ovcno_layout import OVCNOLayoutDataset
from model_ovcno import ObservabilityAwareVCNO
from loss_ovcno import compute_ovcno_loss

def collate_fn(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set training seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    print(f"=== Training OVCNO: layout={args.name} seed={args.seed} on {device} ===")
    
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    
    train_ds = OVCNOLayoutDataset(nc_path, args.layout, pts_per_sample=512,
                                   T_obs=24, split="train", seed=42)
    train_mean = train_ds.train_mean  # computed from train split only
    val_ds = OVCNOLayoutDataset(nc_path, args.layout, pts_per_sample=2048,
                                 T_obs=24, split="val", seed=999,
                                 train_mean=train_mean)
    
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    
    ckpt_path = f"ckpt_{args.name}.pt"
    best_nll = float('inf')
    
    for epoch in range(1, 101):
        model.train()
        t0 = time.time()
        ep_loss = 0; ep_nll = 0; ep_kl = 0; n_batches = 0
        
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk = trunk.view(B*P, 4).to(device)
            labels = labels.view(B*P, 1).to(device)
            d_s = trunk[:, 3:4]
            
            y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk)
            loss, lnll, lkl, lobs = compute_ovcno_loss(y_mu, y_logvar, labels, mu_z, logvar_z, o_i, d_s)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            ep_loss += loss.item()
            ep_nll += lnll.item()
            ep_kl += lkl.item()
            n_batches += 1
        
        sched.step()
        
        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_nll = 0; val_n = 0
            with torch.no_grad():
                for hist, pts, trunk, labels in val_dl:
                    B, P = trunk.shape[0], trunk.shape[1]
                    hist, pts = hist.to(device), pts.to(device)
                    trunk = trunk.view(B*P, 4).to(device)
                    labels = labels.view(B*P, 1).to(device)
                    y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk)
                    _, nll, _, _ = compute_ovcno_loss(y_mu, y_logvar, labels, mz, lz, oi, trunk[:, 3:4])
                    val_nll += nll.item()
                    val_n += 1
            
            val_nll /= max(val_n, 1)
            elapsed = time.time() - t0
            print(f"Ep {epoch:3d} [{elapsed:.1f}s] Train L={ep_loss/n_batches:.3f} NLL={ep_nll/n_batches:.3f} KL={ep_kl/n_batches:.4f} | Val NLL={val_nll:.4f}", end="")
            
            if val_nll < best_nll:
                best_nll = val_nll
                torch.save(model.state_dict(), ckpt_path)
                print(f" ★ Best! Saved {ckpt_path}")
            else:
                print()
        else:
            elapsed = time.time() - t0
            print(f"Ep {epoch:3d} [{elapsed:.1f}s] Train L={ep_loss/n_batches:.3f} NLL={ep_nll/n_batches:.3f} KL={ep_kl/n_batches:.4f}")
    
    print(f"\n=== Done! Best val NLL = {best_nll:.4f} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, required=True, help="Path to sensor layout JSON")
    parser.add_argument("--name", type=str, required=True, help="Experiment name for checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Training random seed")
    args = parser.parse_args()
    train(args)
