import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from loss_ovcno import compute_ovcno_loss

def collate_fn(batch):
    hists, pts, trunks, labels = zip(*batch)
    # hist: (T, K)
    # pts: (K, 2)
    # trunk: (P, 4)
    # labels: (P)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
        
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)

def train_network():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training OVCNO on {device}")
    
    # Setting A: Fixed 16 sensors
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    train_ds = CopernicusOVCNODataset(nc_path, n_sensors=16, pts_per_sample=512, split="train", variable_sensors=False)
    val_ds = CopernicusOVCNODataset(nc_path, n_sensors=16, pts_per_sample=2048, split="val", variable_sensors=False)
    
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    
    best_nll = float('inf')
    
    for epoch in range(1, 41):
        model.train()
        t0 = time.time()
        ep_loss = 0; ep_nll = 0; ep_kl = 0; ep_obs = 0
        
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk = trunk.view(B*P, 4).to(device)
            labels = labels.view(B*P, 1).to(device)
            
            d_s = trunk[:, 3:4]
            
            y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk)
            
            # Use adaptive beta starting beta_0 = 1e-4, beta_1 = 1e-3, lambda_obs=1.0
            loss, lnll, lkl, lobs = compute_ovcno_loss(y_mu, y_logvar, labels, mu_z, logvar_z, o_i, d_s)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            ep_loss += loss.item()
            ep_nll += lnll.item()
            ep_kl += lkl.item()
            ep_obs += lobs.item()
            
        sched.step()
        n_b = len(train_dl)
        print(f"Ep {epoch} [{time.time()-t0:.1f}s] - Train L: {ep_loss/n_b:.3f} | NLL: {ep_nll/n_b:.3f} | KL: {ep_kl/n_b:.4f} | Obs: {ep_obs/n_b:.4f}")
        
        # Validation
        model.eval()
        val_nll = 0
        with torch.no_grad():
            for hist, pts, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                hist, pts = hist.to(device), pts.to(device)
                trunk = trunk.view(B*P, 4).to(device)
                labels = labels.view(B*P, 1).to(device)
                
                y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk)
                _, nll, _, _ = compute_ovcno_loss(y_mu, y_logvar, labels, mz, lz, oi, trunk[:, 3:4])
                val_nll += nll.item()
        
        val_nll /= len(val_dl)
        print(f"   >>> Val NLL: {val_nll:.4f}")
        if val_nll < best_nll:
            best_nll = val_nll
            torch.save(model.state_dict(), "ovcno_checkpoint.pt")
            print("   >>> Saved best model!")

if __name__ == "__main__":
    train_network()
