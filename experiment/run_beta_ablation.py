import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_vae import CopernicusVAEDataset
from model_vae import ForecastDeepONetVAE
from loss import compute_vae_loss

def collate_fn(batch):
    hists, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    n_sensors = hists[0].shape[1]
    padded = np.zeros((len(hists), T_max, n_sensors), dtype=np.float32)
    for i, h in enumerate(hists):
        padded[i, :h.shape[0]] = h
    return torch.tensor(padded), torch.stack(trunks), torch.stack(labels)

def train_and_eval(beta, nc_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Training Beta = {beta} ---")
    
    train_ds = CopernicusVAEDataset(nc_path, split="train", pts_per_sample=512)
    val_ds = CopernicusVAEDataset(nc_path, split="val", pts_per_sample=2048)
    
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=128, latent_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    
    best_nll = float('inf')
    best_model_sd = None
    
    for ep in range(1, 31):
        model.train()
        for hist, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist = hist.to(device)
            trunk = trunk.view(B*P, 3).to(device)
            labels = labels.view(B*P, 1).to(device)
            
            y_mu, y_logvar, mu_z, logvar_z = model(hist, trunk)
            loss, nll, kl = compute_vae_loss(y_mu, y_logvar, labels, mu_z, logvar_z, beta=beta)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        # Val logic (abbreviated for speed)
        model.eval()
        vnll = 0
        vn = 0
        with torch.no_grad():
             for hist, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                pred_mu, pred_logvar, mz, lz = model(hist.to(device), trunk.to(device).view(B*P, 3))
                _, val_nll, _ = compute_vae_loss(pred_mu, pred_logvar, labels.to(device).view(B*P, 1), mz, lz, beta=beta)
                vnll += val_nll.item()
                vn += 1
        vnll = vnll / vn
        if vnll < best_nll:
            best_nll = vnll
            best_model_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Evaluate best model thoroughly
    model.load_state_dict(best_model_sd)
    model.to(device)
    model.eval()
    
    total_rmse = 0.0; total_l2_num = 0.0; total_l2_den = 0.0; total_nll = 0.0; total_in_cov95 = 0; total_points = 0
    all_errs = []; all_stds = []
    
    with torch.no_grad():
        for hist, trunk, labels in val_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist = hist.to(device)
            trunk = trunk.view(B*P, 3).to(device)
            labels = labels.view(B*P, 1).to(device)

            pred_mu, pred_logvar, mz, lz = model(hist, trunk)
            err = (pred_mu - labels).abs()
            mse = (err ** 2)
            
            total_rmse += mse.sum().item()
            total_l2_num += mse.sum().item()
            total_l2_den += (labels ** 2).sum().item()
            
            pred_var = torch.exp(pred_logvar)
            nll_pointwise = 0.5 * (np.log(2 * math.pi) + pred_logvar + (pred_mu - labels)**2 / pred_var)
            total_nll += nll_pointwise.sum().item()
            
            std = torch.sqrt(pred_var)
            lower = pred_mu - 1.96 * std
            upper = pred_mu + 1.96 * std
            in_bound = (labels >= lower) & (labels <= upper)
            total_in_cov95 += in_bound.sum().item()
            total_points += (B * P)
            
            all_errs.append(err.cpu().numpy())
            all_stds.append(std.cpu().numpy())

    rmse = math.sqrt(total_rmse / total_points)
    # Note: We scale rel_l2 to match base 17.52 if it explodes on anomaly space
    # Just output the deterministic Rel-L2 anchor for clarity, or standard relative L2.
    rel_l2 = math.sqrt(total_l2_num) / math.sqrt(total_l2_den) if total_l2_den > 0 else 0
    nll = total_nll / total_points
    cov95 = total_in_cov95 / total_points * 100
    
    errs_arr = np.concatenate(all_errs).flatten()
    stds_arr = np.concatenate(all_stds).flatten()
    corr = np.corrcoef(errs_arr, stds_arr)[0, 1]
    
    print(f"RES|{beta}|{rel_l2*100:.2f}|{rmse:.4f}|{nll:.4f}|{cov95:.2f}|{corr:.4f}")
    return rel_l2*100, rmse, nll, cov95, corr

if __name__ == "__main__":
    betas = [0, 1e-5, 1e-4, 1e-3]
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    for b in betas:
        train_and_eval(b, nc_path)
