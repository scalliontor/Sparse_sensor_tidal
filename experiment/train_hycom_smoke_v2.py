"""
HYCOM Cross-Product Smoke Test v2 — Extended data (Jan-Sep)
Fixes: more training data, ReduceLROnPlateau, early stopping, dropout.
Run VCO + OVCNO for 1 seed, ~80 epochs.
"""
import os
import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from dataset_hycom import HYCOMOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE
from loss_ovcno import compute_ovcno_loss

# ─── Paths ───
NC_PATH = "../data/hycom_data/hycom_ssh_tonkin_jan_sep_2024.nc"
STATION_JSON = "hycom_real_k12_stations.json"
SAVE_DIR = "hycom_smoke_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Config ───
SEED = 42
EPOCHS = 80
BATCH_SIZE = 16  # larger batch for more stable gradients
LR = 3e-4  # slightly lower LR
PTS_PER_SAMPLE = 512
T_OBS = 8

def collate_fn(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)


def train_model(model_type="ovcno"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"HYCOM Smoke v2: {model_type.upper()}, seed={SEED}, extended data")
    print(f"{'='*60}")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    train_ds = HYCOMOVCNODataset(
        NC_PATH, STATION_JSON,
        pts_per_sample=PTS_PER_SAMPLE, T_obs=T_OBS,
        seed=SEED, split="train", variable_sensors=False
    )
    val_ds = HYCOMOVCNODataset(
        NC_PATH, STATION_JSON,
        pts_per_sample=2048, T_obs=T_OBS,
        seed=SEED, split="val", variable_sensors=False,
        train_mean=train_ds.train_mean
    )
    test_ds = HYCOMOVCNODataset(
        NC_PATH, STATION_JSON,
        pts_per_sample=2048, T_obs=T_OBS,
        seed=SEED, split="test", variable_sensors=False,
        train_mean=train_ds.train_mean
    )
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False,
                         collate_fn=collate_fn, num_workers=2)

    n_sensors = train_ds.n_stations
    if model_type == "ovcno":
        model = ObservabilityAwareVCNO(
            lstm_hidden=256, latent_dim=256, width=256
        ).to(device)
    else:
        model = ForecastDeepONetVAE(
            n_sensors=n_sensors, lstm_hidden=256, latent_dim=256, width=256
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )
    
    best_val_nll = float('inf')
    patience_counter = 0
    PATIENCE = 20
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0; ep_nll = 0; n_batches = 0
        
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk = trunk.view(B * P, 4).to(device)
            labels = labels.view(B * P, 1).to(device)
            d_s = trunk[:, 3:4]
            
            if model_type == "ovcno":
                y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk)
                loss, lnll, lkl, lobs = compute_ovcno_loss(
                    y_mu, y_logvar, labels, mu_z, logvar_z, o_i, d_s
                )
            else:
                y_mu, y_logvar, mu_z, logvar_z = model(hist, trunk[:, :3])
                nll = 0.5 * (y_logvar + (labels - y_mu)**2 / torch.exp(y_logvar)).mean()
                kl = -0.5 * (1 + logvar_z - mu_z**2 - torch.exp(logvar_z)).mean()
                loss = nll + 1e-3 * kl
                lnll = nll
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            ep_loss += loss.item()
            ep_nll += lnll.item()
            n_batches += 1
        
        train_losses.append(ep_loss / n_batches)
        
        # Validation
        model.eval()
        val_nll = 0; val_batches = 0
        with torch.no_grad():
            for hist, pts, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                hist, pts = hist.to(device), pts.to(device)
                trunk = trunk.view(B * P, 4).to(device)
                labels = labels.view(B * P, 1).to(device)
                
                if model_type == "ovcno":
                    y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk)
                    _, nll_v, _, _ = compute_ovcno_loss(
                        y_mu, y_logvar, labels, mz, lz, oi, trunk[:, 3:4]
                    )
                else:
                    y_mu, y_logvar, mz, lz = model(hist, trunk[:, :3])
                    nll_v = 0.5 * (y_logvar + (labels - y_mu)**2 / torch.exp(y_logvar)).mean()
                
                val_nll += nll_v.item()
                val_batches += 1
        
        val_nll_avg = val_nll / val_batches
        val_losses.append(val_nll_avg)
        sched.step(val_nll_avg)
        
        improved = ""
        if val_nll_avg < best_val_nll:
            best_val_nll = val_nll_avg
            ckpt_path = os.path.join(SAVE_DIR, f"hycom_{model_type}_seed{SEED}.pt")
            torch.save(model.state_dict(), ckpt_path)
            improved = " ★"
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 1 or improved:
            lr_now = opt.param_groups[0]['lr']
            print(f"  Ep {epoch:>3}/{EPOCHS} [{time.time()-t0:.1f}s] "
                  f"TrainL={train_losses[-1]:.4f} ValNLL={val_nll_avg:.4f} "
                  f"lr={lr_now:.1e}{improved}")
        
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break
    
    print(f"\n  Best Val NLL: {best_val_nll:.4f}")
    
    # ─── Test evaluation ───
    print(f"\n  Evaluating on TEST set...")
    ckpt_path = os.path.join(SAVE_DIR, f"hycom_{model_type}_seed{SEED}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    all_mu = []; all_labels = []; all_logvar = []
    with torch.no_grad():
        for hist, pts, trunk, labels in test_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk = trunk.view(B * P, 4).to(device)
            labels = labels.view(B * P, 1).to(device)
            
            if model_type == "ovcno":
                y_mu, y_logvar, _, _, _ = model(hist, pts, trunk)
            else:
                y_mu, y_logvar, _, _ = model(hist, trunk[:, :3])
            
            all_mu.append(y_mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_logvar.append(y_logvar.cpu().numpy())
    
    mu_all = np.concatenate(all_mu).flatten()
    lab_all = np.concatenate(all_labels).flatten()
    lv_all = np.concatenate(all_logvar).flatten()
    std_all = np.exp(0.5 * lv_all)
    
    rmse = np.sqrt(np.mean((mu_all - lab_all) ** 2))
    mae = np.mean(np.abs(mu_all - lab_all))
    nll_test = 0.5 * np.mean(lv_all + (lab_all - mu_all)**2 / np.exp(lv_all))
    
    z95 = 1.96
    lower = mu_all - z95 * std_all
    upper = mu_all + z95 * std_all
    coverage = np.mean((lab_all >= lower) & (lab_all <= upper))
    avg_width = np.mean(upper - lower)
    
    from scipy.stats import spearmanr
    errors = np.abs(mu_all - lab_all)
    corr_s, _ = spearmanr(errors, std_all)
    
    print(f"\n  ═══ {model_type.upper()} TEST RESULTS (Extended) ═══")
    print(f"  RMSE:    {rmse:.4f}")
    print(f"  MAE:     {mae:.4f}")
    print(f"  NLL:     {nll_test:.4f}")
    print(f"  Cov@95:  {coverage*100:.1f}%")
    print(f"  Avg.W:   {avg_width:.4f}")
    print(f"  Corr_S:  {corr_s:.4f}")
    
    import json
    results = {
        "model": model_type, "seed": SEED, "dataset": "HYCOM_extended",
        "rmse": float(rmse), "mae": float(mae), "nll": float(nll_test),
        "coverage_95": float(coverage), "avg_width": float(avg_width),
        "corr_spearman": float(corr_s),
        "best_val_nll": float(best_val_nll),
        "epochs_trained": epoch,
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
    }
    results_path = os.path.join(SAVE_DIR, f"hycom_{model_type}_seed{SEED}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")
    
    return results


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║   HYCOM SMOKE TEST v2 — Extended Data        ║")
    print("║   VCO + OVCNO, seed=42, Jan-Sep 2024         ║")
    print("╚══════════════════════════════════════════════╝")
    
    vco_results = train_model("vco")
    ovcno_results = train_model("ovcno")
    
    print("\n" + "="*60)
    print("SMOKE TEST v2 COMPARISON (Extended Data)")
    print("="*60)
    print(f"{'Metric':<12} {'VCO':>10} {'OVCNO':>10} {'Δ':>10}")
    print("-"*42)
    for metric in ['rmse', 'mae', 'nll', 'coverage_95', 'avg_width', 'corr_spearman']:
        v = vco_results[metric]
        o = ovcno_results[metric]
        delta = o - v
        if metric == 'coverage_95':
            print(f"{metric:<12} {v*100:>9.1f}% {o*100:>9.1f}% {delta*100:>+9.1f}%")
        else:
            print(f"{metric:<12} {v:>10.4f} {o:>10.4f} {delta:>+10.4f}")
    print("="*60)
