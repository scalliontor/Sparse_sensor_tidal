"""
OVCNO Full Ablation Study — trains 5 model variants and evaluates all metrics.
Variants:
  1. VCO baseline             (old model_vae.py, fixed β=1e-3)
  2. OVCNO w/o obs field       (geometry encoder + set encoder, but NO obs_net, fixed β)
  3. OVCNO w/o adaptive β      (geometry + obs field, but β is FIXED globally)
  4. OVCNO w/o ranking loss    (geometry + obs field + adaptive β, but λ_obs=0)
  5. Full OVCNO               (everything)
"""
import os, sys, time, math, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import norm

# ---------- imports ----------
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from loss_ovcno import compute_ovcno_loss

# Also need old VCO model for variant 1
from model_vae import ForecastDeepONetVAE
from loss import compute_vae_loss

# We use the old dataset for VCO baseline
from dataset_vae import CopernicusVAEDataset

NC_PATH = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== collate functions =====================
def collate_vco(batch):
    hists, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    n_sensors = hists[0].shape[1]
    padded = np.zeros((len(hists), T_max, n_sensors), dtype=np.float32)
    for i, h in enumerate(hists):
        padded[i, :h.shape[0]] = h
    return torch.tensor(padded), torch.stack(trunks), torch.stack(labels)

def collate_ovcno(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)

# ===================== evaluation =====================
def evaluate_vco(model, dl):
    model.eval()
    all_mu, all_logvar, all_y = [], [], []
    with torch.no_grad():
        for hist, trunk, labels in dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist = hist.to(DEVICE)
            trunk = trunk.view(B*P, 3).to(DEVICE)
            labels = labels.view(B*P, 1).to(DEVICE)
            y_mu, y_logvar, _, _ = model(hist, trunk)
            all_mu.append(y_mu.cpu()); all_logvar.append(y_logvar.cpu()); all_y.append(labels.cpu())
    return _compute_metrics(torch.cat(all_mu), torch.cat(all_logvar), torch.cat(all_y))

def evaluate_ovcno(model, dl, return_obs=False):
    model.eval()
    all_mu, all_logvar, all_y, all_oi = [], [], [], []
    with torch.no_grad():
        for hist, pts, trunk, labels in dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(DEVICE), pts.to(DEVICE)
            trunk_flat = trunk.view(B*P, 4).to(DEVICE)
            labels_flat = labels.view(B*P, 1).to(DEVICE)
            y_mu, y_logvar, _, _, oi = model(hist, pts, trunk_flat)
            all_mu.append(y_mu.cpu()); all_logvar.append(y_logvar.cpu())
            all_y.append(labels_flat.cpu()); all_oi.append(oi.cpu())
    metrics = _compute_metrics(torch.cat(all_mu), torch.cat(all_logvar), torch.cat(all_y))
    if return_obs:
        metrics["obs_scores"] = torch.cat(all_oi).numpy()
    return metrics

def _compute_metrics(mu, logvar, y):
    mu, logvar, y = mu.numpy().flatten(), logvar.numpy().flatten(), y.numpy().flatten()
    var = np.exp(logvar)
    sigma = np.sqrt(var)
    err = mu - y
    abs_err = np.abs(err)
    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_err)
    rel_l2 = np.sqrt(np.sum(err**2)) / np.sqrt(np.sum(y**2)) * 100
    nll = np.mean(0.5 * (np.log(2*np.pi) + logvar + err**2 / var))
    z95 = norm.ppf(0.975)
    cov95 = np.mean((y >= mu - z95*sigma) & (y <= mu + z95*sigma)) * 100
    corr = np.corrcoef(abs_err, sigma)[0, 1] if sigma.std() > 1e-12 else 0.0
    return {"rel_l2": rel_l2, "rmse": rmse, "mae": mae, "nll": nll, "cov95": cov95, "corr": corr}

# ===================== training loops =====================
def train_vco(tag="VCO_baseline"):
    print(f"\n{'='*60}\nTraining: {tag}\n{'='*60}")
    train_ds = CopernicusVAEDataset(NC_PATH, n_sensors=16, pts_per_sample=512, split="train")
    val_ds = CopernicusVAEDataset(NC_PATH, n_sensors=16, pts_per_sample=2048, split="val")
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_vco, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_vco, num_workers=2)
    
    model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=256, latent_dim=256).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_nll = float('inf')
    
    for ep in range(1, EPOCHS+1):
        model.train(); t0 = time.time(); ep_loss = 0
        for hist, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist = hist.to(DEVICE)
            trunk = trunk.view(B*P, 3).to(DEVICE)
            labels = labels.view(B*P, 1).to(DEVICE)
            y_mu, y_logvar, mu_z, logvar_z = model(hist, trunk)
            loss, _, _ = compute_vae_loss(y_mu, y_logvar, labels, mu_z, logvar_z, beta=1e-3)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep_loss += loss.item()
        sched.step()
        if ep % 10 == 0:
            metrics = evaluate_vco(model, val_dl)
            print(f"  Ep {ep} [{time.time()-t0:.1f}s] NLL={metrics['nll']:.4f} Cov={metrics['cov95']:.1f}%")
            if metrics['nll'] < best_nll:
                best_nll = metrics['nll']
                torch.save(model.state_dict(), f"ckpt_{tag}.pt")
    
    model.load_state_dict(torch.load(f"ckpt_{tag}.pt", map_location=DEVICE))
    return evaluate_vco(model, val_dl)

def train_ovcno(tag, use_obs_field=True, adaptive_beta=True, use_ranking=True):
    print(f"\n{'='*60}\nTraining: {tag} (obs={use_obs_field}, adapt_β={adaptive_beta}, rank={use_ranking})\n{'='*60}")
    train_ds = CopernicusOVCNODataset(NC_PATH, n_sensors=16, pts_per_sample=512, split="train")
    val_ds = CopernicusOVCNODataset(NC_PATH, n_sensors=16, pts_per_sample=2048, split="val")
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_ovcno, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_ovcno, num_workers=2)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_nll = float('inf')
    
    for ep in range(1, EPOCHS+1):
        model.train(); t0 = time.time(); ep_loss = 0
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(DEVICE), pts.to(DEVICE)
            trunk_flat = trunk.view(B*P, 4).to(DEVICE)
            labels_flat = labels.view(B*P, 1).to(DEVICE)
            d_s = trunk_flat[:, 3:4]
            y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk_flat)
            
            if not use_obs_field:
                # Zero out o_i so it has no effect on adaptive beta or decoder
                o_i_for_loss = torch.ones_like(o_i) * 0.5  # neutral
            else:
                o_i_for_loss = o_i
            
            # Choose beta parameters
            if adaptive_beta and use_obs_field:
                b0, b1 = 1e-4, 1e-3
            else:
                b0, b1 = 1e-3, 0.0  # fixed global beta
                
            lam = 1.0 if use_ranking else 0.0
            
            loss, lnll, lkl, lobs = compute_ovcno_loss(
                y_mu, y_logvar, labels_flat, mu_z, logvar_z, o_i_for_loss, d_s,
                beta_0=b0, beta_1=b1, lambda_obs=lam)
            
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep_loss += loss.item()
        sched.step()
        if ep % 10 == 0:
            metrics = evaluate_ovcno(model, val_dl)
            print(f"  Ep {ep} [{time.time()-t0:.1f}s] NLL={metrics['nll']:.4f} Cov={metrics['cov95']:.1f}%")
            if metrics['nll'] < best_nll:
                best_nll = metrics['nll']
                torch.save(model.state_dict(), f"ckpt_{tag}.pt")
    
    model.load_state_dict(torch.load(f"ckpt_{tag}.pt", map_location=DEVICE))
    return evaluate_ovcno(model, val_dl)

# ===================== main =====================
if __name__ == "__main__":
    results = {}
    
    # 1. VCO baseline (old architecture)
    results["VCO_baseline"] = train_vco("VCO_baseline")
    
    # 2. OVCNO w/o obs field (geometry encoder yes, but obs_net signal zeroed)
    results["OVCNO_no_obs"] = train_ovcno("OVCNO_no_obs", use_obs_field=False, adaptive_beta=False, use_ranking=False)
    
    # 3. OVCNO w/o adaptive β (obs field yes, but β is fixed globally)
    results["OVCNO_no_adapt"] = train_ovcno("OVCNO_no_adapt", use_obs_field=True, adaptive_beta=False, use_ranking=False)
    
    # 4. OVCNO w/o ranking loss (obs + adaptive β, but no ranking)
    results["OVCNO_no_rank"] = train_ovcno("OVCNO_no_rank", use_obs_field=True, adaptive_beta=True, use_ranking=False)
    
    # 5. Full OVCNO
    results["Full_OVCNO"] = train_ovcno("Full_OVCNO", use_obs_field=True, adaptive_beta=True, use_ranking=True)
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Variant':<30s} {'Rel-L2%':>8s} {'RMSE':>8s} {'MAE':>8s} {'NLL':>8s} {'Cov95%':>8s} {'Corr':>8s}")
    print("-"*80)
    for name, m in results.items():
        print(f"{name:<30s} {m['rel_l2']:8.2f} {m['rmse']:8.4f} {m['mae']:8.4f} {m['nll']:8.4f} {m['cov95']:8.2f} {m['corr']:8.4f}")
    
    # Save to JSON
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved ablation_results.json")
