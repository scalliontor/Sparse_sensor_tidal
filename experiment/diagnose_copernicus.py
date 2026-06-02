"""
Quick diagnostic: check if Copernicus OVCNO Corr_S is also inflated by mean collapse.
Uses batch_size=1 to handle variable T_obs lengths.
"""
import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE

NC_PATH = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
device = "cuda" if torch.cuda.is_available() else "cpu"


def diagnose_model(model, val_ds, model_type, name, device):
    """Compute diagnostics one sample at a time (handles variable T_obs)."""
    model.eval()
    
    all_mu, all_lab, all_lv = [], [], []
    pred_stds, gt_stds = [], []
    
    with torch.no_grad():
        for i in range(len(val_ds)):
            hist, pts, trunk, labels = val_ds[i]
            # Add batch dim
            hist = hist.unsqueeze(0).to(device)    # (1, T, K)
            pts = pts.unsqueeze(0).to(device)       # (1, K, 2)
            P = trunk.shape[0]
            trunk = trunk.to(device)                 # (P, 4)
            labels = labels.to(device)               # (P,)
            
            if model_type == "ovcno":
                y_mu, y_logvar, _, _, _ = model(hist, pts, trunk)
            else:
                y_mu, y_logvar, _, _ = model(hist, trunk[:, :3])
            
            y_mu_np = y_mu.cpu().numpy().flatten()
            y_lv_np = y_logvar.cpu().numpy().flatten()
            lab_np = labels.cpu().numpy().flatten()
            
            all_mu.append(y_mu_np)
            all_lab.append(lab_np)
            all_lv.append(y_lv_np)
            
            pred_stds.append(np.std(y_mu_np))
            gt_stds.append(np.std(lab_np))
    
    mu = np.concatenate(all_mu)
    lab = np.concatenate(all_lab)
    lv = np.concatenate(all_lv)
    std = np.exp(0.5 * lv)
    
    rmse = float(np.sqrt(np.mean((mu - lab)**2)))
    errs = np.abs(mu - lab)
    cs, _ = spearmanr(errs, std)
    
    z95 = 1.96
    cov = float(np.mean((lab >= mu - z95*std) & (lab <= mu + z95*std)))
    avgw = float(np.mean(2 * z95 * std))
    
    pred_mean_std = np.mean(pred_stds)
    gt_mean_std = np.mean(gt_stds)
    ratio = pred_mean_std / (gt_mean_std + 1e-8)
    
    print(f"\n  {name}:")
    print(f"    RMSE={rmse:.6f}  Corr_S={cs:.4f}")
    print(f"    Cov@95={cov*100:.1f}%  AvgW={avgw:.4f}")
    print(f"    SpatStdRatio={ratio:.4f} (pred={pred_mean_std:.6f}, gt={gt_mean_std:.6f})")
    print(f"    MeanLogvar={np.mean(lv):.3f}  MeanStd={np.mean(std):.6f}")
    
    return {
        "name": name, "rmse": rmse, "corr_s": float(cs),
        "cov95": cov, "avgw": avgw, "ratio": ratio,
        "pred_std": pred_mean_std, "gt_std": gt_mean_std,
        "mean_logvar": float(np.mean(lv))
    }


# ─── Load dataset ───
print("Loading Copernicus dataset...")
train_ds = CopernicusOVCNODataset(
    NC_PATH, n_sensors=16, pts_per_sample=512,
    T_obs_min=24, T_obs_max=72, seed=42, split="train"
)
val_ds = CopernicusOVCNODataset(
    NC_PATH, n_sensors=16, pts_per_sample=512,
    T_obs_min=24, T_obs_max=72, seed=42, split="val",
    train_mean=train_ds.train_mean
)

results = []

# ─── OVCNO checkpoints ───
ovcno_ckpts = {
    "Full_OVCNO": "ckpt_Full_OVCNO.pt",
    "OVCNO_no_adapt": "ckpt_OVCNO_no_adapt.pt",
    "OVCNO_no_rank": "ckpt_OVCNO_no_rank.pt",
    "OVCNO_no_obs": "ckpt_OVCNO_no_obs.pt",
    "OVCNO_original": "ovcno_checkpoint.pt",
}

for name, ckpt in ovcno_ckpts.items():
    if os.path.exists(ckpt):
        try:
            model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            r = diagnose_model(model, val_ds, "ovcno", name, device)
            results.append(r)
        except Exception as e:
            print(f"  Failed {name}: {e}")

# ─── VCO baseline ───
vco_ckpts = {
    "VCO_baseline": "ckpt_VCO_baseline.pt",
    "VCO_original": "vae_checkpoint.pt",
}

for name, ckpt in vco_ckpts.items():
    if os.path.exists(ckpt):
        try:
            model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=256, latent_dim=256, width=256).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            r = diagnose_model(model, val_ds, "vco", name, device)
            results.append(r)
        except Exception as e:
            print(f"  Failed {name}: {e}")

# ─── Summary ───
print(f"\n{'='*80}")
print("COPERNICUS DIAGNOSTIC SUMMARY")
print(f"{'='*80}")
print(f"{'Checkpoint':<22} {'RMSE':>8} {'CorrS':>7} {'Cov95':>7} {'AvgW':>7} {'StdR':>7}")
print("-"*80)
for r in results:
    print(f"{r['name']:<22} {r['rmse']:>8.5f} {r['corr_s']:>7.4f} "
          f"{r['cov95']*100:>6.1f}% {r['avgw']:>7.4f} {r['ratio']:>7.4f}")
print(f"{'='*80}")

with open("copernicus_diagnostic.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: copernicus_diagnostic.json")
