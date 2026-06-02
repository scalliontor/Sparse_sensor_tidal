"""
Recompute Table 5 metrics with correct train-only normalization.
Tests all available checkpoints.
"""
import torch, numpy as np
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE
from scipy.stats import spearmanr, norm

device = "cuda" if torch.cuda.is_available() else "cpu"
nc = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"

train_ds = CopernicusOVCNODataset(nc, n_sensors=16, pts_per_sample=512,
    T_obs_min=24, T_obs_max=72, seed=42, split="train")
val_ds = CopernicusOVCNODataset(nc, n_sensors=16, pts_per_sample=512,
    T_obs_min=24, T_obs_max=72, seed=42, split="val",
    train_mean=train_ds.train_mean)

def eval_model(model, model_type, val_ds, device):
    model.eval()
    all_mu, all_lab, all_lv = [], [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            h, p, t, l = val_ds[i]
            h = h.unsqueeze(0).to(device)
            p = p.unsqueeze(0).to(device)
            t = t.to(device)
            l = l.numpy()
            if model_type == "ovcno":
                mu, lv, _, _, _ = model(h, p, t)
            else:
                mu, lv, _, _ = model(h, t[:, :3])
            all_mu.append(mu.cpu().numpy().flatten())
            all_lab.append(l.flatten())
            all_lv.append(lv.cpu().numpy().flatten())
    
    mu = np.concatenate(all_mu)
    lab = np.concatenate(all_lab)
    lv = np.concatenate(all_lv)
    std = np.exp(0.5 * lv)
    
    rmse = np.sqrt(np.mean((mu - lab)**2))
    mae = np.mean(np.abs(mu - lab))
    nll = 0.5 * np.mean(lv + (lab - mu)**2 / np.exp(lv))
    z95 = 1.96
    cov = np.mean((lab >= mu - z95*std) & (lab <= mu + z95*std))
    avgw = np.mean(2 * z95 * std)
    errs = np.abs(mu - lab)
    cs, _ = spearmanr(errs, std)
    # CRPS (Gaussian approx)
    z = (lab - mu) / std
    crps = np.mean(std * (z * (2*norm.cdf(z) - 1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi)))
    
    return rmse, mae, nll, crps, avgw, cov, cs

print("=" * 90)
print("CORRECTED TABLE 5 METRICS (train-only normalization)")
print("=" * 90)
print(f"{'Checkpoint':<25} {'RMSE':>7} {'MAE':>7} {'NLL':>7} {'CRPS':>7} {'AvgW':>7} {'Cov95':>7} {'CorrS':>7}")
print("-" * 90)

# OVCNO checkpoints
for name, ckpt in [("ovcno_checkpoint", "ovcno_checkpoint.pt"),
                    ("Full_OVCNO (ablation)", "ckpt_Full_OVCNO.pt"),
                    ("OVCNO_no_rank", "ckpt_OVCNO_no_rank.pt"),
                    ("OVCNO_no_obs", "ckpt_OVCNO_no_obs.pt"),
                    ("OVCNO_no_adapt", "ckpt_OVCNO_no_adapt.pt")]:
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    rmse, mae, nll, crps, avgw, cov, cs = eval_model(model, "ovcno", val_ds, device)
    print(f"{name:<25} {rmse:>7.4f} {mae:>7.4f} {nll:>7.3f} {crps:>7.4f} {avgw:>7.3f} {cov*100:>6.1f}% {cs:>7.3f}")

# VCO
model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=256, latent_dim=256, width=256).to(device)
model.load_state_dict(torch.load("ckpt_VCO_baseline.pt", map_location=device))
rmse, mae, nll, crps, avgw, cov, cs = eval_model(model, "vco", val_ds, device)
print(f"{'VCO_baseline':<25} {rmse:>7.4f} {mae:>7.4f} {nll:>7.3f} {crps:>7.4f} {avgw:>7.3f} {cov*100:>6.1f}% {cs:>7.3f}")

print("=" * 90)
