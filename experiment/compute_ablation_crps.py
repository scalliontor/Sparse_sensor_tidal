"""Compute CRPS and Avg.W for all 5 ablation variants."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import norm

from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE
from dataset_vae import CopernicusVAEDataset

NC_PATH = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def collate_ovcno(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)

def collate_vco(batch):
    hists, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    n_sensors = hists[0].shape[1]
    padded = np.zeros((len(hists), T_max, n_sensors), dtype=np.float32)
    for i, h in enumerate(hists):
        padded[i, :h.shape[0]] = h
    return torch.tensor(padded), torch.stack(trunks), torch.stack(labels)

def gaussian_crps(mu, sigma, y):
    z = (y - mu) / sigma
    return np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)))

def eval_ovcno(ckpt, tag):
    val_ds = CopernicusOVCNODataset(NC_PATH, n_sensors=16, pts_per_sample=2048, split="val")
    dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_ovcno, num_workers=2)
    m = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
    mus, lvs, ys = [], [], []
    with torch.no_grad():
        for h, p, t, l in dl:
            B, P = t.shape[0], t.shape[1]
            h, p = h.to(DEVICE), p.to(DEVICE)
            tf = t.view(B*P,4).to(DEVICE); lf = l.view(B*P,1).to(DEVICE)
            ym, ylv, _, _, _ = m(h, p, tf)
            mus.append(ym.cpu()); lvs.append(ylv.cpu()); ys.append(lf.cpu())
    mu = torch.cat(mus).numpy().flatten()
    lv = torch.cat(lvs).numpy().flatten()
    y = torch.cat(ys).numpy().flatten()
    sig = np.sqrt(np.exp(lv))
    crps = gaussian_crps(mu, sig, y)
    w = np.mean(2 * norm.ppf(0.975) * sig)
    print(f"{tag:<40s} CRPS={crps:.6f}  Avg.W={w:.4f}")

def eval_vco(ckpt, tag):
    val_ds = CopernicusVAEDataset(NC_PATH, n_sensors=16, pts_per_sample=2048, split="val")
    dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_vco, num_workers=2)
    m = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=256, latent_dim=256).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
    mus, lvs, ys = [], [], []
    with torch.no_grad():
        for h, t, l in dl:
            B, P = t.shape[0], t.shape[1]
            h = h.to(DEVICE); tf = t.view(B*P,3).to(DEVICE); lf = l.view(B*P,1).to(DEVICE)
            ym, ylv, _, _ = m(h, tf)
            mus.append(ym.cpu()); lvs.append(ylv.cpu()); ys.append(lf.cpu())
    mu = torch.cat(mus).numpy().flatten()
    lv = torch.cat(lvs).numpy().flatten()
    y = torch.cat(ys).numpy().flatten()
    sig = np.sqrt(np.exp(lv))
    crps = gaussian_crps(mu, sig, y)
    w = np.mean(2 * norm.ppf(0.975) * sig)
    print(f"{tag:<40s} CRPS={crps:.6f}  Avg.W={w:.4f}")

if __name__ == "__main__":
    eval_vco("ckpt_VCO_baseline.pt", "VCO baseline")
    eval_ovcno("ckpt_OVCNO_no_obs.pt", "OVCNO-Geom (no obs)")
    eval_ovcno("ckpt_OVCNO_no_adapt.pt", "OVCNO (proposed)")
    eval_ovcno("ckpt_OVCNO_no_rank.pt", "OVCNO + adaptive β")
    eval_ovcno("ckpt_Full_OVCNO.pt", "OVCNO + adaptive β + rank")
