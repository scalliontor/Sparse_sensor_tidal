"""
Compute diagnostic metrics for OVCNO paper:
1. CRPS (Continuous Ranked Probability Score)
2. Average prediction interval width (95%)
3. Observability diagnostic correlations: Corr(o, -|e|), Corr(o, -sigma), Spearman
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import norm, spearmanr

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
    """CRPS for Gaussian predictive distribution."""
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return np.mean(crps)

def compute_ovcno_diagnostics(ckpt_path, tag):
    val_ds = CopernicusOVCNODataset(NC_PATH, n_sensors=16, pts_per_sample=2048, split="val")
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_ovcno, num_workers=2)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    all_mu, all_logvar, all_y, all_oi = [], [], [], []
    with torch.no_grad():
        for hist, pts, trunk, labels in val_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(DEVICE), pts.to(DEVICE)
            trunk_flat = trunk.view(B*P, 4).to(DEVICE)
            labels_flat = labels.view(B*P, 1).to(DEVICE)
            y_mu, y_logvar, _, _, oi = model(hist, pts, trunk_flat)
            all_mu.append(y_mu.cpu()); all_logvar.append(y_logvar.cpu())
            all_y.append(labels_flat.cpu()); all_oi.append(oi.cpu())
    
    mu = torch.cat(all_mu).numpy().flatten()
    logvar = torch.cat(all_logvar).numpy().flatten()
    y = torch.cat(all_y).numpy().flatten()
    oi = torch.cat(all_oi).numpy().flatten()
    
    var = np.exp(logvar)
    sigma = np.sqrt(var)
    err = np.abs(mu - y)
    
    # 1. CRPS
    crps = gaussian_crps(mu, sigma, y)
    
    # 2. Average 95% interval width
    z95 = norm.ppf(0.975)
    width_95 = 2 * z95 * sigma
    avg_width = np.mean(width_95)
    
    # 3. Coverage 95%
    cov95 = np.mean((y >= mu - z95*sigma) & (y <= mu + z95*sigma)) * 100
    
    # 4. Observability diagnostics
    corr_o_neg_err = np.corrcoef(oi, -err)[0, 1]
    corr_o_neg_sig = np.corrcoef(oi, -sigma)[0, 1]
    spear_o_neg_err = spearmanr(oi, -err).statistic
    spear_o_neg_sig = spearmanr(oi, -sigma).statistic
    
    # 5. Standard metrics
    rmse = np.sqrt(np.mean((mu - y)**2))
    nll = np.mean(0.5 * (np.log(2*np.pi) + logvar + (mu - y)**2 / var))
    corr_err_sig = np.corrcoef(err, sigma)[0, 1]
    
    print(f"\n{'='*60}")
    print(f"Diagnostics for: {tag}")
    print(f"{'='*60}")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  NLL:            {nll:.4f}")
    print(f"  Cov@95%:        {cov95:.1f}%")
    print(f"  Corr(|e|,σ):    {corr_err_sig:.4f}")
    print(f"  CRPS:           {crps:.6f}")
    print(f"  Avg Width@95%:  {avg_width:.6f}")
    print(f"  --- Observability Diagnostics ---")
    print(f"  Pearson(o, -|e|):  {corr_o_neg_err:.4f}")
    print(f"  Pearson(o, -σ):    {corr_o_neg_sig:.4f}")
    print(f"  Spearman(o, -|e|): {spear_o_neg_err:.4f}")
    print(f"  Spearman(o, -σ):   {spear_o_neg_sig:.4f}")
    
    return {
        "rmse": rmse, "nll": nll, "cov95": cov95, "corr_err_sig": corr_err_sig,
        "crps": crps, "avg_width": avg_width,
        "pearson_o_neg_err": corr_o_neg_err, "pearson_o_neg_sig": corr_o_neg_sig,
        "spearman_o_neg_err": spear_o_neg_err, "spearman_o_neg_sig": spear_o_neg_sig,
    }

def compute_vco_diagnostics(ckpt_path, tag):
    val_ds = CopernicusVAEDataset(NC_PATH, n_sensors=16, pts_per_sample=2048, split="val")
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_vco, num_workers=2)
    
    model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=256, latent_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    all_mu, all_logvar, all_y = [], [], []
    with torch.no_grad():
        for hist, trunk, labels in val_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist = hist.to(DEVICE)
            trunk = trunk.view(B*P, 3).to(DEVICE)
            labels = labels.view(B*P, 1).to(DEVICE)
            y_mu, y_logvar, _, _ = model(hist, trunk)
            all_mu.append(y_mu.cpu()); all_logvar.append(y_logvar.cpu()); all_y.append(labels.cpu())
    
    mu = torch.cat(all_mu).numpy().flatten()
    logvar = torch.cat(all_logvar).numpy().flatten()
    y = torch.cat(all_y).numpy().flatten()
    
    var = np.exp(logvar)
    sigma = np.sqrt(var)
    err = np.abs(mu - y)
    
    crps = gaussian_crps(mu, sigma, y)
    z95 = norm.ppf(0.975)
    avg_width = np.mean(2 * z95 * sigma)
    cov95 = np.mean((y >= mu - z95*sigma) & (y <= mu + z95*sigma)) * 100
    rmse = np.sqrt(np.mean((mu - y)**2))
    nll = np.mean(0.5 * (np.log(2*np.pi) + logvar + (mu - y)**2 / var))
    corr_err_sig = np.corrcoef(err, sigma)[0, 1]
    
    print(f"\n{'='*60}")
    print(f"Diagnostics for: {tag}")
    print(f"{'='*60}")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  NLL:            {nll:.4f}")
    print(f"  Cov@95%:        {cov95:.1f}%")
    print(f"  Corr(|e|,σ):    {corr_err_sig:.4f}")
    print(f"  CRPS:           {crps:.6f}")
    print(f"  Avg Width@95%:  {avg_width:.6f}")

if __name__ == "__main__":
    # VCO baseline
    compute_vco_diagnostics("ckpt_VCO_baseline.pt", "VCO baseline")
    
    # OVCNO (core: geom + obs field)
    compute_ovcno_diagnostics("ckpt_OVCNO_no_adapt.pt", "OVCNO (proposed)")
    
    # OVCNO + adaptive beta (for comparison)
    compute_ovcno_diagnostics("ckpt_OVCNO_no_rank.pt", "OVCNO + adaptive β")
