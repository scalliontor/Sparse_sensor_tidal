import os
import math
import numpy as np
import torch
import collections
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_vae import CopernicusVAEDataset
from model_vae import ForecastDeepONetVAE

def collate_fn(batch):
    hists, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    n_sensors = hists[0].shape[1]
    padded = np.zeros((len(hists), T_max, n_sensors), dtype=np.float32)
    for i, h in enumerate(hists):
        padded[i, :h.shape[0]] = h
    return torch.tensor(padded), torch.stack(trunks), torch.stack(labels)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    val_ds = CopernicusVAEDataset(nc_path, split="val", pts_per_sample=2048)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=128, latent_dim=128).to(device)
    model.load_state_dict(torch.load("vae_checkpoint.pt", map_location=device))
    model.eval()

    # Need sensor positions for distance calculation
    # Just dummy coordinates for relative distance if abstract:
    H, W = 73, 61
    sensors = [(i, j) for i in [0, H-1] for j in np.linspace(0, W-1, 8, dtype=int)]
    if len(sensors) > 16: sensors = sensors[:16]

    all_errs = []
    all_stds = []
    all_dists = []
    all_mu = []
    all_y = []

    with torch.no_grad():
        for hist, trunk, labels in val_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, trunk, labels = hist.to(device), trunk.view(B*P, 3).to(device), labels.view(B*P, 1).to(device)

            pred_mu, pred_logvar, _, _ = model(hist, trunk)
            std = torch.exp(0.5 * pred_logvar)
            err = (pred_mu - labels).abs()
            
            all_errs.append(err.cpu().numpy())
            all_stds.append(std.cpu().numpy())
            all_mu.append(pred_mu.cpu().numpy())
            all_y.append(labels.cpu().numpy())

            # Distances: For each point, calculate min dist to sensor
            pts = trunk.cpu().numpy()  # [B*P, 3] -> (idx, x, y)
            for pt in pts:
                _, y_coord, x_coord = pt
                dist = min(math.sqrt((y_coord - sy)**2 + (x_coord - sx)**2) for sy, sx in sensors)
                all_dists.append(dist)

    all_errs = np.concatenate(all_errs).flatten()
    all_stds = np.concatenate(all_stds).flatten()
    all_mu = np.concatenate(all_mu).flatten()
    all_y = np.concatenate(all_y).flatten()
    all_dists = np.array(all_dists)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Uncertainty vs Distance
    # Bin by distance
    bins = np.linspace(0, max(all_dists), 20)
    bin_idxs = np.digitize(all_dists, bins)
    d_means, unc_means, err_means = [], [], []
    for i in range(1, len(bins)):
        mask = (bin_idxs == i)
        if mask.sum() > 0:
            d_means.append(bins[i])
            unc_means.append(all_stds[mask].mean())
            err_means.append(all_errs[mask].mean())

    plt.figure(figsize=(7,5))
    plt.plot(d_means, unc_means, 'bs-', label='Pred $\sigma$')
    plt.plot(d_means, err_means, 'rx-', label='Abs Error $|e|$')
    plt.xlabel("Distance to Nearest Sensor (cells)")
    plt.ylabel("Value (m)")
    plt.title("Metrics Degradation with Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("uncertainty_vs_distance.png", dpi=150)
    plt.close()

    # Plot 2: Error vs Uncertainty (Scatter/Bin)
    plt.figure(figsize=(7,5))
    # Sub-sample for scatter
    idx = np.random.choice(len(all_errs), min(5000, len(all_errs)), replace=False)
    plt.scatter(all_stds[idx], all_errs[idx], alpha=0.3, color='purple', s=4)
    # Regression line
    m, b = np.polyfit(all_stds[idx], all_errs[idx], 1)
    plt.plot(all_stds[idx], m*all_stds[idx] + b, color='red', label=f'Trend (slope={m:.2f})')
    plt.xlabel("Predictive Uncertainty ($\sigma$)")
    plt.ylabel("Realized Absolute Error ($|e|$)")
    plt.title("Error vs Uncertainty Consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_vs_uncertainty.png", dpi=150)
    plt.close()

    # Plot 3: Calibration Curve
    # Calculate coverage at various nominal confidence levels
    confidence_levels = np.linspace(0.1, 0.99, 10)
    empirical_coverage = []
    
    from scipy.stats import norm
    for conf in confidence_levels:
        Z = norm.ppf(0.5 + conf / 2.0)
        lower = all_mu - Z * all_stds
        upper = all_mu + Z * all_stds
        in_bound = ((all_y >= lower) & (all_y <= upper))
        empirical_coverage.append(in_bound.mean())

    plt.figure(figsize=(6,6))
    plt.plot(confidence_levels, empirical_coverage, 'go-', label='Forecast Model')
    plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
    plt.xlabel("Expected Confidence Level")
    plt.ylabel("Empirical Coverage Ratio")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_curve.png", dpi=150)
    plt.close()
    
    print("Done. Generated images.")

if __name__ == "__main__":
    main()
