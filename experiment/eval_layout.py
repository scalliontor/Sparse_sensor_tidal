"""
Evaluate OVCNO checkpoints trained with different sensor layouts.
Reports: RMSE, MAE, NLL, CRPS, Avg.W, Cov@95%, Corr, Spearman.

Usage:
    python eval_layout.py --layout sensors_real_stations.json --ckpt ckpt_real_k12.pt --name real_k12
    python eval_layout.py --layout sensors_equispaced.json --ckpt ckpt_equispaced_k12.pt --name equispaced_k12
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_ovcno_layout import OVCNOLayoutDataset
from model_ovcno import ObservabilityAwareVCNO
from scipy import stats

def collate_fn(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)

def crps_gaussian(mu, sigma, y):
    """CRPS for Gaussian distribution."""
    z = (y - mu) / sigma
    phi_z = stats.norm.cdf(z)
    pdf_z = stats.norm.pdf(z)
    return sigma * (z * (2 * phi_z - 1) + 2 * pdf_z - 1.0 / np.sqrt(np.pi))

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Evaluating: {args.name} on {device} ===")
    
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    
    # Compute train mean first, then pass to val (no leakage)
    train_ds = OVCNOLayoutDataset(nc_path, args.layout, pts_per_sample=512,
                                   T_obs=24, split="train", seed=42)
    train_mean = train_ds.train_mean  # (Ny, Nx)
    del train_ds
    
    val_ds = OVCNOLayoutDataset(nc_path, args.layout, pts_per_sample=-1,
                                 T_obs=24, split="val", seed=999,
                                 train_mean=train_mean)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    N_MC = 50
    all_mu = []
    all_var = []
    all_labels = []
    all_obs = []
    all_ds = []
    
    with torch.no_grad():
        for hist, pts, trunk, labels in val_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B*P, 4).to(device)
            
            # Monte Carlo sampling with latent stochasticity
            mc_preds = []
            mc_vars = []
            mc_obs = []
            for _ in range(N_MC):
                y_mu, y_logvar, _, _, o_i = model(hist, pts, trunk_flat, sample_z=True)
                mc_preds.append(y_mu.cpu().numpy())
                mc_vars.append(np.exp(y_logvar.cpu().numpy()))
                mc_obs.append(o_i.cpu().numpy())
            
            mc_preds = np.array(mc_preds)  # (MC, B*P, 1)
            mc_vars = np.array(mc_vars)
            
            # Predictive mean and variance (law of total variance)
            pred_mean = mc_preds.mean(axis=0)
            pred_var = mc_vars.mean(axis=0) + mc_preds.var(axis=0)
            obs_mean = np.array(mc_obs).mean(axis=0)
            
            all_mu.append(pred_mean.squeeze())
            all_var.append(pred_var.squeeze())
            all_labels.append(labels.view(-1).numpy())
            all_obs.append(obs_mean.squeeze())
            all_ds.append(trunk[:, :, 3].view(-1).numpy())
    
    mu = np.concatenate(all_mu)
    var = np.concatenate(all_var)
    y = np.concatenate(all_labels)
    obs = np.concatenate(all_obs)
    ds = np.concatenate(all_ds)
    sigma = np.sqrt(var)
    
    # Metrics
    errors = mu - y
    abs_err = np.abs(errors)
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(abs_err)
    
    # NLL (Gaussian)
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + errors**2 / var)
    
    # CRPS
    crps = np.mean(crps_gaussian(mu, sigma, y))
    
    # Coverage
    z95 = 1.96
    lower = mu - z95 * sigma
    upper = mu + z95 * sigma
    covered = ((y >= lower) & (y <= upper))
    cov95 = np.mean(covered) * 100
    
    # Average interval width
    avg_w = np.mean(upper - lower)
    
    # Correlation (Pearson) between |error| and sigma
    corr_pearson = np.corrcoef(abs_err, sigma)[0, 1]
    corr_spearman, _ = stats.spearmanr(abs_err, sigma)
    
    print(f"\n{'='*50}")
    print(f"Layout: {args.name}")
    print(f"{'='*50}")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  NLL:       {nll:.4f}")
    print(f"  CRPS:      {crps:.4f}")
    print(f"  Avg.W:     {avg_w:.4f}")
    print(f"  Cov@95%:   {cov95:.1f}%")
    print(f"  Corr(P):   {corr_pearson:.4f}")
    print(f"  Corr(S):   {corr_spearman:.4f}")
    print(f"{'='*50}")
    
    # Save results
    results = {
        "name": args.name,
        "rmse": float(rmse), "mae": float(mae), "nll": float(nll),
        "crps": float(crps), "avg_w": float(avg_w), "cov95": float(cov95),
        "corr_pearson": float(corr_pearson), "corr_spearman": float(corr_spearman),
    }
    import json
    out_path = f"results_{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
