"""
Missing-sensor robustness evaluation (v2).
Masks dropped sensors to zero (keeps fixed K for batching).
"""
import argparse
import numpy as np
import torch
import json
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
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))

def mask_sensors(sensor_hist, sensor_pts, trunk, n_keep, rng):
    """Zero-mask dropped sensors while keeping tensor shapes fixed.
    Also recomputes d_s in trunk based on only the kept sensors."""
    B, T, K = sensor_hist.shape
    P = trunk.shape[1]
    
    hist_out = sensor_hist.clone()
    pts_out = sensor_pts.clone()
    trunk_out = trunk.clone()
    
    for b in range(B):
        keep_idx = np.sort(rng.choice(K, n_keep, replace=False))
        drop_mask = np.ones(K, dtype=bool)
        drop_mask[keep_idx] = False
        
        # Zero out dropped sensor histories and move coords to far-away
        hist_out[b, :, drop_mask] = 0.0
        pts_out[b, drop_mask, :] = 99.0  # far away so they don't affect d_s
        
        # Recompute d_s based on kept sensors only
        kept_pts = sensor_pts[b, keep_idx]  # (n_keep, 2)
        qx = trunk[b, :, 0:1]  # (P, 1)
        qy = trunk[b, :, 1:2]
        sx = kept_pts[:, 0:1].T  # (1, n_keep)
        sy = kept_pts[:, 1:2].T
        dists = torch.sqrt((qx - sx)**2 + (qy - sy)**2)  # (P, n_keep)
        trunk_out[b, :, 3] = dists.min(dim=1).values
    
    return hist_out, pts_out, trunk_out

def eval_missing(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    
    # Compute train mean first, then pass to val (no leakage)
    train_ds = OVCNOLayoutDataset(nc_path, args.layout, pts_per_sample=512,
                                   T_obs=24, split="train", seed=42)
    train_mean = train_ds.train_mean
    del train_ds
    
    val_ds = OVCNOLayoutDataset(nc_path, args.layout, pts_per_sample=-1,
                                 T_obs=24, split="val", seed=999,
                                 train_mean=train_mean)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    K = val_ds.K
    N_MC = 50
    
    print(f"Layout: {args.name}, K={K}")
    
    for frac in [1.0, 0.75, 0.5]:
        n_keep = max(3, round(K * frac))
        n_masks = 1 if frac == 1.0 else 10
        
        mask_results = {m: [] for m in ['rmse', 'crps', 'cov', 'corr', 'avgw']}
        
        for mask_i in range(n_masks):
            rng = np.random.default_rng(mask_i * 100 + 42)
            
            all_mu, all_var, all_labels = [], [], []
            
            with torch.no_grad():
                for hist, pts, trunk, labels in val_dl:
                    B, P = trunk.shape[0], trunk.shape[1]
                    
                    if frac < 1.0:
                        hist_m, pts_m, trunk_m = mask_sensors(hist, pts, trunk, n_keep, rng)
                    else:
                        hist_m, pts_m, trunk_m = hist, pts, trunk
                    
                    hist_m = hist_m.to(device)
                    pts_m = pts_m.to(device)
                    trunk_flat = trunk_m.view(B*P, 4).to(device)
                    
                    mc_preds, mc_vars = [], []
                    for _ in range(N_MC):
                        y_mu, y_logvar, _, _, _ = model(hist_m, pts_m, trunk_flat, sample_z=True)
                        mc_preds.append(y_mu.cpu().numpy())
                        mc_vars.append(np.exp(y_logvar.cpu().numpy()))
                    
                    mc_preds = np.array(mc_preds)
                    mc_vars = np.array(mc_vars)
                    pred_mean = mc_preds.mean(axis=0)
                    pred_var = mc_vars.mean(axis=0) + mc_preds.var(axis=0)
                    
                    all_mu.append(pred_mean.squeeze())
                    all_var.append(pred_var.squeeze())
                    all_labels.append(labels.view(-1).numpy())
            
            mu = np.concatenate(all_mu)
            var = np.concatenate(all_var)
            y = np.concatenate(all_labels)
            sigma = np.sqrt(var)
            errors = mu - y
            abs_err = np.abs(errors)
            
            rmse = float(np.sqrt(np.mean(errors**2)))
            crps = float(np.mean(crps_gaussian(mu, sigma, y)))
            lower = mu - 1.96 * sigma
            upper = mu + 1.96 * sigma
            cov = float(np.mean((y >= lower) & (y <= upper)) * 100)
            avgw = float(np.mean(upper - lower))
            corr_s, _ = stats.spearmanr(abs_err, sigma)
            
            mask_results['rmse'].append(rmse)
            mask_results['crps'].append(crps)
            mask_results['cov'].append(cov)
            mask_results['avgw'].append(avgw)
            mask_results['corr'].append(float(corr_s))
        
        avail_pct = int(frac * 100)
        if n_masks > 1:
            print(f"  {avail_pct:3d}%  K={n_keep:2d}  "
                  f"RMSE={np.mean(mask_results['rmse']):.4f}±{np.std(mask_results['rmse']):.4f}  "
                  f"CRPS={np.mean(mask_results['crps']):.4f}±{np.std(mask_results['crps']):.4f}  "
                  f"Cov={np.mean(mask_results['cov']):.1f}±{np.std(mask_results['cov']):.1f}%  "
                  f"Avg.W={np.mean(mask_results['avgw']):.3f}±{np.std(mask_results['avgw']):.3f}  "
                  f"Corr(S)={np.mean(mask_results['corr']):.3f}±{np.std(mask_results['corr']):.3f}")
        else:
            print(f"  {avail_pct:3d}%  K={n_keep:2d}  "
                  f"RMSE={mask_results['rmse'][0]:.4f}  "
                  f"CRPS={mask_results['crps'][0]:.4f}  "
                  f"Cov={mask_results['cov'][0]:.1f}%  "
                  f"Avg.W={mask_results['avgw'][0]:.3f}  "
                  f"Corr(S)={mask_results['corr'][0]:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--name", type=str, default="missing")
    args = parser.parse_args()
    eval_missing(args)
