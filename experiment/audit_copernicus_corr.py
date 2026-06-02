"""
Deep Copernicus Corr_S Audit — 5 diagnostics to verify Corr_S is genuine.

1. SpatialStdRatio: std(pred_mean) / std(gt) — detect mean collapse
2. Constant-mean baseline: Corr_S with pred_mean = 0, sigma = distance
3. Partial correlations: disentangle error, sigma, distance, SSH amplitude
4. Conditional Corr_S on well-predicted samples only
5. Reliability by sigma bins: does high-sigma actually have high error?
"""
import os, json
import numpy as np
import torch
from scipy.stats import spearmanr
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE

NC_PATH = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load data ───
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

# ─── Collect predictions ───
def collect_predictions(model, dataset, model_type, device, n_samples=None):
    """Collect per-point predictions with full metadata."""
    model.eval()
    if n_samples is None:
        n_samples = len(dataset)
    
    all_data = {
        'mu': [], 'lab': [], 'logvar': [], 'd_s': [],
        'pred_spatial_stds': [], 'gt_spatial_stds': [],
        'sample_rmse': []
    }
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            hist, pts, trunk, labels = dataset[i]
            hist = hist.unsqueeze(0).to(device)
            pts = pts.unsqueeze(0).to(device)
            trunk_gpu = trunk.to(device)
            labels_gpu = labels.to(device)
            
            if model_type == "ovcno":
                y_mu, y_logvar, _, _, o_i = model(hist, pts, trunk_gpu)
            else:
                y_mu, y_logvar, _, _ = model(hist, trunk_gpu[:, :3])
            
            mu_np = y_mu.cpu().numpy().flatten()
            lv_np = y_logvar.cpu().numpy().flatten()
            lab_np = labels.cpu().numpy().flatten()
            ds_np = trunk[:, 3].numpy()  # distance to nearest sensor
            
            all_data['mu'].append(mu_np)
            all_data['lab'].append(lab_np)
            all_data['logvar'].append(lv_np)
            all_data['d_s'].append(ds_np)
            all_data['pred_spatial_stds'].append(np.std(mu_np))
            all_data['gt_spatial_stds'].append(np.std(lab_np))
            all_data['sample_rmse'].append(np.sqrt(np.mean((mu_np - lab_np)**2)))
    
    # Concatenate
    result = {
        'mu': np.concatenate(all_data['mu']),
        'lab': np.concatenate(all_data['lab']),
        'logvar': np.concatenate(all_data['logvar']),
        'd_s': np.concatenate(all_data['d_s']),
        'pred_spatial_stds': np.array(all_data['pred_spatial_stds']),
        'gt_spatial_stds': np.array(all_data['gt_spatial_stds']),
        'sample_rmse': np.array(all_data['sample_rmse']),
    }
    result['std'] = np.exp(0.5 * result['logvar'])
    result['err'] = np.abs(result['mu'] - result['lab'])
    return result


def run_audit(name, data):
    """Run all 5 diagnostics on collected predictions."""
    mu, lab, std, err = data['mu'], data['lab'], data['std'], data['err']
    d_s = data['d_s']
    
    print(f"\n{'='*70}")
    print(f"  AUDIT: {name}")
    print(f"{'='*70}")
    
    # ─── 0. Baseline metrics ───
    rmse = np.sqrt(np.mean((mu - lab)**2))
    z95 = 1.96
    cov = np.mean((lab >= mu - z95*std) & (lab <= mu + z95*std))
    avgw = np.mean(2 * z95 * std)
    corr_s, _ = spearmanr(err, std)
    
    print(f"\n  [0] Baseline:  RMSE={rmse:.5f}  Corr_S={corr_s:.4f}  "
          f"Cov@95={cov*100:.1f}%  AvgW={avgw:.4f}")
    
    # ─── 1. SpatialStdRatio ───
    pred_std_mean = np.mean(data['pred_spatial_stds'])
    gt_std_mean = np.mean(data['gt_spatial_stds'])
    ratio = pred_std_mean / (gt_std_mean + 1e-8)
    
    print(f"\n  [1] SpatialStdRatio = {ratio:.4f}")
    print(f"      pred_spatial_std = {pred_std_mean:.6f}")
    print(f"      gt_spatial_std   = {gt_std_mean:.6f}")
    if ratio < 0.3:
        print(f"      ⚠️  MEAN COLLAPSE DETECTED (ratio < 0.3)")
    elif ratio < 0.6:
        print(f"      ⚠️  Partial mean collapse (ratio < 0.6)")
    else:
        print(f"      ✅  Mean predictor has spatial structure")
    
    # ─── 2. Constant-mean baseline Corr ───
    # If pred_mean was 0 everywhere, error = |gt|
    gt_abs = np.abs(lab)
    corr_const_mean_vs_std, _ = spearmanr(gt_abs, std)
    corr_const_mean_vs_dist, _ = spearmanr(gt_abs, d_s)
    corr_dist_vs_std, _ = spearmanr(d_s, std)
    
    print(f"\n  [2] Constant-mean baseline correlations:")
    print(f"      Corr(|gt|, σ)     = {corr_const_mean_vs_std:.4f}  "
          f"(would Corr_S be high with flat mean?)")
    print(f"      Corr(|gt|, d_s)   = {corr_const_mean_vs_dist:.4f}  "
          f"(does SSH amplitude correlate with sensor distance?)")
    print(f"      Corr(d_s, σ)      = {corr_dist_vs_std:.4f}  "
          f"(does uncertainty track distance?)")
    
    if abs(corr_const_mean_vs_std) > 0.4:
        print(f"      ⚠️  |gt| correlates with σ — Corr_S may be inflated")
    else:
        print(f"      ✅  |gt| does NOT strongly correlate with σ")
    
    # ─── 3. Partial correlations ───
    corr_err_std, _ = spearmanr(err, std)
    corr_err_ds, _ = spearmanr(err, d_s)
    corr_std_ds, _ = spearmanr(std, d_s)
    corr_err_gtabs, _ = spearmanr(err, gt_abs)
    
    print(f"\n  [3] Correlation matrix:")
    print(f"      Corr(|ε|, σ)     = {corr_err_std:.4f}   ← this is Corr_S (main claim)")
    print(f"      Corr(|ε|, d_s)   = {corr_err_ds:.4f}   ← error vs distance")
    print(f"      Corr(σ, d_s)     = {corr_std_ds:.4f}   ← uncertainty vs distance")
    print(f"      Corr(|ε|, |gt|)  = {corr_err_gtabs:.4f}   ← error vs SSH amplitude")
    
    # Partial correlation: Corr(|ε|, σ | d_s)
    # Using residuals approach
    from numpy.polynomial.polynomial import polyfit, polyval
    # Regress err and std on d_s, take residuals
    c_err = polyfit(d_s, err, 1)
    c_std = polyfit(d_s, std, 1)
    err_resid = err - polyval(d_s, c_err)
    std_resid = std - polyval(d_s, c_std)
    partial_corr, _ = spearmanr(err_resid, std_resid)
    
    print(f"      Partial Corr(|ε|, σ | d_s) = {partial_corr:.4f}  "
          f"← Corr after removing distance effect")
    
    if partial_corr > 0.2:
        print(f"      ✅  Uncertainty-error corr survives after controlling for distance")
    else:
        print(f"      ⚠️  Corr_S may be mostly driven by distance geometry")
    
    # ─── 4. Conditional Corr on well-predicted samples ───
    sample_rmses = data['sample_rmse']
    median_rmse = np.median(sample_rmses)
    
    # Re-collect per-sample Corr_S
    n_pts = 512  # per sample
    n_samples = len(sample_rmses)
    good_corrs, bad_corrs = [], []
    
    for i in range(n_samples):
        s, e = i * n_pts, (i + 1) * n_pts
        if e > len(err):
            break
        sample_err = err[s:e]
        sample_std = std[s:e]
        try:
            c, _ = spearmanr(sample_err, sample_std)
            if not np.isnan(c):
                if sample_rmses[i] <= median_rmse:
                    good_corrs.append(c)
                else:
                    bad_corrs.append(c)
        except:
            pass
    
    good_mean = np.mean(good_corrs) if good_corrs else float('nan')
    bad_mean = np.mean(bad_corrs) if bad_corrs else float('nan')
    
    print(f"\n  [4] Conditional Corr_S by sample quality:")
    print(f"      Good samples (RMSE ≤ median):   Corr_S = {good_mean:.4f}  (n={len(good_corrs)})")
    print(f"      Bad samples  (RMSE > median):    Corr_S = {bad_mean:.4f}  (n={len(bad_corrs)})")
    
    if good_mean > 0.2:
        print(f"      ✅  Corr_S remains positive even for well-predicted samples")
    else:
        print(f"      ⚠️  Corr_S driven by poorly-predicted samples")
    
    # ─── 5. Reliability by sigma bins ───
    n_bins = 5
    sigma_percentiles = np.percentile(std, np.linspace(0, 100, n_bins + 1))
    
    print(f"\n  [5] Reliability by predicted σ bins:")
    print(f"      {'Bin':<12} {'σ range':<22} {'MAE':>8} {'RMSE':>8} {'Coverage':>8}  {'n':>6}")
    
    for b in range(n_bins):
        lo, hi = sigma_percentiles[b], sigma_percentiles[b + 1]
        mask = (std >= lo) & (std < hi + 1e-10)
        if mask.sum() == 0:
            continue
        bin_err = err[mask]
        bin_mu = mu[mask]
        bin_lab = lab[mask]
        bin_std = std[mask]
        
        bin_mae = np.mean(bin_err)
        bin_rmse = np.sqrt(np.mean(bin_err**2))
        bin_cov = np.mean((bin_lab >= bin_mu - z95*bin_std) & (bin_lab <= bin_mu + z95*bin_std))
        
        label = f"Q{b*20}-Q{(b+1)*20}"
        print(f"      {label:<12} [{lo:.4f}, {hi:.4f}]   {bin_mae:>8.5f} {bin_rmse:>8.5f} "
              f"{bin_cov*100:>7.1f}%  {mask.sum():>6}")
    
    # Check monotonicity: does higher sigma = higher error?
    bin_maes = []
    for b in range(n_bins):
        lo, hi = sigma_percentiles[b], sigma_percentiles[b + 1]
        mask = (std >= lo) & (std < hi + 1e-10)
        if mask.sum() > 0:
            bin_maes.append(np.mean(err[mask]))
    
    is_monotonic = all(bin_maes[i] <= bin_maes[i+1] for i in range(len(bin_maes)-1))
    print(f"\n      MAE monotonicity across bins: {'✅ YES' if is_monotonic else '⚠️ NO'}")
    print(f"      Bin MAEs: {[f'{m:.5f}' for m in bin_maes]}")
    
    return {
        'name': name, 'rmse': float(rmse), 'corr_s': float(corr_s),
        'cov95': float(cov), 'avgw': float(avgw),
        'spatial_std_ratio': float(ratio),
        'corr_const_mean_vs_std': float(corr_const_mean_vs_std),
        'corr_dist_vs_std': float(corr_dist_vs_std),
        'partial_corr_err_std_given_ds': float(partial_corr),
        'good_sample_corr': float(good_mean),
        'bad_sample_corr': float(bad_mean),
        'mae_monotonic': is_monotonic,
        'bin_maes': [float(m) for m in bin_maes],
    }


# ─── Run audit on main checkpoints ───
results = []

# OVCNO (the one used in paper Table 5)
print("\n" + "="*70)
print("Loading OVCNO (Full_OVCNO checkpoint)...")
ovcno = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
ovcno.load_state_dict(torch.load("ckpt_Full_OVCNO.pt", map_location=device))
ovcno_data = collect_predictions(ovcno, val_ds, "ovcno", device)
r_ovcno = run_audit("Full_OVCNO", ovcno_data)
results.append(r_ovcno)

# VCO baseline
print("\n" + "="*70)
print("Loading VCO baseline...")
vco = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=256, latent_dim=256, width=256).to(device)
vco.load_state_dict(torch.load("ckpt_VCO_baseline.pt", map_location=device))
vco_data = collect_predictions(vco, val_ds, "vco", device)
r_vco = run_audit("VCO_baseline", vco_data)
results.append(r_vco)

# ─── Final Verdict ───
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

ovcno_ok = (r_ovcno['spatial_std_ratio'] > 0.6 and 
            r_ovcno['partial_corr_err_std_given_ds'] > 0.15 and
            r_ovcno['good_sample_corr'] > 0.1 and
            r_ovcno['mae_monotonic'])

if ovcno_ok:
    print("✅ Copernicus OVCNO Corr_S appears GENUINE:")
    print("   - Mean predictor has spatial structure (StdRatio > 0.6)")
    print("   - Partial correlation survives after controlling for distance")
    print("   - Well-predicted samples still show positive Corr_S")
    print("   - Higher predicted σ → higher actual error (monotonic)")
else:
    print("⚠️  Copernicus OVCNO Corr_S has CONCERNS:")
    if r_ovcno['spatial_std_ratio'] < 0.6:
        print("   - Mean predictor shows collapse (StdRatio < 0.6)")
    if r_ovcno['partial_corr_err_std_given_ds'] < 0.15:
        print("   - Corr_S mostly driven by distance geometry")
    if r_ovcno['good_sample_corr'] < 0.1:
        print("   - Corr_S only present in poorly-predicted samples")
    if not r_ovcno['mae_monotonic']:
        print("   - σ bins do NOT monotonically predict error")

# Save
with open("copernicus_deep_audit.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: copernicus_deep_audit.json")
