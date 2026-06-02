"""
HYCOM OVCNO-v2 Decoupled Smoke Test
────────────────────────────────────
Tests the decoupled architecture (obs → variance only) on HYCOM.

Two training modes:
  Mode A: end-to-end decoupled (fast, tests if architecture alone fixes collapse)
  Mode B: two-stage
         Stage 1: train mean only (freeze obs/var heads), MSE loss
         Stage 2: freeze mean (or low-LR), train variance/obs with full NLL

Decision gate (pass if ALL met):
  RMSE ~= VCO        (0.045–0.050)
  StdRatio > 0.55     (no mean collapse)
  NLL  >= VCO - 0.2   (not much worse)
  Cov@95 85-98%       (not just from ultra-wide intervals)

Run:  python train_hycom_v2_smoke.py --mode A
      python train_hycom_v2_smoke.py --mode B
"""
import os, sys, time, math, json, argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, norm
from numpy.polynomial.polynomial import polyfit, polyval

from dataset_hycom import HYCOMOVCNODataset
from model_ovcno_v2 import OVCNOv2Decoupled

# ─── Paths ───
NC_PATH = "../data/hycom_data/hycom_ssh_tonkin_jan_sep_2024.nc"
STATION_JSON = "hycom_real_k12_stations.json"
SAVE_DIR = "hycom_v2_smoke"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Config ───
SEED = 42
EPOCHS_A = 50        # Mode A: end-to-end
EPOCHS_B1 = 25       # Mode B stage 1: mean-only
EPOCHS_B2 = 30       # Mode B stage 2: variance/obs
BATCH_SIZE = 8
LR = 5e-4
LR_STAGE2_MEAN = 1e-5   # low LR for mean head in stage 2
PTS_PER_SAMPLE = 512
T_OBS = 8  # 8×3h = 24h history


def collate_fn(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)


def compute_v2_loss(y_mu, y_logvar, target, mu_z, logvar_z, o_i, d_s,
                    beta=1e-3, lambda_rank=0.5, margin=0.1):
    """Combined loss for OVCNO-v2 (no adaptive beta — simpler)."""
    # 1. NLL
    precision = torch.exp(-y_logvar)
    sq_err = (target - y_mu) ** 2
    nll_pt = 0.5 * precision * sq_err + 0.5 * y_logvar
    L_nll = nll_pt.mean()

    # 2. KL
    kl = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()
    L_kl = beta * kl

    # 3. Ranking loss (optional, soft)
    B_times_P = y_mu.shape[0]
    num_pairs = min(500, B_times_P // 2)
    idx1 = torch.randperm(B_times_P, device=y_mu.device)[:num_pairs]
    idx2 = torch.randperm(B_times_P, device=y_mu.device)[:num_pairs]
    d1, d2 = d_s[idx1], d_s[idx2]
    sigma = torch.exp(0.5 * y_logvar)
    sig1, sig2 = sigma[idx1], sigma[idx2]
    mask = (d1 > d2 + 0.1).float()
    ranking_penalty = torch.relu(margin - (sig1 - sig2))
    L_rank = (ranking_penalty * mask).sum() / (mask.sum() + 1e-6)

    total = L_nll + L_kl + lambda_rank * L_rank
    return total, L_nll, L_kl, L_rank


def compute_mse_loss(y_mu, target, mu_z, logvar_z, beta=1e-3):
    """Stage 1 loss: MSE only (no variance)."""
    mse = ((target - y_mu) ** 2).mean()
    kl = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()
    return mse + beta * kl, mse


def evaluate_full(model, dataloader, device):
    """Collect predictions for full diagnostic suite."""
    model.eval()
    all_mu, all_lab, all_lv, all_ds = [], [], [], []
    per_sample_pred_stds, per_sample_gt_stds, per_sample_rmses = [], [], []

    with torch.no_grad():
        for hist, pts, trunk, labels in dataloader:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B * P, 4).to(device)
            labels_flat = labels.view(B * P, 1).to(device)

            y_mu, y_logvar, _, _, o_i = model(hist, pts, trunk_flat, sample_z=True)

            mu_np = y_mu.cpu().numpy().flatten()
            lv_np = y_logvar.cpu().numpy().flatten()
            lab_np = labels_flat.cpu().numpy().flatten()
            ds_np = trunk_flat[:, 3].cpu().numpy()

            all_mu.append(mu_np)
            all_lab.append(lab_np)
            all_lv.append(lv_np)
            all_ds.append(ds_np)

            # Per-sample stats
            for b in range(B):
                s, e = b * P, (b + 1) * P
                per_sample_pred_stds.append(np.std(mu_np[s:e]))
                per_sample_gt_stds.append(np.std(lab_np[s:e]))
                per_sample_rmses.append(np.sqrt(np.mean((mu_np[s:e] - lab_np[s:e])**2)))

    mu = np.concatenate(all_mu)
    lab = np.concatenate(all_lab)
    lv = np.concatenate(all_lv)
    d_s = np.concatenate(all_ds)
    std = np.exp(0.5 * lv)
    err = np.abs(mu - lab)

    return {
        'mu': mu, 'lab': lab, 'lv': lv, 'std': std, 'err': err, 'd_s': d_s,
        'pred_spatial_stds': np.array(per_sample_pred_stds),
        'gt_spatial_stds': np.array(per_sample_gt_stds),
        'sample_rmses': np.array(per_sample_rmses),
    }


def run_diagnostics(name, data):
    """Full 5-diagnostic suite (same as audit_copernicus_corr.py)."""
    mu, lab, std, err, d_s = data['mu'], data['lab'], data['std'], data['err'], data['d_s']

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {name}")
    print(f"{'='*70}")

    # 0. Baseline metrics
    rmse = np.sqrt(np.mean((mu - lab)**2))
    z95 = 1.96
    cov = np.mean((lab >= mu - z95*std) & (lab <= mu + z95*std))
    avgw = np.mean(2 * z95 * std)
    nll = 0.5 * np.mean(data['lv'] + (lab - mu)**2 / np.exp(data['lv']))
    cs, _ = spearmanr(err, std)
    z = (lab - mu) / std
    crps = np.mean(std * (z * (2*norm.cdf(z) - 1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi)))

    print(f"\n  [0] Metrics: RMSE={rmse:.5f}  NLL={nll:.3f}  CRPS={crps:.4f}  "
          f"Cov={cov*100:.1f}%  AvgW={avgw:.4f}  Corr_S={cs:.4f}")

    # 1. SpatialStdRatio
    pred_std_mean = np.mean(data['pred_spatial_stds'])
    gt_std_mean = np.mean(data['gt_spatial_stds'])
    ratio = pred_std_mean / (gt_std_mean + 1e-8)
    status1 = "✅" if ratio > 0.55 else ("⚠️ PARTIAL" if ratio > 0.3 else "❌ COLLAPSE")
    print(f"\n  [1] SpatialStdRatio = {ratio:.4f}  {status1}")
    print(f"      pred_std={pred_std_mean:.6f}  gt_std={gt_std_mean:.6f}")

    # 2. Constant-mean baseline
    gt_abs = np.abs(lab)
    corr_const, _ = spearmanr(gt_abs, std)
    corr_d_std, _ = spearmanr(d_s, std)
    print(f"\n  [2] Constant-mean baseline:")
    print(f"      Corr(|gt|, σ) = {corr_const:.4f}")
    print(f"      Corr(d_s, σ)  = {corr_d_std:.4f}")

    # 3. Partial correlation
    c_err = polyfit(d_s, err, 1)
    c_std_fit = polyfit(d_s, std, 1)
    err_resid = err - polyval(d_s, c_err)
    std_resid = std - polyval(d_s, c_std_fit)
    partial_corr, _ = spearmanr(err_resid, std_resid)
    status3 = "✅" if partial_corr > 0.15 else "⚠️"
    print(f"\n  [3] Partial Corr(|ε|, σ | d_s) = {partial_corr:.4f}  {status3}")

    # 4. Conditional Corr by sample quality
    P = 2048  # pts per sample in test
    n_samples = len(data['sample_rmses'])
    median_rmse = np.median(data['sample_rmses'])
    good_corrs, bad_corrs = [], []
    for i in range(n_samples):
        s, e = i * P, (i + 1) * P
        if e > len(err):
            break
        try:
            c, _ = spearmanr(err[s:e], std[s:e])
            if not np.isnan(c):
                if data['sample_rmses'][i] <= median_rmse:
                    good_corrs.append(c)
                else:
                    bad_corrs.append(c)
        except:
            pass
    good_mean = np.mean(good_corrs) if good_corrs else float('nan')
    bad_mean = np.mean(bad_corrs) if bad_corrs else float('nan')
    print(f"\n  [4] Conditional Corr_S:")
    print(f"      Good samples: {good_mean:.4f} (n={len(good_corrs)})")
    print(f"      Bad samples:  {bad_mean:.4f} (n={len(bad_corrs)})")

    # 5. σ-bin MAE monotonicity
    n_bins = 5
    sigma_pcts = np.percentile(std, np.linspace(0, 100, n_bins + 1))
    bin_maes = []
    print(f"\n  [5] σ-bin reliability:")
    print(f"      {'Bin':<10} {'σ range':>20} {'MAE':>10} {'n':>8}")
    for b in range(n_bins):
        lo, hi = sigma_pcts[b], sigma_pcts[b+1]
        mask = (std >= lo) & (std < hi + 1e-10)
        if mask.sum() > 0:
            bin_mae = np.mean(err[mask])
            bin_maes.append(bin_mae)
            print(f"      Q{b*20}-Q{(b+1)*20}     [{lo:.4f}, {hi:.4f}]  {bin_mae:>10.5f}  {mask.sum():>8}")
    is_monotonic = all(bin_maes[i] <= bin_maes[i+1] for i in range(len(bin_maes)-1))
    print(f"      Monotonic: {'✅ YES' if is_monotonic else '⚠️ NO'}")

    # ─── Decision gate ───
    print(f"\n{'─'*70}")
    print(f"  DECISION GATE:")
    gates = {
        'RMSE ≤ 0.055':       rmse <= 0.055,
        'StdRatio > 0.55':    ratio > 0.55,
        'NLL ≥ -2.7':        nll >= -2.7,
        'Cov@95 in [85,98]':  85 <= cov*100 <= 98,
    }
    all_pass = True
    for desc, ok in gates.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"    {desc:<25} {status}")
        if not ok:
            all_pass = False

    verdict = "🎉 SMOKE PASS" if all_pass else "🛑 SMOKE FAIL"
    print(f"\n  >>> {verdict} <<<")
    print(f"{'='*70}")

    return {
        'rmse': float(rmse), 'nll': float(nll), 'crps': float(crps),
        'cov95': float(cov), 'avgw': float(avgw), 'corr_s': float(cs),
        'std_ratio': float(ratio), 'partial_corr': float(partial_corr),
        'good_sample_corr': float(good_mean), 'bad_sample_corr': float(bad_mean),
        'mae_monotonic': is_monotonic, 'all_pass': all_pass,
    }


def train_mode_a():
    """Mode A: end-to-end decoupled training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'═'*60}")
    print(f"  MODE A: End-to-end decoupled OVCNO-v2, seed={SEED}")
    print(f"{'═'*60}")

    torch.manual_seed(SEED); np.random.seed(SEED)

    train_ds = HYCOMOVCNODataset(NC_PATH, STATION_JSON, pts_per_sample=PTS_PER_SAMPLE,
                                  T_obs=T_OBS, seed=SEED, split="train")
    val_ds = HYCOMOVCNODataset(NC_PATH, STATION_JSON, pts_per_sample=2048,
                                T_obs=T_OBS, seed=SEED, split="val",
                                train_mean=train_ds.train_mean)
    test_ds = HYCOMOVCNODataset(NC_PATH, STATION_JSON, pts_per_sample=2048,
                                 T_obs=T_OBS, seed=SEED, split="test",
                                 train_mean=train_ds.train_mean)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = OVCNOv2Decoupled(lstm_hidden=256, latent_dim=256, width=256).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_A)

    best_val_nll = float('inf')
    for epoch in range(1, EPOCHS_A + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0; n_b = 0
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B*P, 4).to(device)
            labels_flat = labels.view(B*P, 1).to(device)

            y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk_flat)
            loss, lnll, lkl, lrank = compute_v2_loss(
                y_mu, y_logvar, labels_flat, mu_z, logvar_z, o_i, trunk_flat[:, 3:4]
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1

        sched.step()

        # Val
        model.eval()
        val_nll = 0; vb = 0
        with torch.no_grad():
            for hist, pts, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                hist, pts = hist.to(device), pts.to(device)
                trunk_flat = trunk.view(B*P, 4).to(device)
                labels_flat = labels.view(B*P, 1).to(device)
                y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk_flat, sample_z=True)
                nll_v = 0.5 * (y_logvar + (labels_flat - y_mu)**2 / torch.exp(y_logvar)).mean()
                val_nll += nll_v.item(); vb += 1

        val_avg = val_nll / vb
        improved = ""
        if val_avg < best_val_nll:
            best_val_nll = val_avg
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "v2_modeA.pt"))
            improved = " ★"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:>3}/{EPOCHS_A} [{time.time()-t0:.1f}s] "
                  f"Loss={ep_loss/n_b:.4f} ValNLL={val_avg:.4f}{improved}")

    # Load best & evaluate
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "v2_modeA.pt"), map_location=device))
    data = evaluate_full(model, test_dl, device)
    diag = run_diagnostics("OVCNO-v2 Mode A (end-to-end decoupled)", data)
    with open(os.path.join(SAVE_DIR, "modeA_diagnostics.json"), 'w') as f:
        json.dump(diag, f, indent=2, default=str)
    return diag


def train_mode_b():
    """Mode B: two-stage training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'═'*60}")
    print(f"  MODE B: Two-stage decoupled OVCNO-v2, seed={SEED}")
    print(f"{'═'*60}")

    torch.manual_seed(SEED); np.random.seed(SEED)

    train_ds = HYCOMOVCNODataset(NC_PATH, STATION_JSON, pts_per_sample=PTS_PER_SAMPLE,
                                  T_obs=T_OBS, seed=SEED, split="train")
    val_ds = HYCOMOVCNODataset(NC_PATH, STATION_JSON, pts_per_sample=2048,
                                T_obs=T_OBS, seed=SEED, split="val",
                                train_mean=train_ds.train_mean)
    test_ds = HYCOMOVCNODataset(NC_PATH, STATION_JSON, pts_per_sample=2048,
                                 T_obs=T_OBS, seed=SEED, split="test",
                                 train_mean=train_ds.train_mean)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = OVCNOv2Decoupled(lstm_hidden=256, latent_dim=256, width=256).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    # ════════════════════════════════════════════
    # Stage 1: Train mean head only (MSE + KL)
    # Freeze: trunk_logvar, obs_net, bias_logvar
    # ════════════════════════════════════════════
    print(f"\n  ── Stage 1: Mean-only training ({EPOCHS_B1} epochs) ──")
    for name, p in model.named_parameters():
        if 'trunk_logvar' in name or 'obs_net' in name or 'bias_logvar' in name:
            p.requires_grad = False

    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LR, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS_B1)

    best_val_mse = float('inf')
    for epoch in range(1, EPOCHS_B1 + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0; n_b = 0
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B*P, 4).to(device)
            labels_flat = labels.view(B*P, 1).to(device)

            y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk_flat)
            loss, mse = compute_mse_loss(y_mu, labels_flat, mu_z, logvar_z)

            opt1.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            ep_loss += loss.item(); n_b += 1

        sched1.step()

        # Val MSE
        model.eval()
        val_mse = 0; vb = 0
        with torch.no_grad():
            for hist, pts, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                hist, pts = hist.to(device), pts.to(device)
                trunk_flat = trunk.view(B*P, 4).to(device)
                labels_flat = labels.view(B*P, 1).to(device)
                y_mu, _, mz, lz, _ = model(hist, pts, trunk_flat)
                val_mse += ((labels_flat - y_mu)**2).mean().item(); vb += 1

        val_avg = val_mse / vb
        improved = ""
        if val_avg < best_val_mse:
            best_val_mse = val_avg
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "v2_modeB_stage1.pt"))
            improved = " ★"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  S1 Ep {epoch:>3}/{EPOCHS_B1} [{time.time()-t0:.1f}s] "
                  f"Loss={ep_loss/n_b:.4f} ValMSE={val_avg:.6f}{improved}")

    # Stage 1 RMSE check
    s1_rmse = np.sqrt(best_val_mse)
    print(f"\n  Stage 1 done. Val RMSE = {s1_rmse:.5f}")
    if s1_rmse > 0.06:
        print("  ⚠️  Stage 1 RMSE still high — mean may not have converged.")

    # ════════════════════════════════════════════
    # Stage 2: Freeze/low-LR mean, train variance + obs
    # ════════════════════════════════════════════
    print(f"\n  ── Stage 2: Variance/obs training ({EPOCHS_B2} epochs) ──")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "v2_modeB_stage1.pt"), map_location=device))

    # Unfreeze everything, but use separate param groups
    for p in model.parameters():
        p.requires_grad = True

    # Mean head gets very low LR, everything else gets normal LR
    mean_params = list(model.trunk_mu.parameters())
    mean_param_ids = set(id(p) for p in mean_params)
    other_params = [p for p in model.parameters() if id(p) not in mean_param_ids]

    opt2 = torch.optim.AdamW([
        {'params': mean_params, 'lr': LR_STAGE2_MEAN},
        {'params': other_params, 'lr': LR},
    ], weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS_B2)

    best_val_nll = float('inf')
    for epoch in range(1, EPOCHS_B2 + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0; n_b = 0
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B*P, 4).to(device)
            labels_flat = labels.view(B*P, 1).to(device)

            y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk_flat)
            loss, lnll, lkl, lrank = compute_v2_loss(
                y_mu, y_logvar, labels_flat, mu_z, logvar_z, o_i, trunk_flat[:, 3:4]
            )

            opt2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            ep_loss += loss.item(); n_b += 1

        sched2.step()

        # Val NLL
        model.eval()
        val_nll = 0; vb = 0
        with torch.no_grad():
            for hist, pts, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                hist, pts = hist.to(device), pts.to(device)
                trunk_flat = trunk.view(B*P, 4).to(device)
                labels_flat = labels.view(B*P, 1).to(device)
                y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk_flat, sample_z=True)
                nll_v = 0.5 * (y_logvar + (labels_flat - y_mu)**2 / torch.exp(y_logvar)).mean()
                val_nll += nll_v.item(); vb += 1

        val_avg = val_nll / vb
        improved = ""
        if val_avg < best_val_nll:
            best_val_nll = val_avg
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "v2_modeB.pt"))
            improved = " ★"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  S2 Ep {epoch:>3}/{EPOCHS_B2} [{time.time()-t0:.1f}s] "
                  f"Loss={ep_loss/n_b:.4f} ValNLL={val_avg:.4f}{improved}")

    # Load best & evaluate
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "v2_modeB.pt"), map_location=device))
    data = evaluate_full(model, test_dl, device)
    diag = run_diagnostics("OVCNO-v2 Mode B (two-stage decoupled)", data)
    with open(os.path.join(SAVE_DIR, "modeB_diagnostics.json"), 'w') as f:
        json.dump(diag, f, indent=2, default=str)
    return diag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   HYCOM OVCNO-v2 DECOUPLED SMOKE TEST           ║")
    print("║   obs → variance only, mean blind to obs        ║")
    print("╚══════════════════════════════════════════════════╝")

    results = {}
    if args.mode in ("A", "both"):
        results['modeA'] = train_mode_a()
    if args.mode in ("B", "both"):
        results['modeB'] = train_mode_b()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for mode, diag in results.items():
        status = "🎉 PASS" if diag['all_pass'] else "🛑 FAIL"
        print(f"  {mode}: RMSE={diag['rmse']:.4f} NLL={diag['nll']:.3f} "
              f"Cov={diag['cov95']*100:.1f}% StdR={diag['std_ratio']:.3f} "
              f"Corr_S={diag['corr_s']:.3f} → {status}")
    print("="*70)

    with open(os.path.join(SAVE_DIR, "summary.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {SAVE_DIR}/summary.json")
