"""
HYCOM Diagnostic Ablation: 4 variants to diagnose OVCNO mean collapse.
Single seed (42), 100 epochs, patience 20.

Variants:
  A: OVCNO-Decouple — obs only modulates variance head
  B: OVCNO original, lambda_obs=0.1 (reduced ranking loss)
  C: OVCNO original, checkpoint by val RMSE (not NLL)
  D: OVCNO original, lambda_obs=0.0 (no ranking loss at all)
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from dataset_hycom import HYCOMOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_ovcno_v2 import OVCNOv2Decoupled
from loss_ovcno import compute_ovcno_loss

# ─── Config ───
NC_PATH = "../data/hycom_data/hycom_ssh_tonkin_jan_sep_2024.nc"
STATION_JSON = "hycom_real_k12_stations.json"
SAVE_DIR = "hycom_diagnostic"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 4
LR = 1e-3
T_OBS = 8
PTS_PER_SAMPLE = 512
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    hist, pts, trunk, labels = zip(*batch)
    return (torch.stack(hist), torch.stack(pts),
            torch.stack(trunk), torch.stack(labels))


def compute_spatial_std_ratio(model, test_dl, model_type, device):
    """Compute spatial_std(pred_mean) / spatial_std(gt) to detect mean collapse."""
    model.eval()
    pred_stds, gt_stds = [], []
    with torch.no_grad():
        for hist, pts, trunk, labels in test_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B * P, 4).to(device)
            labels_flat = labels.view(B * P, 1).to(device)
            
            if model_type in ("ovcno", "decouple"):
                y_mu, _, _, _, _ = model(hist, pts, trunk_flat)
            else:
                y_mu, _, _, _ = model(hist, trunk_flat[:, :3])
            
            # Per-sample spatial std
            y_mu_batch = y_mu.view(B, P)
            lab_batch = labels_flat.view(B, P)
            
            for b in range(B):
                pred_stds.append(y_mu_batch[b].std().item())
                gt_stds.append(lab_batch[b].std().item())
    
    pred_mean_std = np.mean(pred_stds)
    gt_mean_std = np.mean(gt_stds)
    ratio = pred_mean_std / (gt_mean_std + 1e-8)
    return ratio, pred_mean_std, gt_mean_std


def train_variant(variant_name, model_type, lambda_obs, ckpt_by, seed=SEED):
    """
    Train one diagnostic variant.
    model_type: 'ovcno' | 'decouple'
    lambda_obs: float (ranking loss weight)
    ckpt_by: 'nll' | 'rmse'
    """
    print(f"\n{'='*60}")
    print(f"  Variant: {variant_name}")
    print(f"  model={model_type}, lambda_obs={lambda_obs}, ckpt_by={ckpt_by}")
    print(f"{'='*60}")
    
    train_ds = HYCOMOVCNODataset(
        NC_PATH, STATION_JSON, pts_per_sample=PTS_PER_SAMPLE,
        T_obs=T_OBS, seed=seed, split="train", variable_sensors=False
    )
    val_ds = HYCOMOVCNODataset(
        NC_PATH, STATION_JSON, pts_per_sample=2048,
        T_obs=T_OBS, seed=seed, split="val", variable_sensors=False,
        train_mean=train_ds.train_mean
    )
    test_ds = HYCOMOVCNODataset(
        NC_PATH, STATION_JSON, pts_per_sample=2048,
        T_obs=T_OBS, seed=seed, split="test", variable_sensors=False,
        train_mean=train_ds.train_mean
    )
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=4, shuffle=False,
                         collate_fn=collate_fn, num_workers=2)
    
    if model_type == "decouple":
        model = OVCNOv2Decoupled(
            lstm_hidden=256, latent_dim=256, width=256
        ).to(device)
    else:
        model = ObservabilityAwareVCNO(
            lstm_hidden=256, latent_dim=256, width=256
        ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )
    
    best_val_metric = float('inf')
    patience_ctr = 0
    ckpt_path = os.path.join(SAVE_DIR, f"hycom_{variant_name}.pt")
    
    epoch_log = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0; n_b = 0
        
        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B * P, 4).to(device)
            labels_flat = labels.view(B * P, 1).to(device)
            d_s = trunk_flat[:, 3:4]
            
            y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk_flat)
            loss, lnll, _, _ = compute_ovcno_loss(
                y_mu, y_logvar, labels_flat, mu_z, logvar_z, o_i, d_s,
                lambda_obs=lambda_obs
            )
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1
        
        # Validation
        model.eval()
        val_nll = 0; val_rmse_sum = 0; val_n = 0; vb = 0
        with torch.no_grad():
            for hist, pts, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                hist, pts = hist.to(device), pts.to(device)
                trunk_flat = trunk.view(B * P, 4).to(device)
                labels_flat = labels.view(B * P, 1).to(device)
                
                y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk_flat)
                _, nv, _, _ = compute_ovcno_loss(
                    y_mu, y_logvar, labels_flat, mz, lz, oi, trunk_flat[:, 3:4],
                    lambda_obs=lambda_obs
                )
                val_nll += nv.item(); vb += 1
                val_rmse_sum += ((y_mu - labels_flat)**2).sum().item()
                val_n += y_mu.numel()
        
        val_nll_avg = val_nll / vb
        val_rmse = float(np.sqrt(val_rmse_sum / val_n))
        
        # Pick metric for checkpoint
        if ckpt_by == "rmse":
            val_metric = val_rmse
        else:
            val_metric = val_nll_avg
        
        sched.step(val_nll_avg)  # scheduler always uses NLL
        
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), ckpt_path)
            patience_ctr = 0
            tag = " ★"
        else:
            patience_ctr += 1
            tag = ""
        
        if epoch % 5 == 0 or epoch == 1 or tag:
            print(f"  Ep {epoch:>3} [{time.time()-t0:.1f}s] TrL={ep_loss/n_b:.4f} "
                  f"ValNLL={val_nll_avg:.4f} ValRMSE={val_rmse:.4f}{tag}")
        
        epoch_log.append({
            "epoch": epoch, "train_loss": ep_loss/n_b,
            "val_nll": val_nll_avg, "val_rmse": val_rmse,
            "best": tag.strip() == "★"
        })
        
        if patience_ctr >= PATIENCE:
            print(f"  Early stop at ep {epoch}")
            break
    
    # ─── Test ───
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    all_mu, all_lab, all_lv = [], [], []
    with torch.no_grad():
        for hist, pts, trunk, labels in test_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk_flat = trunk.view(B * P, 4).to(device)
            labels_flat = labels.view(B * P, 1).to(device)
            y_mu, y_logvar, _, _, _ = model(hist, pts, trunk_flat)
            all_mu.append(y_mu.cpu().numpy())
            all_lab.append(labels_flat.cpu().numpy())
            all_lv.append(y_logvar.cpu().numpy())
    
    mu = np.concatenate(all_mu).flatten()
    lab = np.concatenate(all_lab).flatten()
    lv = np.concatenate(all_lv).flatten()
    std = np.exp(0.5 * lv)
    
    rmse = float(np.sqrt(np.mean((mu - lab)**2)))
    mae = float(np.mean(np.abs(mu - lab)))
    nll_t = float(0.5 * np.mean(lv + (lab - mu)**2 / np.exp(lv)))
    
    z95 = 1.96
    cov = float(np.mean((lab >= mu - z95*std) & (lab <= mu + z95*std)))
    avgw = float(np.mean(2 * z95 * std))
    
    errs = np.abs(mu - lab)
    cs, _ = spearmanr(errs, std)
    
    # Spatial std ratio
    ratio, pred_std, gt_std = compute_spatial_std_ratio(model, test_dl, model_type, device)
    
    best_epoch = max((e for e in epoch_log if e['best']), key=lambda x: x['epoch'], default={'epoch': 0})['epoch']
    
    res = {
        "variant": variant_name, "model_type": model_type,
        "lambda_obs": lambda_obs, "ckpt_by": ckpt_by,
        "rmse": rmse, "mae": mae, "nll": nll_t,
        "cov95": cov, "avgw": avgw, "corr_s": float(cs),
        "spatial_std_ratio": ratio,
        "pred_spatial_std": pred_std, "gt_spatial_std": gt_std,
        "best_epoch": best_epoch, "total_epochs": epoch,
        "mean_logvar": float(np.mean(lv))
    }
    
    print(f"\n  TEST: RMSE={rmse:.4f} MAE={mae:.4f} NLL={nll_t:.4f}")
    print(f"        Cov95={cov*100:.1f}% AvgW={avgw:.4f} CorrS={cs:.4f}")
    print(f"        SpatStdRatio={ratio:.4f} (pred_std={pred_std:.4f}, gt_std={gt_std:.4f})")
    print(f"        BestEpoch={best_epoch} MeanLogvar={np.mean(lv):.3f}")
    
    # Save epoch log
    with open(os.path.join(SAVE_DIR, f"log_{variant_name}.json"), 'w') as f:
        json.dump(epoch_log, f, indent=1)
    
    return res


if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════╗")
    print("║  HYCOM DIAGNOSTIC ABLATION (4 variants)       ║")
    print("╚═══════════════════════════════════════════════╝")
    
    variants = [
        # (name, model_type, lambda_obs, ckpt_by)
        ("A_decouple",  "decouple", 1.0, "nll"),    # obs → variance only
        ("B_lobs01",    "ovcno",    0.1, "nll"),     # reduced ranking loss
        ("C_ckpt_rmse", "ovcno",    1.0, "rmse"),    # checkpoint by RMSE
        ("D_lobs00",    "ovcno",    0.0, "nll"),     # no ranking loss (OVCNO-Geom)
    ]
    
    all_results = []
    for name, mt, lobs, ckpt in variants:
        r = train_variant(name, mt, lobs, ckpt)
        all_results.append(r)
    
    # ─── Summary ───
    print(f"\n{'='*90}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*90}")
    print(f"{'Variant':<16} {'RMSE':>7} {'NLL':>7} {'Cov95':>7} {'AvgW':>7} "
          f"{'CorrS':>7} {'StdR':>7} {'BstEp':>6}")
    print("-"*90)
    for r in all_results:
        print(f"{r['variant']:<16} {r['rmse']:>7.4f} {r['nll']:>7.3f} "
              f"{r['cov95']*100:>6.1f}% {r['avgw']:>7.4f} "
              f"{r['corr_s']:>7.4f} {r['spatial_std_ratio']:>7.4f} {r['best_epoch']:>6}")
    print(f"{'='*90}")
    
    with open(os.path.join(SAVE_DIR, "diagnostic_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {SAVE_DIR}/diagnostic_results.json")
