"""
HYCOM Cross-Product Full Run — 3 seeds
VCO + OVCNO on extended Jan-Sep 2024 data.
Train on Jan-Jun, validate Jul, test Aug-Sep.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader

from dataset_hycom import HYCOMOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE
from loss_ovcno import compute_ovcno_loss

NC_PATH = "../data/hycom_data/hycom_ssh_tonkin_jan_sep_2024.nc"
STATION_JSON = "hycom_real_k12_stations.json"
SAVE_DIR = "hycom_full"
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
EPOCHS = 80
BATCH_SIZE = 16
LR = 3e-4
PTS_PER_SAMPLE = 512
T_OBS = 8
PATIENCE = 20


def collate_fn(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)


def train_and_evaluate(model_type, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  {model_type.upper()} | seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    n_sensors = train_ds.n_stations
    if model_type == "ovcno":
        model = ObservabilityAwareVCNO(
            lstm_hidden=256, latent_dim=256, width=256
        ).to(device)
    else:
        model = ForecastDeepONetVAE(
            n_sensors=n_sensors, lstm_hidden=256, latent_dim=256, width=256
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )

    best_val_nll = float('inf')
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0; n_b = 0

        for hist, pts, trunk, labels in train_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk = trunk.view(B * P, 4).to(device)
            labels = labels.view(B * P, 1).to(device)
            d_s = trunk[:, 3:4]

            if model_type == "ovcno":
                y_mu, y_logvar, mu_z, logvar_z, o_i = model(hist, pts, trunk)
                loss, lnll, _, _ = compute_ovcno_loss(
                    y_mu, y_logvar, labels, mu_z, logvar_z, o_i, d_s
                )
            else:
                y_mu, y_logvar, mu_z, logvar_z = model(hist, trunk[:, :3])
                nll = 0.5 * (y_logvar + (labels - y_mu)**2 / torch.exp(y_logvar)).mean()
                kl = -0.5 * (1 + logvar_z - mu_z**2 - torch.exp(logvar_z)).mean()
                loss = nll + 1e-3 * kl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1

        # Validation
        model.eval()
        val_nll = 0; vb = 0
        with torch.no_grad():
            for hist, pts, trunk, labels in val_dl:
                B, P = trunk.shape[0], trunk.shape[1]
                hist, pts = hist.to(device), pts.to(device)
                trunk = trunk.view(B * P, 4).to(device)
                labels = labels.view(B * P, 1).to(device)
                if model_type == "ovcno":
                    y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk)
                    _, nv, _, _ = compute_ovcno_loss(
                        y_mu, y_logvar, labels, mz, lz, oi, trunk[:, 3:4]
                    )
                else:
                    y_mu, y_logvar, mz, lz = model(hist, trunk[:, :3])
                    nv = 0.5 * (y_logvar + (labels - y_mu)**2 / torch.exp(y_logvar)).mean()
                val_nll += nv.item(); vb += 1

        val_avg = val_nll / vb
        sched.step(val_avg)

        if val_avg < best_val_nll:
            best_val_nll = val_avg
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"hycom_{model_type}_s{seed}.pt"))
            patience_ctr = 0
            tag = " ★"
        else:
            patience_ctr += 1
            tag = ""

        if epoch % 10 == 0 or epoch == 1 or tag:
            print(f"  Ep {epoch:>3} [{time.time()-t0:.1f}s] "
                  f"TrL={ep_loss/n_b:.4f} ValNLL={val_avg:.4f}{tag}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at ep {epoch}")
            break

    # ─── Test ───
    ckpt = os.path.join(SAVE_DIR, f"hycom_{model_type}_s{seed}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_mu, all_lab, all_lv = [], [], []
    with torch.no_grad():
        for hist, pts, trunk, labels in test_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts = hist.to(device), pts.to(device)
            trunk = trunk.view(B * P, 4).to(device)
            labels = labels.view(B * P, 1).to(device)
            if model_type == "ovcno":
                y_mu, y_logvar, _, _, _ = model(hist, pts, trunk)
            else:
                y_mu, y_logvar, _, _ = model(hist, trunk[:, :3])
            all_mu.append(y_mu.cpu().numpy())
            all_lab.append(labels.cpu().numpy())
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

    from scipy.stats import spearmanr
    errs = np.abs(mu - lab)
    cs, _ = spearmanr(errs, std)

    res = {"model": model_type, "seed": seed, "rmse": rmse, "mae": mae,
           "nll": nll_t, "cov95": cov, "avgw": avgw, "corr_s": float(cs),
           "best_val_nll": float(best_val_nll), "epochs": epoch}

    print(f"\n  TEST: RMSE={rmse:.4f} MAE={mae:.4f} NLL={nll_t:.4f} "
          f"Cov95={cov*100:.1f}% AvgW={avgw:.4f} CorrS={cs:.4f}")
    return res


if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════╗")
    print("║  HYCOM CROSS-PRODUCT FULL RUN (3 seeds)       ║")
    print("╚═══════════════════════════════════════════════╝")

    all_results = []
    for model_type in ["vco", "ovcno"]:
        for seed in SEEDS:
            r = train_and_evaluate(model_type, seed)
            all_results.append(r)

    # ─── Summary table ───
    print("\n" + "="*70)
    print("FINAL CROSS-PRODUCT RESULTS (HYCOM, independent test)")
    print("="*70)
    print(f"{'Model':<8} {'Seed':>5} {'RMSE':>8} {'MAE':>8} {'NLL':>8} "
          f"{'Cov95':>7} {'AvgW':>8} {'CorrS':>8}")
    print("-"*70)
    for r in all_results:
        print(f"{r['model']:<8} {r['seed']:>5} {r['rmse']:>8.4f} {r['mae']:>8.4f} "
              f"{r['nll']:>8.4f} {r['cov95']*100:>6.1f}% {r['avgw']:>8.4f} "
              f"{r['corr_s']:>8.4f}")

    # Means
    print("-"*70)
    for mt in ["vco", "ovcno"]:
        subset = [r for r in all_results if r['model'] == mt]
        for metric in ['rmse', 'mae', 'nll', 'cov95', 'avgw', 'corr_s']:
            vals = [r[metric] for r in subset]
        mean_r = {k: np.mean([r[k] for r in subset])
                  for k in ['rmse', 'mae', 'nll', 'cov95', 'avgw', 'corr_s']}
        std_r = {k: np.std([r[k] for r in subset])
                 for k in ['rmse', 'mae', 'nll', 'cov95', 'avgw', 'corr_s']}
        print(f"{mt:<8} {'mean':>5} {mean_r['rmse']:>8.4f} {mean_r['mae']:>8.4f} "
              f"{mean_r['nll']:>8.4f} {mean_r['cov95']*100:>6.1f}% "
              f"{mean_r['avgw']:>8.4f} {mean_r['corr_s']:>8.4f}")
    print("="*70)

    # Save all
    with open(os.path.join(SAVE_DIR, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {SAVE_DIR}/all_results.json")
