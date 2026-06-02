import os
import math
import numpy as np
import torch
import traceback
from torch.utils.data import DataLoader
from dataset_vae import CopernicusVAEDataset
from model_vae import ForecastDeepONetVAE
from loss import compute_vae_loss

def collate_fn(batch):
    hists, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    n_sensors = hists[0].shape[1]

    padded = np.zeros((len(hists), T_max, n_sensors), dtype=np.float32)
    for i, h in enumerate(hists):
        padded[i, :h.shape[0]] = h

    return (
        torch.tensor(padded),
        torch.stack(trunks),
        torch.stack(labels)
    )

def evaluate_metrics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating on device:", device)

    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    val_ds = CopernicusVAEDataset(nc_path, split="val", pts_per_sample=2048)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=128, latent_dim=128).to(device)
    
    ckpt_path = "vae_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return

    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded checkpoint successfully.")
    except Exception as e:
        print("Failed to load checkpoint:")
        traceback.print_exc()
        return

    model.eval()

    total_rmse = 0.0
    total_mae = 0.0
    total_l2_num = 0.0
    total_l2_den = 0.0
    total_nll = 0.0
    total_in_cov95 = 0
    total_points = 0
    
    with torch.no_grad():
        for hist, trunk, labels in val_dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist = hist.to(device)
            trunk = trunk.view(B*P, 3).to(device)
            labels = labels.view(B*P, 1).to(device)

            pred_mu, pred_logvar, mz, lz = model(hist, trunk)
            
            # Predict
            err = (pred_mu - labels).abs()
            mse = (err ** 2)
            
            # Metric aggregations
            total_rmse += mse.sum().item()
            total_mae  += err.sum().item()
            total_l2_num += mse.sum().item()
            total_l2_den += (labels ** 2).sum().item()
            
            # NLL
            pred_var = torch.exp(pred_logvar)
            nll_pointwise = 0.5 * (np.log(2 * math.pi) + pred_logvar + (pred_mu - labels)**2 / pred_var)
            total_nll += nll_pointwise.sum().item()
            
            # Cov@95%
            std = torch.sqrt(pred_var)
            lower = pred_mu - 1.96 * std
            upper = pred_mu + 1.96 * std
            in_bound = (labels >= lower) & (labels <= upper)
            total_in_cov95 += in_bound.sum().item()
            
            total_points += (B * P)

    rmse = math.sqrt(total_rmse / total_points)
    mae = total_mae / total_points
    rel_l2 = math.sqrt(total_l2_num) / math.sqrt(total_l2_den) if total_l2_den > 0 else 0
    nll = total_nll / total_points
    cov95 = total_in_cov95 / total_points * 100

    print("=== Variational HYCOM METRICS ===")
    print(f"Rel-L2: {rel_l2*100:.2f}%")
    print(f"RMSE:   {rmse:.4f}")
    print(f"MAE:    {mae:.4f}")
    print(f"NLL:    {nll:.4f}")
    print(f"Cov@95: {cov95:.2f}%")
    print("================================")

if __name__ == "__main__":
    evaluate_metrics()
