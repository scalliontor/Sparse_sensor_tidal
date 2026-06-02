import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO

def collate_fn(batch):
    hists, pts, trunks, labels = zip(*batch)
    T_max = max(h.shape[0] for h in hists)
    K = hists[0].shape[1]
    
    padded_h = torch.zeros((len(hists), T_max, K), dtype=torch.float32)
    for i, h in enumerate(hists):
        padded_h[i, :h.shape[0]] = h
        
    return padded_h, torch.stack(pts), torch.stack(trunks), torch.stack(labels)

def evaluate_sensor_count(n_sensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
    
    ds = CopernicusOVCNODataset(nc_path, n_sensors=n_sensor, pts_per_sample=2048, 
                                split="val", variable_sensors=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    model = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
    model.load_state_dict(torch.load("ovcno_checkpoint.pt", map_location=device))
    model.eval()
    
    total_rmse = 0.0; total_nll = 0.0; total_points = 0; total_l2_num = 0.0; total_l2_den = 0.0
    with torch.no_grad():
        for hist, pts, trunk, labels in dl:
            B, P = trunk.shape[0], trunk.shape[1]
            hist, pts, trunk, labels = hist.to(device), pts.to(device), trunk.view(B*P, 4).to(device), labels.view(B*P, 1).to(device)
            
            # Subsample to strictly simulate Setting C Unseen layout/missing capability if n_sensor < 16 on a 16-trained model!
            # Since the model is trained on 16 or mixed, inference on different K tests permutation invariance.
            y_mu, y_logvar, mz, lz, oi = model(hist, pts, trunk)
            
            err = (y_mu - labels).abs()
            mse = err ** 2
            total_rmse += mse.sum().item()
            
            total_l2_num += mse.sum().item()
            total_l2_den += (labels ** 2).sum().item()
            
            pred_var = torch.exp(y_logvar)
            nll_pointwise = 0.5 * (np.log(2 * math.pi) + y_logvar + mse / pred_var)
            total_nll += nll_pointwise.sum().item()
            
            total_points += (B * P)
            
    rmse = math.sqrt(total_rmse / total_points)
    nll = total_nll / total_points
    rel_l2 = math.sqrt(total_l2_num) / math.sqrt(total_l2_den) if total_l2_den > 0 else 0
    return rmse, nll, rel_l2

if __name__ == "__main__":
    print("OVCNO Variable Sensor Resilience Test")
    for k in [8, 16, 32]:
        rmse, nll, rel_l2 = evaluate_sensor_count(k)
        print(f"Sensors: {k:2d} -> RMSE: {rmse:.4f} | NLL: {nll:.4f} | Raw Rel-L2: {rel_l2*100:.2f}%")
