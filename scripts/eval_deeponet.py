"""Evaluate DeepONet on held-out test windows."""
import argparse
import numpy as np
import torch
import sys
sys.path.insert(0, "/home/namnx/deepOnet_solver/deeponet")
from model import DeepONet

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/namnx/deepOnet_solver/data/dataset_2d_tidal_ocean.npz")
    p.add_argument("--ckpt", default="/home/namnx/deepOnet_solver/checkpoints/deeponet_2d_ocean_best.pt")
    p.add_argument("--batch", type=int, default=8192)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset
    d = np.load(args.data)
    labels = d["labels"]
    branch = d["branch"]
    trunk = d["trunk"]
    window_id = d["window_id"]
    test_windows = d["test_windows"]
    y_mu = float(d["y_mu"])
    y_std = float(d["y_std"])

    print(f"y_mu={y_mu:.4f}m  y_std={y_std:.4f}m")
    print(f"Test windows: {test_windows}")

    # Filter test samples
    test_mask = np.isin(window_id, test_windows)
    branch_t = torch.tensor(branch[test_mask], dtype=torch.float32)
    trunk_t  = torch.tensor(trunk[test_mask],  dtype=torch.float32)
    labels_t = torch.tensor(labels[test_mask], dtype=torch.float32)
    print(f"Test samples: {len(labels_t)}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    m = branch_t.shape[1]
    model = DeepONet(
        branch_in=m, trunk_in=3,
        width=256, depth=4, latent_dim=128,
        activation="gelu", dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Loaded checkpoint from epoch", ckpt.get("epoch", "?"), "best_val_rmse:", ckpt.get("best_val", "?"))

    # Inference in batches
    all_pred = []
    all_true = []
    with torch.no_grad():
        for i in range(0, len(labels_t), args.batch):
            b_in = branch_t[i:i+args.batch].to(device)
            t_in = trunk_t[i:i+args.batch].to(device)
            pred = model(b_in, t_in).cpu()
            all_pred.append(pred)
            all_true.append(labels_t[i:i+args.batch])

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)

    # Metrics in normalized space
    rmse_norm = torch.sqrt(torch.mean((pred - true)**2)).item()
    mae_norm  = torch.mean(torch.abs(pred - true)).item()
    std_norm  = true.std().item()
    rel_l2_norm = rmse_norm / std_norm * 100

    # Convert to physical units (meters)
    pred_m = pred.numpy() * y_std + y_mu
    true_m = true.numpy() * y_std + y_mu
    rmse_m = float(np.sqrt(np.mean((pred_m - true_m)**2)))
    mae_m  = float(np.mean(np.abs(pred_m - true_m)))
    rel_l2 = rmse_m / float(np.std(true_m)) * 100

    # Baseline: predict mean
    baseline_rmse_m = float(np.std(true_m))
    baseline_rel    = 100.0

    print("\n=== RESULTS (test windows 16-19) ===")
    print(f"RMSE (norm):     {rmse_norm:.4f}  (std_test={std_norm:.4f})")
    print(f"MAE  (norm):     {mae_norm:.4f}")
    print(f"RelL2 (norm):    {rel_l2_norm:.1f}%")
    print()
    print(f"RMSE (meters):   {rmse_m*100:.1f} cm")
    print(f"MAE  (meters):   {mae_m*100:.1f} cm")
    print(f"RelL2 (meters):  {rel_l2:.1f}%")
    print()
    print(f"Baseline (mean): {baseline_rmse_m*100:.1f} cm  (RelL2={baseline_rel:.1f}%)")
    print(f"Improvement over baseline: {baseline_rel - rel_l2:.1f}% RelL2")
    print()
    print(f"True eta range:  [{true_m.min()*100:.1f}, {true_m.max()*100:.1f}] cm")
    print(f"True eta std:    {np.std(true_m)*100:.1f} cm")

if __name__ == "__main__":
    main()
