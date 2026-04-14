#!/usr/bin/env python3
"""
evaluate_deeponet.py

Evaluate a trained DeepONet checkpoint against a raw Roe simulation file.

Raw sim .npz expected keys:
  - eta_vec: (m,)
  - x: (Nx,)
  - t: (Nt,)
  - h: (Nt, Nx)

Normalization used during training:
  x_norm = x / L
  t_norm = t / T_max

Outputs:
  - Field RMSE/MAE
  - Station amplitude error and phase-lag error (via cross-correlation)
  - Optional plots

Usage example:
  python3 evaluate_deeponet.py \
    --sim_npz data/raw_train/sim_0000.npz \
    --ckpt checkpoints/deeponet_best.pt \
    --L 100000 --Tmax 172800 \
    --stations_km 10 50 90 \
    --batch 65536 \
    --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import sys
import os

# Add parent directory to path to allow importing deeponet modules if run from script dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# assumes you have deeponet/model.py from the skeleton
from deeponet.model import DeepONet


def nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def field_metrics(h_pred: np.ndarray, h_true: np.ndarray) -> dict:
    err = h_pred - h_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": rmse, "mae": mae}


def amplitude(ts: np.ndarray) -> float:
    # peak-to-peak / 2
    return float(0.5 * (np.max(ts) - np.min(ts)))


def phase_lag_seconds(gt: np.ndarray, pred: np.ndarray, dt_seconds: float) -> float:
    """
    Estimate phase lag using cross-correlation.
    Returns lag in seconds: positive => pred lags behind gt.
    """
    gt0 = gt - np.mean(gt)
    pr0 = pred - np.mean(pred)

    # If either is nearly constant, lag is undefined; return 0.
    if np.std(gt0) < 1e-12 or np.std(pr0) < 1e-12:
        return 0.0

    corr = np.correlate(pr0, gt0, mode="full")  # correlate(pred, gt)
    lag_idx = int(np.argmax(corr) - (len(gt0) - 1))
    return float(lag_idx * dt_seconds)


@torch.no_grad()
def predict_full_field(
    model: torch.nn.Module,
    eta_vec: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    L: float,
    Tmax: float,
    batch: int,
    device: torch.device,
) -> np.ndarray:
    """
    Predict h(t,x) on the full grid defined by arrays t (Nt,) and x (Nx,).
    Returns h_pred of shape (Nt, Nx).
    """
    model.eval()

    Nt = t.shape[0]
    Nx = x.shape[0]

    # trunk grid in normalized coordinates
    x_norm = (x / L).astype(np.float32)
    t_norm = (t / Tmax).astype(np.float32)

    # Build all (x,t) pairs in the same order as true field: (Nt, Nx)
    # We'll create flattened arrays of length Nt*Nx:
    # For each time index k, traverse all x indices i.
    X = np.tile(x_norm[None, :], (Nt, 1)).reshape(-1)           # (Nt*Nx,)
    T = np.tile(t_norm[:, None], (1, Nx)).reshape(-1)           # (Nt*Nx,)
    trunk = np.stack([X, T], axis=1)                            # (Nt*Nx, 2)

    # Branch: repeat eta_vec for each point
    eta_vec = eta_vec.astype(np.float32)
    m = eta_vec.shape[0]

    total = trunk.shape[0]
    out = np.empty((total, 1), dtype=np.float32)

    # Torch tensors prepared per batch
    eta_t = torch.from_numpy(eta_vec).to(device).view(1, m)

    for start in range(0, total, batch):
        end = min(start + batch, total)
        trunk_b = torch.from_numpy(trunk[start:end]).to(device)               # (B,2)
        branch_b = eta_t.expand(end - start, -1).contiguous()                 # (B,m)
        pred_b = model(branch_b, trunk_b)                                     # (B,1)
        out[start:end, :] = pred_b.detach().cpu().numpy()

    return out.reshape(Nt, Nx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim_npz", type=str, required=True, help="Path to raw sim_XXXX.npz")
    p.add_argument("--ckpt", type=str, required=True, help="Path to DeepONet checkpoint .pt")

    # Normalization constants (must match dataset_builder.py)
    p.add_argument("--L", type=float, default=100000.0, help="Domain length in meters used for x_norm = x/L")
    p.add_argument("--Tmax", type=float, default=48.0 * 3600.0, help="Max time in seconds used for t_norm = t/Tmax")

    # Model hyperparams must match training
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--act", type=str, default="gelu")
    p.add_argument("--dropout", type=float, default=0.0)

    # Eval config
    p.add_argument("--batch", type=int, default=65536, help="Batch size for full-grid inference")
    p.add_argument("--stations_km", type=float, nargs="*", default=[10.0, 50.0, 90.0], help="Station x locations in km")
    p.add_argument("--out_json", type=str, default="", help="Optional path to save metrics as json")
    p.add_argument("--save_pred_npz", type=str, default="", help="Optional path to save predicted field .npz")
    p.add_argument("--plot", action="store_true", help="Show plots (matplotlib required)")
    p.add_argument("--save_plot", type=str, default="evaluation_results.png", help="Path to save plot")
    args = p.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load raw sim
    sim = np.load(args.sim_npz)
    eta_vec = sim["eta_vec"]          # (m,)
    x = sim["x"]                      # (Nx,)
    t = sim["t"]                      # (Nt,)
    h_true = sim["h"]                 # (Nt,Nx)

    if h_true.ndim != 2:
        raise ValueError(f"Expected h to be (Nt,Nx), got shape {h_true.shape}")

    Nt, Nx = h_true.shape
    m = eta_vec.shape[0]
    print(f"Loaded sim: Nt={Nt}, Nx={Nx}, m={m}")

    # Build model and load checkpoint
    model = DeepONet(
        branch_in=m,
        trunk_in=2,
        width=args.width,
        depth=args.depth,
        latent_dim=args.latent,
        activation=args.act,
        dropout=args.dropout,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # supports either full checkpoint dict or pure state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Predict full field
    h_pred = predict_full_field(
        model=model,
        eta_vec=eta_vec,
        x=x,
        t=t,
        L=args.L,
        Tmax=args.Tmax,
        batch=args.batch,
        device=device,
    )

    # Field metrics
    fm = field_metrics(h_pred, h_true)
    print("\nField metrics:")
    print(f"  RMSE: {fm['rmse']:.6e}")
    print(f"  MAE : {fm['mae']:.6e}")

    # Station metrics
    # dt from snapshots (assume uniform; if not, we use median)
    if len(t) >= 2:
        dt_seconds = float(np.median(np.diff(t)))
    else:
        dt_seconds = 0.0

    station_results = []
    for km in args.stations_km:
        xm = km * 1000.0
        ix = nearest_index(x, xm)

        gt_ts = h_true[:, ix]
        pr_ts = h_pred[:, ix]

        amp_gt = amplitude(gt_ts)
        amp_pr = amplitude(pr_ts)
        amp_err = float(amp_pr - amp_gt)
        amp_rel = float(amp_err / (amp_gt + 1e-12))

        lag_s = phase_lag_seconds(gt_ts, pr_ts, dt_seconds)

        station_results.append(
            {
                "station_km": float(km),
                "x_m": float(x[ix]),
                "ix": int(ix),
                "amp_gt": amp_gt,
                "amp_pred": amp_pr,
                "amp_err": amp_err,
                "amp_rel": amp_rel,
                "phase_lag_s": lag_s,
            }
        )

    print("\nStation metrics:")
    for r in station_results:
        print(
            f"  x={r['station_km']:.1f} km (ix={r['ix']}): "
            f"amp_gt={r['amp_gt']:.4f}, amp_pred={r['amp_pred']:.4f}, "
            f"amp_err={r['amp_err']:+.4f} ({r['amp_rel']*100:+.2f}%), "
            f"lag={r['phase_lag_s']:+.1f} s"
        )

    # Save metrics json if requested
    metrics_out = {
        "sim_npz": str(args.sim_npz),
        "ckpt": str(args.ckpt),
        "normalization": {"L": args.L, "Tmax": args.Tmax},
        "field": fm,
        "stations": station_results,
    }

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics_out, indent=2))
        print("\nSaved metrics to:", str(outp))

    # Save prediction field if requested
    if args.save_pred_npz:
        outp = Path(args.save_pred_npz)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            outp,
            x=x,
            t=t,
            h_true=h_true,
            h_pred=h_pred,
            eta_vec=eta_vec,
        )
        print("Saved prediction npz to:", str(outp))

    # Plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required for --plot") from e

        # Combined plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Station time series plots
        for r in station_results:
            ix = r["ix"]
            axes[0].plot(t / 3600.0, h_true[:, ix], label=f"GT {r['station_km']:.0f} km")
            axes[0].plot(t / 3600.0, h_pred[:, ix], "--", label=f"Pred {r['station_km']:.0f} km")
        axes[0].set_xlabel("Time (hours)")
        axes[0].set_ylabel("h (m)")
        axes[0].set_title("Station time series: GT vs DeepONet")
        axes[0].legend()
        axes[0].grid(True)

        # Error heatmap
        err = h_pred - h_true
        im = axes[1].imshow(
            err.T, # Transpose to make x horizontal, t vertical or vice-versa? 
            # imshow origin='lower' expects [row, col]. 
            # h is (Nt, Nx). x is columns. t is rows.
            # extent order: [left, right, bottom, top]
            # left=x0, right=x1. bottom=t0, top=t1.
            # So we typically want x on x-axis (columns), t on y-axis (rows).
            # So imshow(h) works if origin=lower.
            # But the user script said: 
            # extent=[x[0], x[-1], t[0], t[-1]]
            # extent=[x_min, x_max, y_min, y_max]
            # So x is horizontal, t is vertical.
            # And imshow(err, aspect='auto') with err shape (Nt, Nx)
            # rows=Nt (time), cols=Nx (space).
            # So x-label should be 'x', y-label should be 't'.
            aspect="auto",
            origin="lower",
            extent=[x[0] / 1000.0, x[-1] / 1000.0, t[0] / 3600.0, t[-1] / 3600.0],
        )
        fig.colorbar(im, ax=axes[1], label="h_pred - h_true (m)")
        axes[1].set_xlabel("x (km)")
        axes[1].set_ylabel("t (hours)")
        axes[1].set_title("Error field heatmap")
        
        plt.tight_layout()
        plt.savefig(args.save_plot)
        print(f"Saved plot to {args.save_plot}")
        # plt.show()


if __name__ == "__main__":
    main()
