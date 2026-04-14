"""
Evaluate ForecastDeepONet at multiple forecast horizons.
Reports Rel L2 separately for T_obs = 20, 40, 60, 80 timesteps.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import ForecastDeepONet
from dataset import load_pdebench, make_boundary_sensors


def eval_at_horizon(model, data, t_norm, x_norm, y_norm, sx, sy,
                    T_obs: int, device, pts: int = 2048, n_samples: int = 100):
    """Evaluate Rel L2 when observing up to T_obs, predicting all future."""
    model.eval()
    T, Nx, Ny = data.shape[1], data.shape[2], data.shape[3]
    err2 = norm2 = 0.0

    with torch.no_grad():
        for idx in range(min(n_samples, len(data))):
            field = data[idx]  # (T, Nx, Ny)

            # Branch: sensor history up to T_obs
            hist_np = field[:T_obs, sx, sy].astype(np.float32)   # (T_obs, n_sensors)
            hist = torch.tensor(hist_np).unsqueeze(0).to(device)  # (1, T_obs, n_sensors)

            # Future points: all t > T_obs, random spatial
            n_t = T - T_obs
            t_idx = np.arange(T_obs, T)
            x_idx = np.random.randint(0, Nx, pts)
            y_idx = np.random.randint(0, Ny, pts)

            # Sample random future (t, x, y)
            t_s = np.random.choice(t_idx, pts)
            trunk_np = np.stack([x_norm[x_idx], y_norm[y_idx], t_norm[t_s]], axis=-1).astype(np.float32)
            trunk = torch.tensor(trunk_np).to(device)  # (pts, 3)

            labels_np = field[t_s, x_idx, y_idx]

            # Expand hist to match pts
            hist_exp = hist.expand(pts, -1, -1)  # wrong — need (B=1, T_obs, n_s)
            # Actually model expects (B, T_obs, n_sensors) and trunk (B*P, 3)
            # Set B=1, P=pts
            pred = model(hist, trunk.unsqueeze(0).reshape(pts, 3)).view(-1).cpu().numpy()
            # Hmm, need to fix: hist is (1, T_obs, n_sensors), trunk should be (1*pts, 3)
            pred = model(hist, trunk).view(-1).cpu().numpy()

            err2  += float(np.sum((pred - labels_np) ** 2))
            norm2 += float(np.sum(labels_np ** 2))

    return float(np.sqrt(err2 / (norm2 + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="../data/pdebench/2D_rdb_NA_NA.h5")
    ap.add_argument("--ckpt",    default="../checkpoints/forecast_best.pt")
    ap.add_argument("--n_sensors", type=int, default=16)
    ap.add_argument("--n_eval",    type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from dataset import load_pdebench, make_boundary_sensors
    data, t_c, x_c, y_c = load_pdebench(args.data, 1000)
    val_data = data[800:]

    def norm(c): return (2*(c - c[0])/(c[-1] - c[0] + 1e-12) - 1).astype(np.float32)
    t_norm, x_norm, y_norm = norm(t_c), norm(x_c), norm(y_c)

    sx, sy = make_boundary_sensors(128, 128, args.n_sensors)

    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    model = ForecastDeepONet(
        n_sensors=args.n_sensors,
        lstm_hidden=saved_args.get("lstm_hidden", 256),
        lstm_layers=saved_args.get("lstm_layers", 2),
        latent_dim=saved_args.get("latent", 256),
        width=saved_args.get("width", 256),
        depth=saved_args.get("depth", 4),
        n_fourier_freqs=saved_args.get("n_fourier", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch {ckpt['epoch']}  val_relL2={ckpt['val_rel_l2']*100:.2f}%\n")

    print("=== Forecast Rel L2 by observation horizon ===")
    print(f"{'T_obs':>8} {'T_future':>10} {'Rel L2':>10}")
    for T_obs in [20, 40, 60, 80]:
        rl2 = eval_at_horizon(model, val_data, t_norm, x_norm, y_norm,
                               sx, sy, T_obs=T_obs, device=device,
                               n_samples=args.n_eval)
        T = data.shape[1]
        print(f"  {T_obs:>4}h     {T-T_obs:>4} future     {rl2*100:>6.2f}%")


if __name__ == "__main__":
    main()
