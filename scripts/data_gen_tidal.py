"""Generate simulation windows using tidal SSH forcing + ocean sensor positions."""
import numpy as np
import os
import multiprocessing as mp
import argparse
from solver_2d.swe_hll_real_2d import simulate_real_2d

def run_scenario(args):
    idx, ssh_slice, bathymetry, outdir, sensor_yx = args
    sim_name = f"sim_2d_{idx:04d}"
    outpath = os.path.join(outdir, f"{sim_name}.npz")
    if os.path.exists(outpath):
        return f"Skipping {sim_name}, already exists."
    try:
        t_timeseries = np.arange(len(ssh_slice))
        out_t, out_h = simulate_real_2d(
            bathymetry=bathymetry,
            ssh_timeseries=ssh_slice,
            t_timeseries=t_timeseries,
            save_every=2000,
            sensor_yx=sensor_yx
        )
        np.savez_compressed(outpath, t=out_t, h=out_h, ssh_input=ssh_slice)
        return f"Completed {sim_name}: {len(out_t)} frames, h=[{out_h.min():.3f}, {out_h.max():.3f}]"
    except Exception as e:
        return f"Failed {sim_name}: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", default="data/simulations_2d_tidal")
    parser.add_argument("--window-hours", type=int, default=168)
    parser.add_argument("--stride-hours", type=int, default=12)
    parser.add_argument("--max-sims", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    bathymetry = np.load("data/processed/elev_grid.npy")
    ssh = np.load("data/processed/ssh_tidal_20s.npy")
    sp = np.load("data/processed/sensor_positions.npz")
    sensor_yx = {
        "south": [(int(y), int(x)) for y, x in sp["south_yx"]],
        "east": [(int(y), int(x)) for y, x in sp["east_yx"]],
    }

    Nt = ssh.shape[0]
    tasks = []
    sim_idx = 0
    start = 0
    while start + args.window_hours <= Nt and sim_idx < args.max_sims:
        ssh_slice = ssh[start:start + args.window_hours]
        tasks.append((sim_idx, ssh_slice, bathymetry, args.outdir, sensor_yx))
        start += args.stride_hours
        sim_idx += 1

    print(f"Prepared {len(tasks)} tasks.")

    if args.workers == 1:
        for t in tasks:
            print(run_scenario(t))
    else:
        with mp.Pool(args.workers) as pool:
            for result in pool.imap_unordered(run_scenario, tasks):
                print(result)

if __name__ == "__main__":
    main()
