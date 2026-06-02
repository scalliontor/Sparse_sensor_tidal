import torch
import numpy as np
from dataset_ovcno import CopernicusOVCNODataset
from model_ovcno import ObservabilityAwareVCNO
from model_vae import ForecastDeepONetVAE
from scipy.stats import norm, spearmanr

class HorizonCopernicusOVCNODataset(CopernicusOVCNODataset):
    def __init__(self, horizon_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon_idx = horizon_idx
        
    def __getitem__(self, start_idx):
        # We fix T_obs to 24 for a consistent horizon baseline, or keep random
        # Let's keep T_obs as 24 to match standard history length
        T_obs = 24 
        k = self.max_sensors
        
        # Draw k sensors consistently
        # Use a fixed seed for exactly reproducible validation layout per step
        rng = np.random.default_rng(start_idx)
        active_sensor_idx = rng.choice(len(self.master_sensor_coords), k, replace=False)
        active_coords = self.master_sensor_coords[active_sensor_idx]
        
        sensor_hist = np.zeros((T_obs, k), dtype=np.float32)
        sensor_pts = np.zeros((k, 2), dtype=np.float32)
        
        for i, (y, x) in enumerate(active_coords):
            sensor_hist[:, i] = self.data[start_idx : start_idx + T_obs, y, x]
            sensor_pts[i, 0] = self.x_norm[x]
            sensor_pts[i, 1] = self.y_norm[y]
            
        future_start = start_idx + T_obs
        t_idx_scalar = future_start + self.horizon_idx
        if t_idx_scalar >= self.T:
            t_idx_scalar = self.T - 1
            
        # Use full ocean coordinates
        y_idx = self.ocean_coords[:, 0]
        x_idx = self.ocean_coords[:, 1]
        t_idx = np.full(len(y_idx), t_idx_scalar)
        
        q_x = self.x_norm[x_idx]
        q_y = self.y_norm[y_idx]
        
        dx = q_x[:, None] - sensor_pts[None, :, 0]
        dy = q_y[:, None] - sensor_pts[None, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        
        trunk = np.stack([
            q_x, q_y, self.t_norm[t_idx], min_dists
        ], axis=-1).astype(np.float32)
        
        labels = self.data[t_idx_scalar, y_idx, x_idx].astype(np.float32)
        
        return (torch.tensor(sensor_hist, dtype=torch.float32),
                torch.tensor(sensor_pts, dtype=torch.float32),
                torch.tensor(trunk, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32))

def evaluate_horizon(model, model_type, ds, device):
    model.eval()
    all_mu, all_lab, all_lv = [], [], []
    with torch.no_grad():
        # Evaluate over a subset of temporal steps to save time, or all.
        # Let's take step of 6 hours to speed up, or all if it's small (e.g. 100 steps)
        # Validation size is typically 258 steps. Let's do all.
        for i in range(len(ds) - 24): # Ensure enough room for +24h
            h, p, t, l = ds[i]
            h = h.unsqueeze(0).to(device)
            p = p.unsqueeze(0).to(device)
            t = t.to(device) 
            
            if model_type == "ovcno":
                mu, lv, _, _, _ = model(h, p, t)
            else:
                mu, lv, _, _ = model(h, t[:, :3])
                
            all_mu.append(mu.cpu().numpy().flatten())
            all_lab.append(l.numpy().flatten())
            all_lv.append(lv.cpu().numpy().flatten())
            
    mu = np.concatenate(all_mu)
    lab = np.concatenate(all_lab)
    lv = np.concatenate(all_lv)
    std = np.exp(0.5 * lv)
    
    rmse = np.sqrt(np.mean((mu - lab)**2))
    mae = np.mean(np.abs(mu - lab))
    nll = 0.5 * np.mean(lv + (lab - mu)**2 / np.exp(lv))
    z95 = 1.96
    cov = np.mean((lab >= mu - z95*std) & (lab <= mu + z95*std))
    avgw = np.mean(2 * z95 * std)
    errs = np.abs(mu - lab)
    cs, _ = spearmanr(errs, std)
    z = (lab - mu) / std
    crps = np.mean(std * (z * (2*norm.cdf(z) - 1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi)))
    
    return rmse, mae, nll, crps, cov, avgw, cs

device = "cuda" if torch.cuda.is_available() else "cpu"
nc = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"

# Load models
print("Loading models...")
ovcno = ObservabilityAwareVCNO(lstm_hidden=256, latent_dim=256, width=256).to(device)
ovcno.load_state_dict(torch.load("ovcno_checkpoint.pt", map_location=device))

vco = ForecastDeepONetVAE(n_sensors=16, lstm_hidden=256, latent_dim=256, width=256).to(device)
vco.load_state_dict(torch.load("ckpt_VCO_baseline.pt", map_location=device))

# Baseline train dataset to get mean
train_ds = CopernicusOVCNODataset(nc, n_sensors=16, pts_per_sample=512,
    T_obs_min=24, T_obs_max=72, seed=42, split="train")
train_mean = train_ds.train_mean

horizons = [(1, 0), (3, 2), (6, 5), (12, 11), (24, 23)]  # (Hours, array index)

print("="*90)
print(f"{'Lead Time':<10} {'RMSE':>7} {'MAE':>7} {'NLL':>7} {'CRPS':>7} {'Cov95':>7} {'AvgW':>7} {'CorrS':>7}")
print("="*90)

for hr, idx in horizons:
    ds = HorizonCopernicusOVCNODataset(horizon_idx=idx, nc_path=nc, n_sensors=16, 
         pts_per_sample=1024, T_obs_min=24, T_obs_max=24, seed=42, split="val", train_mean=train_mean)
         
    # VCO
    rmse, mae, nll, crps, cov, avgw, cs = evaluate_horizon(vco, "vco", ds, device)
    print(f"VCO  +{hr}h   {rmse:>7.4f} {mae:>7.4f} {nll:>7.3f} {crps:>7.4f} {cov*100:>6.1f}% {avgw:>7.3f} {cs:>7.3f}")
    
    # OVCNO
    rmse, mae, nll, crps, cov, avgw, cs = evaluate_horizon(ovcno, "ovcno", ds, device)
    print(f"OVCNO +{hr}h   {rmse:>7.4f} {mae:>7.4f} {nll:>7.3f} {crps:>7.4f} {cov*100:>6.1f}% {avgw:>7.3f} {cs:>7.3f}")
    print("-" * 90)
