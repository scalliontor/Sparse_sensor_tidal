# Sparse-Sensor Tidal Reconstruction & Forecasting with DeepONet

> **Course Project** — Thực tập Cơ sở (TTCS)  
> Gulf of Tonkin tidal field reconstruction from sparse boundary observations using Deep Operator Networks.

---

## Overview

This project demonstrates that a **Deep Operator Network (DeepONet)** can learn the tidal dynamics operator mapping sparse sensor observations to full-field sea surface height (η) predictions, achieving:

| Model | Dataset | Sensors | Rel L2 | Task |
|-------|---------|---------|--------|------|
| **Fourier DeepONet** | Gulf of Tonkin | 20 ocean sensors | **14.8%** | Reconstruction |
| **ForecastDeepONet** | PDEBench SWE 2D | 16 boundary (0.1%) | **4.46%** | Causal Forecasting |
| Full-State DeepONet | PDEBench SWE 2D | Full IC | 3.74% | Reconstruction |
| Boundary-DeepONet | PDEBench SWE 2D | 16 boundary | 5.25% | Reconstruction |
| FNO (reference) | PDEBench SWE 2D | Full field | 5.13% | Reconstruction |

**Key insight:** An LSTM-based causal branch that only sees past sensor data (4.46%) outperforms a non-causal MLP branch that sees the entire time series (5.25%).

---

## Project Structure

```
├── solver_2d/                     # 2D Shallow Water Equations solver
│   ├── swe_hll_real_2d.py         # Well-balanced HLL + Hydrostatic Reconstruction
│   └── swe_hll_2d.py              # Base HLL solver (PDEBench-style)
│
├── deeponet/                      # DeepONet for reconstruction (non-causal)
│   ├── model.py                   # DeepONet + FourierFeatures classes
│   ├── train_deeponet_2d.py       # Training with stratified window split
│   ├── data.py                    # NPZ dataset loader
│   ├── metrics.py                 # RMSE, MAE
│   └── utils.py                   # Seed, device, checkpoint utilities
│
├── forecasting/                   # ForecastDeepONet for causal prediction
│   ├── model.py                   # LSTM branch + Fourier trunk DeepONet
│   ├── train.py                   # Causal training with random T_obs horizons
│   ├── eval.py                    # Evaluation by forecast horizon
│   └── dataset.py                 # Causal dataset builder
│
├── scripts/                       # Data generation & evaluation pipelines
│   ├── data_gen_tidal.py          # Generate simulations with tidal forcing
│   ├── dataset_builder_2d.py      # Build training dataset (ocean-masked, stratified)
│   ├── eval_deeponet.py           # Evaluate Fourier DeepONet
│   ├── benchmark_pdebench_swe.py  # PDEBench SWE 2D benchmark
│   ├── benchmark_pdebench_deeponet.py
│   └── process_static_data.py     # GEBCO bathymetry processing
│
├── generate_tidal_forcing.py      # Synthetic tidal SSH (K1+O1+P1+M2+S2)
│
├── docs/                          # Reports and documentation
│   ├── report_week9.tex           # Week 9 report (debug + Fourier DeepONet)
│   ├── report_week7_8.tex         # Week 7-8 report (PDEBench benchmark)
│   ├── paper.tex                  # Paper draft
│   ├── project_report.md          # Technical project report
│   ├── debug_results.md           # 4-layer systematic debug findings
│   └── data_architecture_analysis.md
│
└── papers/                        # Reference papers
```

---

## Method

### 1. Physics Solver: Well-Balanced HLL

2D Shallow Water Equations solved with:
- **HLL Riemann solver** with adaptive CFL timestep
- **Hydrostatic Reconstruction** for exact lake-at-rest preservation (C-property)
- Relaxation nudging boundary conditions (avoids shock blow-up)
- Tidal forcing with 5 constituents: K1, O1, P1, M2, S2 (±1.3m amplitude)

### 2. Fourier DeepONet (Reconstruction)

```
Branch: SSH(20 sensors × 168h) → MLP [256×4] → R^256
Trunk:  (x,y,t) → FourierPE(8 freqs) → 51D → MLP [256×4] → R^256
Output: η = Σ b_k · t_k + bias
```

Fourier Positional Encoding lifts 3 scalar coordinates to 51 dimensions using sin/cos at 8 frequency bands, enabling the trunk to represent high-frequency spatial patterns.

### 3. ForecastDeepONet (Causal Forecasting)

```
Branch: SSH(16 sensors, t=0..T_obs) → LSTM(256, 2 layers) → R^256
Trunk:  (x,y,t) → FourierPE → MLP → R^256
Output: η(x,y,t) for t > T_obs
```

The LSTM branch enforces **causality** — it only processes past observations. During training, T_obs is randomly sampled to make the model robust across all forecast horizons.

---

## Key Results

### Gulf of Tonkin (Real Bathymetry)

| Run | Description | RMSE | Rel L2 |
|-----|-------------|------|--------|
| Baseline | Copernicus SLA, land sensors | ~50cm | 93.6% |
| + Physics fixes | Tidal forcing, ocean sensors, ocean mask | 6.45cm | 27.1% |
| **+ Fourier PE** | **Fourier encoding + latent 256** | **3.54cm** | **14.8%** |

### PDEBench SWE 2D (Causal Forecasting)

| Observation Window | Future Steps | Rel L2 |
|--------------------|-------------|--------|
| 20 timesteps | 81 future | 4.72% |
| 40 timesteps | 61 future | 4.43% |
| 60 timesteps | 41 future | 4.33% |
| 80 timesteps | 21 future | 4.46% |
| **Average** | | **4.46%** |

No forecast degradation — accuracy is stable regardless of prediction horizon.

---

## Reproduce

### Prerequisites

```bash
pip install numpy scipy matplotlib torch
```

### 1. Generate tidal forcing + run simulations

```bash
python generate_tidal_forcing.py
python scripts/data_gen_tidal.py --workers 4 --max-sims 20
```

### 2. Build stratified dataset

```bash
python scripts/dataset_builder_2d.py \
  --sims_dir data/simulations_2d_tidal \
  --out data/dataset_2d_tidal_strat.npz \
  --points_per_window 20000 \
  --train_windows '0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18' \
  --test_windows '4,9,14,19'
```

### 3. Train Fourier DeepONet

```bash
cd deeponet
python train_deeponet_2d.py \
  --data ../data/dataset_2d_tidal_strat.npz \
  --epochs 300 --batch 8192 --lr 1e-3 \
  --latent 256 --n_fourier_freqs 8
```

### 4. Train ForecastDeepONet

```bash
cd forecasting
python train.py --epochs 100 --batch 512
```

### 5. Evaluate

```bash
python scripts/eval_deeponet.py \
  --data data/dataset_2d_tidal_strat.npz \
  --ckpt checkpoints/deeponet_2d_fourier_best.pt

cd forecasting
python eval.py --ckpt ../checkpoints/forecast_best.pt
```

---

## Debug Journey: 93.6% → 14.8%

The project included an extensive **4-layer systematic debug** that reduced error from 93.6% to 14.8%:

1. **Physics Layer:** Fixed sign bug in hydrostatic source term (C-property violation)
2. **Forcing Layer:** Replaced Copernicus SLA (±0.07m, no tides) with synthetic tidal constituents (±1.3m)
3. **Sensor Layer:** Moved all 20 sensors from land cells (bath=0) to ocean cells (bath<0)
4. **Dataset Layer:** Applied ocean mask + stratified train/test split by tidal regime

See [`docs/debug_results.md`](docs/debug_results.md) for the full forensic analysis.

---

## References

1. Lu, L. et al. (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence*.
2. Takamoto, M. et al. (2022). PDEBench. *NeurIPS 2022*.
3. Li, Z. et al. (2021). Fourier Neural Operator. *ICLR 2021*.
4. Mildenhall, B. et al. (2020). NeRF. *ECCV 2020*.
5. Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.

---

## License

This project is for educational purposes (TTCS course project).
