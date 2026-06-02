# Project Journey — Sparse-Sensor Tidal Reconstruction & Forecasting with DeepONet

> **TTCS Course Project** — Gulf of Tonkin
> From Week 1 (1D Poc) → Week 10 (Causal Forecasting breakthrough)

---

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [Week 1–4: 1D Proof of Concept](#2-week-14-1d-proof-of-concept)
3. [Week 5–6: 2D Solver Bring-up](#3-week-56-2d-solver-bring-up)
4. [Week 7–8: PDEBench Benchmark](#4-week-78-pdebench-benchmark)
5. [Week 9: The Great Debug (93.6% → 14.8%)](#5-week-9-the-great-debug-936--148)
6. [Week 10: Causal Forecasting Pivot](#6-week-10-causal-forecasting-pivot)
7. [Final Results Summary](#7-final-results-summary)
8. [Repository Structure](#8-repository-structure)
9. [Key Lessons Learned](#9-key-lessons-learned)

---

## 1. Project Goal

**Original question:** Can a Deep Operator Network (DeepONet) learn the tidal dynamics operator that maps sparse boundary SSH observations → full sea surface field η(x, y, t) over the Gulf of Tonkin?

**Evolved question (Week 10):** Can the same operator perform **causal forecasting** — observing only past boundary sensors and predicting the entire future field?

**Why DeepONet:**
- Branch/trunk architecture decouples input function (sensors) from query points (x,y,t)
- Natural fit for sparse-input → dense-output problems
- Resolution-independent (unlike CNN-based approaches)

---

## 2. Week 1–4: 1D Proof of Concept

### Goal
Validate the DeepONet idea on 1D Shallow Water Equations before scaling to 2D.

### Built
- **1D HLL Godunov solver** (`archive/1d_poc/solver/`)
  - Riemann solver with CFL-adaptive timestep
  - Reflective & transmissive boundary conditions
  - Test cases: dam break, standing wave, tidal forcing
- **1D DeepONet** — branch: SSH(t) at 1 boundary point, trunk: (x, t)
- **Ablation study** — h (water depth) vs η (anomaly) as target

### Findings
- η (anomaly) trained better than h (total depth) — removes large static baseline
- Model successfully reconstructed wave propagation from 1 boundary sensor
- **Moved files to `archive/1d_poc/`** — kept for reference only

### Deliverable
- `report_week5_6.pdf` summarized 1D results

---

## 3. Week 5–6: 2D Solver Bring-up

### Goal
Extend to 2D Shallow Water on a real bathymetry (Gulf of Tonkin).

### Built
- **Well-balanced 2D HLL solver** (`solver_2d/swe_hll_real_2d.py`)
  - Hydrostatic Reconstruction (HR) for C-property
  - Wet/dry handling
  - Hard Dirichlet BC (later replaced)
- **GEBCO bathymetry processing** (`scripts/process_static_data.py`)
  - Downloaded `GEBCO_2025` tile for Vịnh Bắc Bộ
  - Processed to 130×110 grid (5 km resolution)
- **Initial tidal forcing** — used **Copernicus SLA** (Sea Level Anomaly)
- **2D DeepONet** — branch: SSH(168h, 20 sensors), trunk: (x, y, t)

### Issue (undetected at the time)
- Copernicus SLA **has tide removed** — amplitude only ±0.07 m
- Boundary sensors placed at grid edge row 0 / col 109 → **all on land** (bath=0)
- Solver source term had sign bug violating C-property

### Initial result
**RelL2 = 93.6%** on test set — essentially no better than predicting the mean.

---

## 4. Week 7–8: PDEBench Benchmark

### Goal
Validate the 2D DeepONet architecture on a **clean, standard benchmark** (PDEBench 2D Radial Dam Break) to isolate whether the issue was the architecture or the GoT data/physics.

### Setup
- **Dataset:** `2D_rdb_NA_NA.h5` — 1000 sims, 101 timesteps, 128×128 grid
- **Three variants:**
  1. **Full-State DeepONet** — branch sees entire IC (128×128 = 16,384 inputs)
  2. **Boundary DeepONet** — branch sees 16 sensors × 101 timesteps (non-causal MLP)
  3. **FNO reference** — PDEBench paper baseline

### Results (published in `report_week7_8.tex`)

| Model | Rel L2 | Comment |
|-------|--------|---------|
| FNO (PDEBench paper) | 5.13% | Full field input |
| **Full-State DeepONet** | **3.74%** | Best — full IC available |
| Boundary DeepONet (MLP) | 5.25% | Only 16 sensors, beat FNO |

### Critical conclusion
**The architecture works.** Boundary DeepONet with only **0.1% of grid as sensors** beat FNO. The 93.6% error on GoT was therefore **not an architecture problem** — it had to be data/physics.

→ This conclusion set up Week 9's forensic debug.

---

## 5. Week 9: The Great Debug (93.6% → 14.8%)

### The puzzle
- PDEBench: 5.25% (clean)
- Gulf of Tonkin: 93.6% (broken)
- Same model architecture

### 4-Layer Systematic Debug

Documented in `docs/debug_results.md`. Tested 5 hypotheses:

| # | Hypothesis | Verdict |
|---|-----------|---------|
| H1 | Wrong target η definition | ❌ Correct |
| H2 | Boundary preprocessing | ✅ **Partial** — sensors on land (bath=0) |
| H3 | Solver mass drift | ✅ **Confirmed** — monotonic η increase, no oscillation |
| H4 | Extreme values at hotspots | ✅ **Confirmed** — max 4.4 m at land wetting cells |
| H5 | Branch input useless | ✅ **Confirmed** — DeepONet ≈ time-only baseline |

### Six Fixes Applied

**Fix 1 — C-property bug** (`solver_2d/swe_hll_real_2d.py`):
```python
# Before: -= doubled the imbalance
# After:  += cancels flux divergence
U[1:-1, 1:-1, 1] += (dt / dx) * (Sx_right - Sx_left)
```
→ Lake-at-rest test: η = 0 preserved exactly.

**Fix 2 — Tidal forcing** (`generate_tidal_forcing.py`):
Replaced Copernicus SLA with 5 synthetic constituents:

| Constituent | Period | Amplitude |
|------------|--------|-----------|
| K1 | 23.93 h | 0.50 m |
| O1 | 25.82 h | 0.40 m |
| P1 | 24.07 h | 0.15 m |
| M2 | 12.42 h | 0.20 m |
| S2 | 12.00 h | 0.10 m |

→ Total amplitude ±1.3 m — matches real GoT (Hải Phòng K1 ~0.8m).

**Fix 3 — Sensor placement**:
- Before: 20 sensors at grid edge → all on land (bath=0)
- After: Moved to ocean cells (bath < -23m for South, bath < -40m for East)

**Fix 4 — Relaxation boundary conditions**:
Hard Dirichlet → shock waves → blow-up. Replaced with nudging (α=0.05):
```python
nudged_h = (1 - alpha) * h_current + alpha * target_h
```
→ η bounded ∈ [-1.52, +1.02] m over 24h test.

**Fix 5 — Ocean-only sampling** (`scripts/dataset_builder_2d.py`):
Applied bathymetry mask — sample only from 7,645 ocean cells (53.5% of grid).

**Fix 6 — Stratified window split**:
Old test set [16,17,18,19] was all spring tide (distribution shift). New split:

| Window | Regime | η std |
|--------|--------|-------|
| 4 | Neap | 18.4 cm |
| 9 | Neap/medium | 16.9 cm |
| 14 | Medium/large | 25.4 cm |
| 19 | Spring | 31.7 cm |

Plus: training script rewritten to **filter by `train_windows`**, eliminating leakage.

### Architecture Enhancement — Fourier Positional Encoding

After all 6 fixes, RelL2 = 27.1%. Still limited by **spectral bias** of MLPs.

Added **Fourier PE** to trunk network:
```python
# (x, y, t) → 51 dims via sin/cos at frequencies 2^0..2^7
fourier_features = [x, y, t, sin(πx), cos(πx), sin(2πx), cos(2πx), ..., sin(128πt), cos(128πt)]
```

Effect:
- Trunk input 3D → 51D (8 frequency bands)
- Latent dim 128 → 256
- Params: 1.32M → 1.40M (+6%)
- **RelL2: 27.1% → 14.8%**

### Final Week 9 Results

| Run | Config | RMSE | RelL2 |
|-----|--------|------|-------|
| Original | Copernicus SLA, land sensors, random split | ~50 cm | 93.6% |
| Run 3 | Tidal + ocean sensors + stratified | 6.45 cm | 27.1% |
| **Run 4** | **+ Fourier PE + latent 256** | **3.54 cm** | **14.8%** |

Per-window breakdown:

| Window | RMSE | RelL2 |
|--------|------|-------|
| 4 (neap) | 2.31 cm | 12.5% |
| 9 (neap/med) | 3.19 cm | 18.9% |
| 14 (medium) | 3.75 cm | 14.8% |
| 19 (spring) | 4.52 cm | 14.3% |

→ Model generalizes across tidal regimes (neap to spring).

Documented in `report_week9.tex` and `docs/project_report.md`.

---

## 6. Week 10: Causal Forecasting Pivot

### New goal
Upgrade from **reconstruction** (non-causal, sees all times) to **causal forecasting** (only sees past).

### Key distinction
- **Reconstruction (Week 7-8 Boundary DeepONet):** branch = full 101-timestep history → predict any (x,y,t). Not causal.
- **Forecasting (Week 10):** branch = history up to T_obs only → predict future t > T_obs. Strictly causal.

### Architecture — `ForecastDeepONet`

Documented in `forecasting/model.py`:

```
Branch (causal):
  sensor_hist[0..T_obs]  →  LSTM(16 → 256, 2 layers)
                         →  Linear(256 → 256)
                         →  b ∈ R^256

Trunk (spatial):
  (x, y, t_future)       →  FourierPE(8 freqs) [3 → 51]
                         →  MLP[256 × 4] → 256
                         →  t ∈ R^256

Output: η = <b, t> + bias
```

**Causality enforced by LSTM** — it only processes past observations.

### Training trick — random T_obs
During training, `T_obs` is sampled per-sample in [10, 80]. This makes the model **robust across any forecast horizon** — it learns to forecast from 10h of history or 80h of history.

### Dataset
- **PDEBench 2D SWE** (Radial Dam Break) — 1000 sims, 101 timesteps, 128×128
- 16 sensors on boundary (4 per edge) = 0.1% of grid
- 800 sims train / 200 val
- Points per sample: 512 future (x,y,t) queries

### Results

**Training:**
- 100 epochs in 174 seconds (GPU)
- Best epoch: 98
- Train Rel L2: 4.38%, Val Rel L2: 4.36% → gap -0.02% (no overfitting)

**Multi-horizon evaluation:**

| T_obs observed | T_future predicted | Rel L2 | PDEBench FNO baseline |
|---------------|-------------------|--------|----------------------|
| 20 timesteps | 81 future | 4.72% | 9.80% |
| 40 timesteps | 61 future | 4.43% | 9.52% |
| 60 timesteps | 41 future | 4.33% | 9.82% |
| 80 timesteps | 21 future | 4.46% | 10.23% |
| **Average** | — | **4.46%** | ~10% |

### Key insight
**Causal LSTM branch (4.46%) beat non-causal MLP branch (5.25%)** — despite only seeing the past. This shows:
1. Temporal encoding via LSTM > flat concatenation via MLP
2. The operator is genuinely learnable from causal inputs
3. No forecast-horizon degradation → uniform accuracy

→ Documented in `docs/project_report.md` §11 and `docs/report_week9.tex` week-10 plan.

---

## 7. Final Results Summary

### Headline Numbers

| Setting | Model | Sensors | Rel L2 | Task |
|---------|-------|---------|--------|------|
| Gulf of Tonkin | Fourier DeepONet | 20 ocean | **14.8%** | Reconstruction |
| PDEBench SWE 2D | **ForecastDeepONet** | 16 boundary (0.1%) | **4.46%** | **Causal Forecasting** |
| PDEBench SWE 2D | Full-State DeepONet | Full IC | 3.74% | Reconstruction |
| PDEBench SWE 2D | Boundary DeepONet (MLP) | 16 boundary | 5.25% | Reconstruction |
| PDEBench SWE 2D | FNO (reference) | Full field | 5.13% | Reconstruction |

### Runtime comparison (GoT)
| Method | Time to produce 48h field |
|--------|--------------------------|
| Classical HLL Godunov solver | ~6 hours |
| DeepONet inference | ~10 seconds |
| **Speedup** | **~2,000×** |

### What works scientifically
- Physics: well-balanced HLL + HR satisfies C-property (lake-at-rest)
- Data: realistic tidal amplitudes ±1.3m, ocean-only sampling, stratified split
- Architecture: Fourier PE fights spectral bias of MLP trunk
- Causality: LSTM branch ensures no future leakage

### What's still missing for a real digital twin
| Component | Impact | Complexity |
|-----------|--------|------------|
| Coriolis force | Large (shapes wave pattern) | Medium |
| Bottom friction | Medium (shallow dissipation) | Easy |
| TPXO real tidal forcing | Medium | Medium |
| Meteorological forcing | Small | Hard |

**Verdict:** ~60–70% realistic vs a production oceanographic simulation. Sufficient for TTCS; not yet sufficient for peer-reviewed oceanography.

---

## 8. Repository Structure

```
ttcs/
├── README.md                         # Project overview & quick start
│
├── solver_2d/                        # 2D SWE solver
│   ├── swe_hll_real_2d.py            # Well-balanced HLL + HR (GoT)
│   └── swe_hll_2d.py                 # Base HLL (PDEBench)
│
├── deeponet/                         # Reconstruction model
│   ├── model.py                      # DeepONet + FourierFeatures
│   ├── train_deeponet_2d.py          # Window-filtered training
│   ├── data.py                       # NPZ / array datasets
│   ├── metrics.py                    # RMSE, MAE
│   └── utils.py                      # Seed, device, checkpoints
│
├── forecasting/                      # Causal forecasting (Week 10)
│   ├── model.py                      # ForecastDeepONet (LSTM branch)
│   ├── dataset.py                    # PDEBench causal dataset
│   ├── train.py                      # Variable T_obs training
│   └── eval.py                       # Multi-horizon evaluation
│
├── scripts/                          # Data pipeline
│   ├── data_gen_tidal.py             # GoT simulations
│   ├── dataset_builder_2d.py         # Stratified dataset builder
│   ├── eval_deeponet.py              # GoT evaluation
│   ├── benchmark_pdebench_swe.py     # PDEBench SWE benchmark
│   ├── benchmark_pdebench_deeponet.py # PDEBench DeepONet benchmark
│   └── process_static_data.py        # GEBCO bathymetry processing
│
├── generate_tidal_forcing.py         # Synthetic tidal constituents
│
├── docs/
│   ├── project_report.md             # Full technical report
│   ├── project_journey.md            # This file — project wrap-up
│   ├── debug_results.md              # 4-layer debug findings
│   ├── data_architecture_analysis.md # Dataset analysis
│   ├── report_week7_8.tex            # Week 7-8 LaTeX report
│   ├── report_week9.tex              # Week 9 LaTeX report
│   └── paper.tex                     # Paper draft
│
├── archive/                          # Deprecated files (1D PoC, old reports)
└── papers/                           # Reference PDFs (gitignored)
```

### Checkpoints on server (`namnx@171.226.10.121`)
- `deeponet_2d_strat_best.pt` — Run 3 (no Fourier) → 27.1%
- `deeponet_2d_fourier_best.pt` — Run 4 (Fourier PE, latent 256) → **14.8%**
- `forecast_best.pt` — ForecastDeepONet → **4.46%**

---

## 9. Key Lessons Learned

### Physics & numerics
1. **Always test C-property first.** A sign bug in the source term can silently violate lake-at-rest and create mass drift that contaminates everything downstream.
2. **Hard Dirichlet BC explodes.** Relaxation nudging (α ~0.05) is essential for long-term stability.
3. **Domain knowledge matters.** Using Copernicus SLA (tide-removed) vs raw SSH is a one-line-of-code mistake that invalidates the entire experiment.

### ML methodology
1. **Sensor placement is data.** Sensors on land cells produce zero signal — model can't learn from dead inputs, no matter how clever.
2. **Test-set design is critical.** Consecutive windows as test = distribution shift. Always stratify.
3. **Random split leaks.** For window-structured data, random 90/10 within the full dataset = information leakage. Must filter training to `train_windows` explicitly.
4. **Spectral bias is real.** MLPs with raw (x,y,t) can't represent high-frequency spatial patterns. Fourier PE (NeRF-style) is a cheap fix.
5. **Fourier PE ≠ FNO.** PE encodes query coordinates (static sin/cos). FNO does learnable frequency filters on full fields. Different tools for different jobs.

### Causal forecasting
1. **LSTM branch enforces causality structurally.** No need for masking tricks — the model simply cannot see the future.
2. **Random T_obs during training = horizon robustness.** Model learns to forecast from any amount of history without degradation.
3. **Causal can beat non-causal.** LSTM (4.46%) < MLP (5.25%) shows that temporal encoding > flat concatenation, even when the MLP has strictly more information.

### Engineering
1. **Debug in layers.** Don't chase one hypothesis — enumerate all possible failure modes (physics, data, architecture, split), then eliminate systematically.
2. **Baselines are non-negotiable.** "Predict the mean" as a baseline exposed that the original DeepONet was barely learning anything.
3. **Document as you go.** `debug_results.md` was written during debugging, not after — made it possible to recover context across sessions.

---

**Project status:** Both tracks (reconstruction on GoT + forecasting on PDEBench) completed. Repository organized, reports written, code reproducible. Ready for presentation and/or paper submission.
