# OVCNO — Observability-Aware Causal Neural Operators for Sparse-Boundary Ocean Forecasting

> **Course Project** — Thực tập Cơ sở (TTCS), PTIT
> Probabilistic full-field sea-surface-height forecasting over the Gulf of Tonkin from sparse boundary sensor histories, with learned per-query observability.

---

## Overview

This repository contains the full pipeline for **OVCNO** — an Observability-Aware Variational Causal Neural Operator — that learns to forecast the entire sea-surface-height (SSH) field of a coastal domain from only $K{\ll}N_xN_y$ boundary sensor histories.

Beyond predicting the future field, OVCNO learns a **data-driven observability proxy** $o_\psi(x,y,t) \in [0,1]$ that estimates how strongly each query location is constrained by the available sensor configuration, and uses it to condition the predictive distribution. The model thus tells the operator not only *what* it predicts, but also *where* it should be trusted.

### Headline results — Copernicus SSH OSSE, Gulf of Tonkin (Jan 2024, K=16 boundary sensors)

| Model | RMSE$\downarrow$ | NLL$\downarrow$ | CRPS$\downarrow$ | Cov@95 | Corr$_S\uparrow$ |
|---|---|---|---|---|---|
| Optimal Interpolation | 0.114 | — | — | — | — |
| Persistence | 0.082 | — | — | — | — |
| Deterministic Causal Op. | 0.068 | — | — | — | — |
| VCO (variational baseline) | 0.077 | −1.92 | 0.043 | 79.7% | 0.446 |
| **OVCNO (ours)** | **0.068** | **−2.39** | **0.036** | **94.8%** | **0.523** |

OVCNO improves probabilistic reliability (coverage 79.7%→94.8%, NLL −1.92→−2.39) while maintaining point accuracy comparable to the deterministic baseline. Trade-off: predictive intervals are wider (Avg.W 0.209→0.287). Per-horizon evaluation shows OVCNO maintains 93–95% coverage from +1h to +24h while VCO degrades from 84% to 77%.

### Headline results — PDEBench 2D SWE (16 boundary sensors = 0.098% of grid)

| Model | Input regime | Rel-$L_2$ | RMSE | NLL | Cov@95 |
|---|---|---|---|---|---|
| U-Net 2D (PDEBench paper) | Dense/full state | ~12.80% | — | — | — |
| FNO-2D (PDEBench paper) | Dense/full state | 5.13% | — | — | — |
| VCO | 16 boundary | 4.42% | 0.0121 | −3.12 | 94.8% |
| **OVCNO (ours)** | **16 boundary** | **4.38%** | **0.0118** | **−3.25** | **95.2%** |

On the controlled PDEBench benchmark, OVCNO achieves competitive point accuracy while operating on **0.1% of the spatial grid**.

📄 **Paper draft:** [`docs/paper.pdf`](docs/paper.pdf) · [`docs/paper.tex`](docs/paper.tex)

---

## What's in this repo

```
ttcs/
├── docs/
│   ├── paper.tex / paper.pdf       Paper draft (29 pages)
│   ├── figures/                    20 paper figures
│   ├── generate_figures.py         Regenerate paper figures
│   ├── project_journey.md          Week-by-week narrative
│   ├── project_report.md           Full technical report
│   ├── report_week*.tex            Weekly LaTeX reports
│   └── eval.md                     Comparison tables and trade-offs
│
├── experiment/                     OVCNO pipeline (Week 11, paper-track)
│   ├── model_ovcno.py              Main OVCNO architecture
│   ├── model_ovcno_v2.py           Decoupled mean/variance variant (HYCOM study)
│   ├── model_vae.py                VCO baseline
│   ├── dataset_ovcno.py            Copernicus dataset
│   ├── dataset_ovcno_layout.py     Real-station layout variants
│   ├── dataset_hycom.py            HYCOM cross-product loader
│   ├── train_layout.py             Multi-layout OSSE trainer
│   ├── train_hycom_*.py            HYCOM smoke/full/diagnostic runs
│   ├── eval_*.py                   5 evaluation scripts (horizon, layout,
│   │                               missing-sensor, ablation, uncertainty)
│   ├── plot_*.py                   8 plotting scripts
│   ├── audit_copernicus_corr.py    Sanity-check diagnostics (StdRatio, partial corr)
│   ├── diagnose_copernicus.py      Spatial collapse diagnostics
│   └── sensors_real_stations.json  GLOSS/PSMSL/IOC station coordinates
│
├── solver_2d/                      2D shallow-water solver (used by Week 9)
│   ├── swe_hll_real_2d.py          Well-balanced HLL + Hydrostatic Reconstruction
│   └── swe_hll_2d.py               PDEBench-style base HLL
│
├── deeponet/                       Week 9 Fourier DeepONet (reconstruction)
├── forecasting/                    Week 10 ForecastDeepONet (causal LSTM branch)
├── scripts/                        Data pipeline (GoT simulations, HYCOM download,
│                                   dataset builder, PDEBench benchmark drivers)
├── generate_tidal_forcing.py       Synthetic K1+O1+P1+M2+S2 tidal SSH
│
├── bao_cao_tuan/                   Vietnamese weekly reports + final_report (PTIT)
├── hycom_complete_report.md        HYCOM cross-product investigation journal
├── hycom_v2_experiments.md         OVCNOv2 decoupled-architecture follow-up
├── audit_250_qa.md                 250-question pre-submission self-audit
├── guide.md                        Operational benchmark strategy notes
│                                   (NOAA CBOFS, GLOSS, HF radar options)
└── reasoning.md                    Coastal-vs-altimetry framing rationale
```

---

## Method — OVCNO architecture

OVCNO has five components:

1. **Sensor-geometry encoder.** Each sensor $k$ at time $t$ becomes a coordinate-valued token $e_k(t) = \mathrm{MLP}_s(x_k, y_k, s_k(t))$, so identical measurements from different locations are not collapsed.

2. **DeepSets aggregation.** $r_t = \mathrm{MLP}_{\text{pool}}\bigl(\frac{1}{K}\sum_k e_k(t)\bigr)$ — permutation-invariant, supports variable $K$ at inference.

3. **Causal LSTM history encoder.** $h_T = \mathrm{LSTM}(r_0, \ldots, r_T)$ — strict causality up to $T_{\mathrm{obs}}$ (no bidirectional).

4. **Variational latent + observability proxy.**
   - $z \sim r_\phi(z \mid \mathcal{S}) = \mathcal{N}(\mu_z, \mathrm{diag}(\sigma_z^2))$, regularized with $\beta = 10^{-2}$ KL.
   - $o_\psi(x,y,t) = \sigma\bigl(g_\psi(h_T, x, y, t, d_s(x,y))\bigr) \in [0,1]$ — learned, not control-theoretic.

5. **Observability-conditioned Gaussian decoder.** $(\mu_\eta, \log\sigma_\eta^2) = f_\theta\bigl(z, \phi(x,y,t), o_\psi(x,y,t)\bigr)$, where $\phi$ is a Fourier-feature lift of the query coordinate.

At inference, predictive moments are computed by Monte-Carlo over $M{=}50$ latent samples using the law of total variance (aleatoric + epistemic).

Training objective: $\mathcal{L} = \mathcal{L}_{\mathrm{NLL}} + \beta \mathcal{L}_{\mathrm{KL}}$. Adaptive-$\beta$ and observability-ranking regularizers are explored in Appendix A.

---

## Reproduce

### Prerequisites
```bash
pip install torch numpy scipy xarray netCDF4 matplotlib cartopy
```

### Data
- **Copernicus SSH (CMEMS) — Gulf of Tonkin, Jan 2024.** Download via the CMEMS marine data store; subset to $105.5^\circ$E–$110.5^\circ$E, $16.5^\circ$N–$22.5^\circ$N, hourly resolution.
- **HYCOM GOFS 3.1 GLBy0.08** (for the cross-product negative-results study). See [`experiment/download_hycom_extended.py`](experiment/download_hycom_extended.py).
- **PDEBench 2D Shallow Water (Radial Dam Break)** — `2D_rdb_NA_NA.h5` from PDEBench.

### Train OVCNO on Copernicus
```bash
cd experiment
python train_layout.py --layout sensors_real_stations.json --epochs 100 --beta 0.01
```

### Evaluate
```bash
python eval_horizon.py        # Horizon-conditioned metrics (Table 3)
python eval_layout.py         # Real/Eq/Random layout OSSE (Table 5)
python eval_missing_sensors.py # Sensor dropout robustness (Table 6)
python audit_copernicus_corr.py # Sanity-check diagnostics (Table 7)
```

### Regenerate paper figures
```bash
cd docs && python generate_figures.py
```

---

## Project journey — 11 weeks

| Weeks | Milestone | Key outcome |
|---|---|---|
| 1–4 | 1D HLL + DeepONet proof of concept | Validated branch–trunk idea |
| 5–6 | 2D well-balanced HLL solver (GoT) | First end-to-end pipeline broke at 93.6% RelL2 |
| 7–8 | PDEBench 2D SWE benchmark | Architecture vindicated (5.25% with 16 boundary sensors) |
| 9 | **The Great Debug** — 4-layer hypothesis test | C-property bug + tidal forcing + ocean sensors + Fourier PE → **14.8%** |
| 10 | **Causal Forecasting pivot** — ForecastDeepONet | LSTM-causal (4.46%) beats MLP-non-causal (5.25%) on PDEBench |
| 11 | **OVCNO** — variational + observability-aware | Paper-track contribution; primary applied OSSE on Copernicus |

Full narrative: [`docs/project_journey.md`](docs/project_journey.md).

---

## Honest scientific note

The HYCOM cross-product validation initially looked positive (Corr$_S$ = 0.41) but was traced to a **mean-collapse artifact** through four ablation variants and a Round-2 decoupled-architecture follow-up that confirmed StdRatio plateaus at ~0.30 on HYCOM regardless of architecture. HYCOM is therefore **excluded from the paper main body** and kept as a future-work direction. The full investigation is in [`hycom_complete_report.md`](hycom_complete_report.md) and [`hycom_v2_experiments.md`](hycom_v2_experiments.md). The Copernicus uncertainty signal passes five independent sanity checks (StdRatio = 1.54, partial Corr = 0.564 after controlling for sensor distance, monotonic $\hat\sigma$-bin MAE) — see Table 7 of the paper.

A 250-question pre-submission self-audit covering dataset, sensor construction, architecture, training, metrics, ablations, and reviewer-risk defenses is in [`audit_250_qa.md`](audit_250_qa.md).

---

## Limitations (acknowledged in paper §5)

1. **Validation, not final test.** The held-out one-month Copernicus period is used both for evaluation and for checkpoint selection. A multi-month dataset with separate val/test splits is needed for operational claims.
2. **Calibration–sharpness trade-off.** OVCNO intervals are wider than VCO (Avg.W 0.287 vs 0.209). Post-hoc calibration (e.g. conformal) is a natural next step.
3. **Observability proxy is data-driven**, not formal control-theoretic. Correlations with error/uncertainty are moderate (Pearson 0.22).
4. **Level-1 OSSE.** Real station coordinates, but sensor values still sampled from the gridded analysis. Level-2 with actual tide-gauge measurements would strengthen operational claims.
5. **Mean–variance coupling.** The decoder conditions both heads on $o_\psi$; this caused mean collapse on HYCOM and motivates decoupled designs in future work.
6. **Training-seed variance.** At $K{=}12$, training initialization variance is larger than layout-induced variance, limiting placement conclusions.

---

## Citation

If you find this work useful:

```bibtex
@misc{vu2026ovcno,
  title  = {Observability-Aware Causal Neural Operators for Probabilistic
            Full-Field Forecasting from Sparse Boundary Sensors},
  author = {Vu Hung Anh},
  year   = {2026},
  note   = {TTCS course project, Posts and Telecommunications Institute of Technology (PTIT)},
  url    = {https://github.com/scalliontor/Sparse_sensor_tidal}
}
```

---

## License

For educational purposes (TTCS course project, PTIT).
