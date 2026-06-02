# HYCOM Cross-Product Validation — Complete Report

> **Date**: 2026-04-28
> **Status**: Experiment completed, section removed from paper
> **Decision**: HYCOM Corr_S = 0.41 is an artifact → removed from main paper, kept as future work

---

## 1. Experiment Overview

### Goal
Validate whether OVCNO's observability-aware uncertainty generalizes from Copernicus CMEMS to an independent gridded ocean-analysis product: **HYCOM GOFS 3.1 GLBy0.08**.

### Dataset

| Item | Value |
|---|---|
| Product | HYCOM + NCODA GOFS 3.1 / GLBy0.08 |
| Domain | Gulf of Tonkin, 105.5–110.5°E, 16.5–22.5°N |
| Grid | 0.08° lon × 0.04° lat |
| Variable | `surf_el` (SSH, absolute, meters) |
| Temporal resolution | 3-hourly |
| Time range | Jan–Sep 2024 |
| Split | Train: Jan–Jun (1468 steps) · Val: Jul (258 steps) · Test: Aug–Sep (258 steps) |
| Sensors | Real-K12 tide-gauge coordinates (same as Copernicus OSSE) |
| Normalization | Train-period temporal mean only (no leakage) |

### Key difference from Copernicus
- HYCOM stores **absolute SSH** (~0.04–1.37 m), not SSH anomaly
- Multi-month data → enables true independent test set (Aug–Sep)
- Different data assimilation system → tests cross-product robustness

---

## 2. Phase 1 — Smoke Test (1 seed)

### v1 (5-month data) vs v2 (9-month data)

| Metric | v1-VCO | v1-OVCNO | **v2-VCO** | **v2-OVCNO** |
|---|---|---|---|---|
| RMSE | 0.059 | 0.102 | **0.044** | 0.101 |
| NLL | -2.11 | -1.78 | **-2.56** | -1.92 |
| Cov@95 | 81.5% ⚠️ | 97.6% | **98.4%** ✅ | **98.0%** ✅ |
| Avg.W | 0.170 | 0.404 | **0.222** | 0.408 |
| **Corr_S** | 0.042 | 0.106 | 0.001 | **0.538** ✅ |

### Smoke findings
1. v2 (9-month) substantially improved VCO: RMSE 0.059→0.044, Cov 81.5%→98.4%
2. OVCNO Corr_S jumped from 0.106 to **0.538** with more data
3. OVCNO RMSE stuck at ~0.10, not improving with more data ← **early warning sign**
4. OVCNO best checkpoint at **epoch 4** (very early), val NLL diverges after epoch 5

---

## 3. Phase 2 — Full Run (3 seeds)

### Per-seed results on independent test (Aug–Sep 2024)

| Model | Seed | RMSE | MAE | NLL | Cov@95 | Avg.W | Corr_S |
|---|---|---|---|---|---|---|---|
| VCO | 42 | **0.044** | **0.035** | **-2.56** | 98.4% | **0.222** | 0.001 |
| VCO | 123 | **0.045** | **0.036** | **-2.58** | 96.6% | **0.201** | -0.049 |
| VCO | 456 | 0.053 | 0.042 | -2.38 | 98.7% | 0.269 | -0.009 |
| OVCNO | 42 | 0.101 | 0.080 | -1.92 | 98.0% | 0.408 | **0.538** |
| OVCNO | 123 | 0.048 | 0.039 | -2.48 | 98.5% | 0.250 | 0.139 |
| OVCNO | 456 | 0.095 | 0.076 | -1.92 | 99.8% | 0.449 | **0.560** |

### Aggregated

| Metric | VCO (mean±std) | OVCNO (mean±std) | Winner |
|---|---|---|---|
| RMSE | **0.047±0.005** | 0.081±0.029 | VCO |
| NLL | **-2.51±0.11** | -2.11±0.32 | VCO |
| Cov@95 | 97.9% | **98.8%** | ≈ tie |
| Avg.W | **0.230** | 0.369 | VCO |
| **Corr_S** | -0.019±0.026 | **0.412±0.234** | OVCNO |

### Initial interpretation (later overturned)
- VCO point forecast clearly better (RMSE 0.047 vs 0.081)
- OVCNO uncertainty appears informative (Corr_S = 0.41 vs -0.02)
- Suggested claim: "OVCNO's observability-aware uncertainty generalizes to HYCOM"

---

## 4. Phase 3 — Diagnostic Red Flags

### Spatial prediction maps revealed

Visualization of OVCNO predictions showed:
- **Predicted mean**: near-constant / near-zero everywhere (spatial structure missing)
- **Uncertainty field**: still has spatial gradient (higher away from sensors)
- **Observability field**: has learned geometry (gradient follows sensor distance)

> ⚠️ The model learned WHERE sensors are (in the variance/obs head), but NOT the actual SSH dynamics (in the mean head).

### Comparison with Copernicus

| Diagnostic | Copernicus OVCNO | HYCOM OVCNO |
|---|---|---|
| SpatialStdRatio | **1.54** (healthy) | **~0.38** (collapse) |
| Mean predictor | Learned spatial structure | Near-constant (collapsed) |
| Corr_S source | Genuine | Artifact of mean collapse |

---

## 5. Phase 4 — Diagnostic Ablation (4 variants)

To isolate the root cause, we trained 4 OVCNO variants on HYCOM:

| Variant | Description | Key Change |
|---|---|---|
| **A_decouple** | obs only feeds variance head | Separate mean/variance pathways |
| **B_lobs01** | ranking loss weight = 0.1 | Reduced observability regularization |
| **C_ckpt_rmse** | checkpoint by best RMSE | Alternative model selection |
| **D_lobs00** | ranking loss weight = 0.0 | No observability ranking at all |

### Results

| Variant | RMSE | NLL | Cov@95 | Avg.W | **Corr_S** | **StdRatio** |
|---|---|---|---|---|---|---|
| A_decouple | 0.048 | -2.51 | 93.7% | 0.188 | **-0.038** | 0.621 |
| B_lobs01 | 0.047 | -2.52 | 96.1% | 0.213 | 0.016 | 0.542 |
| C_ckpt_rmse | 0.048 | -2.45 | 88.4% | 0.155 | 0.058 | 0.579 |
| D_lobs00 | **0.045** | **-2.57** | **98.1%** | 0.214 | -0.003 | 0.376 |
| *Original* | *0.081* | *-2.11* | *98.8%* | *0.369* | ***0.412*** | — |
| *VCO ref* | *0.047* | *-2.51* | *97.9%* | *0.230* | *-0.019* | — |

### Devastating finding

**All 4 variants achieve VCO-level RMSE (~0.045–0.048), but ALL lose Corr_S (→ ≈ 0).**

This proves:

```
When OVCNO learns good mean predictions on HYCOM → Corr_S drops to ~0
When OVCNO has bad mean predictions (collapse) → Corr_S inflates to ~0.41
```

---

## 6. Root Cause: Mean Collapse Artifact

### The mechanism

```
Original OVCNO on HYCOM:
    mean predictor → collapsed (near-constant output)
    ↓
    error |ε| ≈ |gt - const| ≈ spatial SSH pattern
    ↓
    variance head → learned sensor distance geometry
    ↓
    Corr(|ε|, σ) > 0  ← ARTIFACT, not genuine uncertainty tracking
```

### Why it happens on HYCOM but not Copernicus

| Factor | Copernicus | HYCOM |
|---|---|---|
| SSH scale | Anomaly (~±0.2m) | Absolute (~0.04–1.37m) |
| Spatial std | ~0.03m | Larger |
| Training convergence | Mean learns structure | Mean collapses early |
| Best checkpoint | ~epoch 15-20 | Epoch 2-5 |
| Var head interferes with mean? | No (StdRatio=1.54) | Yes (StdRatio<0.4) |

The observability field feeds into BOTH mean and variance heads. On HYCOM's larger-scale data, the observability signal dominates the mean head too early, causing it to predict a geometry-shaped artifact instead of SSH dynamics.

### Evidence matrix

| Scenario | RMSE | Corr_S | Interpretation |
|---|---|---|---|
| OVCNO original (mean=collapse) | 0.081 | 0.41 | ❌ Artificial: errors ≈ SSH spatial pattern |
| OVCNO fixed (mean=learned) | 0.045–0.048 | ≈0 | ✅ Honest: good mean → no error-sigma structure |
| VCO (always learned) | 0.047 | ≈0 | ✅ Baseline: no spatial uncertainty info |

---

## 7. Cross-Check: Copernicus NOT Affected

We ran the same 5-diagnostic audit on Copernicus OVCNO:

| Diagnostic | Result | Verdict |
|---|---|---|
| SpatialStdRatio | **1.54** | ✅ No collapse (pred std > gt std) |
| Partial Corr(|ε|, σ \| d_s) | **0.564** | ✅ Survives distance control |
| Good-sample Corr_S | **0.272** | ✅ Not driven by outliers |
| σ-bin MAE monotonicity | 0.027 → 0.117 | ✅ Genuine risk stratification |
| Constant-mean baseline Corr | 0.365 | ✅ Below model's 0.577 |

**Conclusion**: Copernicus Corr_S = 0.52 is genuine. HYCOM Corr_S = 0.41 is artifact.

---

## 8. Decision: Remove from Paper

### Rationale
1. HYCOM Corr_S = 0.41 is an artifact of mean collapse
2. When mean is fixed, OVCNO = VCO on HYCOM (both Corr_S ≈ 0)
3. Including it would expose the paper to reviewer attack
4. The honest finding is negative: "OVCNO does not provide informative uncertainty on HYCOM"

### What was removed from paper
- Abstract: HYCOM sentence
- Section 4.7: Cross-Product Validation (entire subsection + table)
- Summary: bullet about HYCOM
- Conclusion: HYCOM mention
- Limitations: HYCOM-specific items
- Bibliography: `chassignet2007hycom`

### What remains
- Future Work mentions "cross-product validation on independent products such as HYCOM" as future direction

---

## 9. Strategic Options for Future Work

### Option A: Fix OVCNO architecture → retrain on HYCOM

**The Mode C approach**: Decouple mean and variance training.

```
Phase 1: Train mean head only (freeze obs/variance)
Phase 2: Freeze mean, train obs + variance heads
```

This prevents the observability field from interfering with mean learning. A prototype `model_ovcno_v2.py` was created with:
- Separate `MeanDecoder` and `VarianceDecoder`
- Observability field feeds ONLY into variance decoder
- Mean decoder never sees obs signal

**Estimated effort**: 1-2 days training + evaluation

### Option B: Use HYCOM for point-accuracy validation only

Reposition HYCOM as: "OVCNO achieves comparable point accuracy to VCO on independent products."

This is true (when mean is fixed properly), but weak as a paper claim.

### Option C: Different domain / product

Test on a completely different ocean region (e.g., South China Sea, Mediterranean) where the SSH dynamics may be more amenable to the current architecture.

### Option D: Multi-product ensemble

Train on both Copernicus + HYCOM jointly. This would require:
- Normalization harmonization
- Domain-adaptive layers
- Much longer training

---

## 10. Key Takeaways

> [!IMPORTANT]
> ### What we learned
> 1. **OVCNO uncertainty informativeness is NOT automatically portable across products.** It depends on whether the mean predictor converges properly on the new data.
> 2. **Corr_S can be misleading.** A model with collapsed mean + spatial variance can produce high Corr_S without being genuinely useful.
> 3. **The diagnostic pattern is clear**: high RMSE + high Corr_S = red flag for mean collapse.
> 4. **The Mode C decoupled architecture is the right fix**, but requires additional development.
> 5. **Copernicus results are genuine** — confirmed by 5 independent diagnostics.

> [!CAUTION]
> ### What NOT to claim
> - ❌ "OVCNO generalizes to HYCOM"
> - ❌ "Corr_S = 0.41 on independent product"
> - ❌ "Observability-aware uncertainty is product-agnostic"

> [!TIP]
> ### What CAN be claimed (future work)
> - ✅ "Cross-product validation is identified as a necessary future step"
> - ✅ "A decoupled mean-variance architecture is proposed to address the mean-collapse failure mode"
> - ✅ "Copernicus results pass rigorous sanity checks (5 diagnostics)"

---

## Appendix: All Experiment Files

| File | Purpose | Location |
|---|---|---|
| `dataset_hycom.py` | HYCOM data loading + Real-K12 sensor extraction | Server |
| `download_hycom.py` | HYCOM GOFS 3.1 data download script | Server |
| `train_hycom_smoke.py` | 1-seed smoke test training | Server |
| `train_hycom_full.py` | 3-seed full training | Server |
| `train_hycom_diagnostic.py` | 4-variant ablation training | Server |
| `model_ovcno_v2.py` | Decoupled-heads OVCNO prototype | Local |
| `audit_copernicus_corr.py` | 5-diagnostic Copernicus audit | Server |
| `plot_hycom_predictions.py` | Spatial prediction maps (cartopy) | Server |
