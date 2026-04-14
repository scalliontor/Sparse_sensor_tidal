# Phân tích Data, Architecture & Problems

## 1. Data Pipeline

### 1.1 SSH Sensor Input (Branch)
- **Source**: Copernicus SSH, 744 giờ (31 ngày), 20 sensors tại biên
- **Shape**: `(744, 20)` — mỗi giờ 1 sample, 20 sensors
- **Range**: [-0.211, 0.253] m — rất nhỏ
- **Mean**: ~0, Std: 0.043 m

**Vấn đề lớn: 7/20 sensors CHẾT (std = 0)**
| Sensor | Vị trí | Std (m) | Trạng thái |
|--------|--------|---------|-------------|
| 0-3 | South biên | 0.000 | DEAD — luôn = 0 |
| 15 | East biên | 0.000 | DEAD |
| 18-19 | East biên | 0.000 | DEAD |
| 4-14, 16-17 | | 0.015 – 0.106 | Active |

→ **Chỉ 13/20 sensors thực sự có tín hiệu**. 7 sensors = zero padding vô nghĩa chiếm 35% branch input.

### 1.2 Bathymetry (Domain)
- **Grid**: 130 × 110 (Vịnh Bắc Bộ)
- **Range**: [-1574.7m, +1799.2m] (deep ocean → high mountain)
- **Ocean cells**: 7,645 / 14,300 = 53.5%
- **Land cells**: 46.5% — gần nửa domain là đất, anomaly ≈ 0

### 1.3 Simulation Output (Ground Truth)
- **Solver**: HLL Godunov + Hydrostatic Reconstruction (well-balanced)
- **20 windows**, sliding 12h stride, mỗi window = 168h (7 ngày)
- **Output**: η = h - base_h (anomaly mực nước), shape `(167, 130, 110)` per window

**Anomaly statistics:**
| Metric | Giá trị |
|--------|---------|
| Global range | [-8.3e-5, 4.39] m |
| Mean across windows | 0.045 – 0.050 m |
| Std across windows | 0.188 m |

**Temporal pattern (Window 0, ocean cells):**
```
t=0h:   mean=0.000, max=0.000   ← khởi tạo từ rest
t=24h:  mean=0.006, max=0.330
t=48h:  mean=0.015, max=0.639
t=72h:  mean=0.021, max=0.995
t=96h:  mean=0.027, max=1.351
t=120h: mean=0.031, max=1.832
t=144h: mean=0.035, max=1.945   ← tăng monotonic, KHÔNG oscillate
```

→ **Anomaly chỉ TĂNG, không có chu kỳ triều rõ ràng**. Vì SSH forcing quá yếu (±0.07m) và simulation bắt đầu từ rest.

### 1.4 Dataset (Training)
- **Sampling**: 10,000 random (x,y,t) points per window
- **Train**: 16 windows × 10k = 160,000 samples
- **Test**: 4 windows × 10k = 40,000 samples
- **Branch dim**: 3,360 = flatten(168 × 20)
- **Trunk dim**: 3 = (x_norm, y_norm, t_norm)
- **Normalization**: z-score, y_mu=0.048, y_std=0.188

**Label distribution (denormalized, meters):**
```
p1=0.000  p5=0.000  p10=0.000  p25=0.000
p50=0.010  p75=0.022  p90=0.031  p95=0.189  p99=0.997

50.3% of labels < 0.01m (gần zero)
93.9% of labels < 0.1m
```

→ **Phần lớn labels gần 0**. Signal thật chỉ ở top 6% samples. Model dễ collapse vào predict mean ≈ 0.

---

## 2. Architecture hiện tại

### 2.1 DeepONet (Standard)

```
Branch MLP: 3360 → 256 → 256 → 256 → 256 → 128
Trunk MLP:  3    → 256 → 256 → 256 → 256 → 128
Output:     dot(branch, trunk) + bias → scalar η
```

- **Total params**: 1,321,985 (1.3M)
- **Branch**: 1,090,688 params (83%)
- **Trunk**: 231,296 params (17%)
- **Activation**: GELU
- **Optimizer**: AdamW, lr=1e-3 fixed, weight_decay=1e-6

### 2.2 Training
- **Epochs**: 100
- **Batch size**: 8192
- **Val split**: 10% random (KHÔNG split by window — leakage risk)
- **Loss**: MSE trên normalized labels
- **LR schedule**: Không có

### 2.3 Results
```
Train MSE:     0.916 (baseline predict-mean = 1.0)
Best Val RMSE: 1.013 (normalized)
Test RMSE:     0.186 m
Test RelL2:    93.6%
Test MAE:      0.066 m

GT range:   [0.000, 4.226] m
Pred range: [-0.009, 0.431] m  ← model chỉ predict gần mean
```

---

## 3. Vấn đề (Root Causes)

### 3.1 Data Quality Issues

**P1: SSH forcing quá yếu**
- SSH amplitude chỉ ±0.07m, tạo anomaly rất nhỏ (mean 0.05m)
- Copernicus SSH ở vùng này có thể đã bị smooth/filter, mất tín hiệu triều
- Cần kiểm tra: dùng TPXO tidal model thay Copernicus cho SSH forcing mạnh hơn?

**P2: 7/20 dead sensors**
- Sensors 0-3 (South boundary gần bờ) và 15,18-19 (East boundary gần bờ) luôn = 0
- 35% branch input là zeros — Branch MLP phải học ignore chúng
- Wasted capacity + noise

**P3: Anomaly monotonic, không oscillate**
- Simulation bắt đầu từ rest (η=0), nước chỉ vào chưa kịp rút
- 7 ngày có thể chưa đủ cho tidal cycle đầy đủ (M2 = 12.42h, nhưng spin-up phase?)
- Gần như không có negative anomaly

**P4: 50% labels ≈ 0 (land cells + early timesteps)**
- 46.5% domain là đất → anomaly = 0
- Random sampling hit nhiều land cells và timestep đầu
- Model dễ collapse vào predict 0

**P5: Chỉ 16 unique branch inputs cho training**
- 16 windows × 85% overlap → thực chất chỉ ~2-3 independent SSH patterns
- Model gần như memorize chứ không generalize

### 3.2 Architecture Issues

**A1: Branch MLP flatten destroys temporal structure**
- `flatten(168 × 20) = 3360` scalars — MLP không biết đâu là time, đâu là sensor
- Phải tự học temporal correlation từ raw positions — rất khó
- 7 dead sensors = 168 × 7 = 1176 zeros trong 3360 dims

**A2: Không có LR scheduling**
- LR = 1e-3 cố định → có thể quá lớn ở late training, oscillate quanh minimum

**A3: Val split random (không by window)**
- Train và val có thể share same window → leakage
- Val loss không reflect generalization thật

### 3.3 Signal-to-Noise Problem
- Useful signal: **top 6% samples** có |η| > 0.1m
- Noise floor: **94% samples** có |η| < 0.1m (phần lớn ≈ 0)
- MSE loss dominated by near-zero samples → model learns to predict ~0

---

## 4. Đề xuất cải thiện

### Short-term (không đổi architecture)
1. **Loại land cells khỏi training** — chỉ sample ocean cells (53.5% domain)
2. **Importance sampling**: oversample cells có |η| > 0.1m
3. **Bỏ dead sensors**: branch_in = 168 × 13 = 2184 thay vì 3360
4. **LR scheduling**: CosineAnnealing, warmup 10 epochs
5. **Train lâu hơn**: 500-1000 epochs
6. **Tăng points/window**: 50k thay vì 10k

### Medium-term (đổi architecture)
7. **Branch temporal encoder**: LSTM hoặc 1D-CNN thay MLP
   ```
   SSH (168, 13) → LSTM(hidden=256) → Linear(256→128)
   ```
8. **Attention mechanism** cho branch (transformer encoder)

### Long-term (đổi data)
9. **TPXO tidal forcing** thay Copernicus SSH → signal mạnh hơn (±1-2m)
10. **Thêm nhiều tháng data** (3-6 tháng) cho diversity
11. **Longer spin-up**: bỏ 24-48h đầu mỗi window, chỉ dùng quasi-steady state
