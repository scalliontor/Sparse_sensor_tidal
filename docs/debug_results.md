# Debug Results — Gulf of Tonkin Case

## Tầng 1: Vật lý & Target

### H3 CONFIRMED: Mass drift, không phải tidal oscillation
```
Ocean mean η:  0.000 → 0.006 → 0.015 → 0.021 → 0.027 → 0.031 → 0.035 → 0.037
               t=0     t=24    t=48    t=72    t=96    t=120   t=144   t=156
```
**Monotonic increase, không có oscillation.** Đây là mass accumulation, không phải triều.

### H4 CONFIRMED: Extreme values ở coastal hotspot
```
t=80h:  max=2.06m tại (y=8, x=36),  bath=+1.2m  → LAND CELL
t=120h: max=3.32m tại (y=8, x=36),  bath=+1.2m  → LAND CELL  
t=160h: max=4.23m tại (y=119,x=89), bath=+59.3m → LAND CELL
```
**Max anomaly KHÔNG nằm ở ocean** — nằm ở ô đất nông bị ngập (wetting).

### Eta theo depth band (t=166h):
| Depth band | N cells | Mean η | Max η |
|---|---|---|---|
| Very shallow (0-10m, LAND) | 558 | 0.078m | 2.16m |
| Shallow (10-50m, LAND) | 3288 | 0.035m | 0.087m |
| Mid ocean (50-200m) | 3442 | 0.035m | 0.036m |
| Deep ocean (>200m) | 357 | 0.035m | 0.035m |

→ **Ocean cells gần như đồng nhất** (η ≈ 0.035m, std ≈ 0). Toàn bộ variance nằm ở shallow land cells bị flooding.

---

## Tầng 2: Input Data

### H2 PARTIALLY CONFIRMED: Dead sensors do land, KHÔNG phải NaN

```
NaN count: 0/744 cho TẤT CẢ 20 sensors
```
**Không có NaN!** Dead sensors thật sự = 0.000 vì:

```
South sensor 0-9: ALL nằm tại y=0 (row 0), bath = 0.0 (land boundary)
East sensor 10-19: ALL nằm tại x=109 (last col), bath = 0.0 (land boundary)  
```

**Cả 20 sensors đều nằm trên ô bath=0!** Không có sensor nào nằm trong ocean cell. Copernicus SSH ở điểm gần bờ cho giá trị = 0 hoặc rất nhỏ vì resolution không đủ.

→ **Sensor placement sai**: boundary grid row 0 và column 109 đều là đất, không phải ocean.

### Active sensors (std > 0):
- South: 4-9 (std 0.017–0.075m) — dù bath=0, có một ít SSH signal lan tới
- East: 10-14, 16-17 (std 0.015–0.106m)
- Dead: 0-3 (South gần bờ West), 15, 18-19 (East gần bờ North/South)

---

## Tầng 3: Dataset Analysis

### Label distribution (Window 0):

| Subset | p50 | p90 | p99 | Max |
|---|---|---|---|---|
| All cells | 0.012m | 0.034m | 0.991m | 4.39m |
| Ocean only | 0.023m | 0.033m | 0.035m | 2.16m |
| Ocean + t>24h | 0.025m | 0.034m | 0.035m | 2.16m |
| Ocean + t>48h | 0.027m | 0.034m | 0.036m | 2.16m |

**Phát hiện quan trọng**: Ocean-only labels cực kỳ tập trung:
- p99 = 0.035m (tức 99% ocean cells có η < 3.5cm)
- Variance gần như bằng 0 trong ocean
- Tất cả extreme values (>0.1m) đến từ **land/shallow cells bị ngập**

→ **Bài toán "predict ocean η" thực chất là predict một hằng số** (≈ mean boundary SSH tích lũy)

---

## Tầng 4: Baselines

| Method | RMSE (m) | RelL2 |
|---|---|---|
| Predict train mean | 0.194 | 97.4% |
| Predict by time-bin | 0.192 | 96.5% |
| Per-window mean (oracle) | 0.194 | 97.4% |
| **DeepONet (100 epochs)** | **0.186** | **93.6%** |

### H5 CONFIRMED: Branch input gần như vô dụng

DeepONet chỉ tốt hơn baseline "predict by time" **3% RelL2** (96.5% → 93.6%).
Per-window oracle (biết window nào) **không giúp gì** so với predict mean.

→ **SSH boundary input không mang thêm thông tin có ý nghĩa** cho prediction.

---

## Kết luận Debug

### 5 giả thuyết — kết quả:

| # | Giả thuyết | Kết quả |
|---|---|---|
| H1 | Sai định nghĩa target η | **KHÔNG** — η = h - base_h = free surface, đúng |
| H2 | Boundary preprocessing lỗi | **MỘT PHẦN** — sensors nằm trên land (bath=0), không phải NaN fill |
| H3 | Solver tích lũy mass | **ĐÚNG** — monotonic increase, no oscillation |
| H4 | Extreme values ở hotspot | **ĐÚNG** — max 4.4m ở land cells bị wetting |
| H5 | Branch input vô dụng | **ĐÚNG** — DeepONet ≈ time-only baseline |

### Root causes chính:

1. **SSH signal quá yếu** (SLA, đã remove tide) → forcing ±0.07m → ocean response đồng nhất ≈ 0.035m
2. **Sensor placement trên land** → bath=0 tại tất cả sensors → mất kết nối với ocean dynamics  
3. **Mass drift** → không có oscillation → bài toán degenerate thành "predict một hằng số tăng dần"
4. **Land cells chiếm variance** → 100% extreme values từ wetting, không phải tidal dynamics

### Để fix Gulf of Tonkin cần:
1. **TPXO tidal forcing** thay Copernicus SLA → amplitude ±1-2m
2. **Sensors trong ocean** (không phải trên boundary grid edge)
3. **Spin-up** + periodic BCs để có oscillation
4. **Mask land cells** khỏi training


