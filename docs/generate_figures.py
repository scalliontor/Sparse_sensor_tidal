import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('/mnt/DA0054DE0054C365/ttcs/docs/figures', exist_ok=True)
plt.style.use('ggplot')

# 1. Spatial Error U-Shape Plot
distances = ['0-40km\n(Near sensor)', '40-80km\n(Moderate)', '80-160km\n(Far interior)', '>160km\n(Deep interior)']
rmse = [0.41, 0.22, 0.16, 0.30]

plt.figure(figsize=(8, 5))
plt.bar(distances, rmse, color=['#e63946', '#457b9d', '#1d3557', '#e63946'], width=0.5)
plt.plot(distances, rmse, marker='o', markersize=10, color='black', linestyle='--')
plt.title('Spatial Error Degradation by Distance to Nearest Observed Sensor', fontsize=14, fontweight='bold')
plt.ylabel('RMSE (meters)', fontsize=12)
plt.xlabel('Distance to Nearest Sensor', fontsize=12)
plt.ylim(0, 0.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/mnt/DA0054DE0054C365/ttcs/docs/figures/spatial_error_ushape.png', dpi=300)
plt.close()

# 2. Diminishing Returns on Sensors
sensors = ['16 Sensors\n(~75km gap)', '32 Sensors\n(~37km gap)', '64 Sensors\n(~18km gap)']
t_obs_20 = [17.57, 18.33, 17.38]
t_obs_60 = [19.42, 19.58, 19.40]

x = np.arange(len(sensors))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, t_obs_20, width, label='Forecast 80 steps (far future)', color='#2a9d8f')
rects2 = ax.bar(x + width/2, t_obs_60, width, label='Forecast 40 steps (near future)', color='#e9c46a')

ax.set_ylabel('Relative L2 Error (%)', fontsize=12)
ax.set_title('Sensor Topology Limits (Diminishing Returns)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sensors, fontsize=11)
ax.set_ylim(10, 22)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig('/mnt/DA0054DE0054C365/ttcs/docs/figures/sensor_ablation.png', dpi=300)
plt.close()

# 3. Model Capacity Plot
models = ['Light Model\n(~1.1M Params, 30 Ep)', 'Heavy Model\n(~4M Params, 100 Ep)', 'Full-State\nBaseline']
error = [22.09, 17.57, 13.79]

plt.figure(figsize=(8, 5))
plt.plot(models, error, marker='D', markersize=12, color='#e76f51', linewidth=3)
plt.fill_between(models, error, color='#e76f51', alpha=0.1)
plt.title('Impact of Model Capacity on Forecasting', fontsize=14, fontweight='bold')
plt.ylabel('Relative L2 Error (%)', fontsize=12)
plt.ylim(10, 25)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(error):
    plt.text(i, v + 0.5, f'{v}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('/mnt/DA0054DE0054C365/ttcs/docs/figures/model_capacity.png', dpi=300)
plt.close()
print("Plots generated in /mnt/DA0054DE0054C365/ttcs/docs/figures/")
