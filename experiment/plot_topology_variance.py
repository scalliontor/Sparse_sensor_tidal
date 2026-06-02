"""
Regenerate topology_variance.png with 2-panel boxplots.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load(name):
    with open(f"results_{name}.json") as f:
        return json.load(f)

real_seeds = [42, 123, 777, 2024, 31415, 2718, 999]
eq_seeds = [42, 123, 777, 2024, 31415, 2718, 999]
random_ls = [0, 1, 2, 3, 4]

real = [load(f"real_k12_s{s}") for s in real_seeds]
eq = [load(f"eq_k12_s{s}") for s in eq_seeds]
rand = [load(f"random_k12_ls{s}") for s in random_ls]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

metrics = [
    ('RMSE $\downarrow$', 'rmse'),
    ('CRPS $\downarrow$', 'crps'),
]

colors = ['#e74c3c', '#3498db', '#95a5a6']
labels = ['Real-K12\n(7 train seeds)', 'Eq-K12\n(7 train seeds)', 'Random-K12\n(5 layout seeds)']

for ax, (title, key) in zip(axes, metrics):
    r_vals = [d[key] for d in real]
    e_vals = [d[key] for d in eq]
    n_vals = [d[key] for d in rand]
    
    data = [r_vals, e_vals, n_vals]
    
    bplot = ax.boxplot(data, patch_artist=True, widths=0.5,
                       medianprops=dict(color='black', linewidth=1.5),
                       flierprops=dict(marker='o', markersize=5))
    
    # Overplot the points
    for i, vals in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, color='black', alpha=0.5, s=20, zorder=3)
        bplot['boxes'][i].set_facecolor(colors[i])
        bplot['boxes'][i].set_alpha(0.7)
        
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(title, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Add annotation clearly
axes[0].text(3, max([d['rmse'] for d in real]), 
             "Training-seed variance\ndominates layout variance\nRMSE std: 0.017 / 0.0005 $\\approx 33\\times$",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
             ha='right', va='top')

plt.savefig("topology_variance.png", dpi=300, bbox_inches='tight')
print("Saved topology_variance.png")

