import numpy as np
import matplotlib.pyplot as plt

def plot_robustness_curve():
    # Data from Table 7
    avail = np.array([100, 75, 50])
    
    # Real-K12 Avg.W
    real_w_mean = np.array([0.259, 0.260, 0.261])
    real_w_std  = np.array([0.019, 0.023, 0.026])
    
    # Eq-K12 Avg.W
    eq_w_mean = np.array([0.274, 0.280, 0.289])
    eq_w_std  = np.array([0.020, 0.015, 0.003])
    
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    
    # Plot Equispaced first
    ax.errorbar(avail, eq_w_mean, yerr=eq_w_std, fmt='-o', color='blue', 
                linewidth=2, markersize=8, capsize=5, capthick=2, 
                label='Equispaced-K12 (Synthetic)')
    
    # Plot Real
    ax.errorbar(avail, real_w_mean, yerr=real_w_std, fmt='-^', color='red', 
                linewidth=2, markersize=8, capsize=5, capthick=2, 
                label='Real-K12 (Tide-Gauge Stations)')
    
    ax.set_xlim(105, 45) # Invert axis so it drops from 100 to 50
    ax.set_xticks([100, 75, 50])
    ax.set_xticklabels(['100%', '75%', '50%'])
    
    ax.set_xlabel("Sensor Availability (Test-Time Dropout)", fontsize=12)
    ax.set_ylabel("Average Interval Width (Avg.W) $\downarrow$", fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Missing-Sensor Robustness", fontsize=14, fontweight='bold', pad=10)
    
    ax.legend(fontsize=11, loc='upper left')
    
    # Set y limit to show the gap properly
    ax.set_ylim(0.22, 0.31)
    
    plt.savefig("robustness_curve.png", dpi=300, bbox_inches='tight')
    print("Saved robustness_curve.png")

if __name__ == "__main__":
    plot_robustness_curve()
