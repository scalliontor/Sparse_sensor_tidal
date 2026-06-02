import json
import numpy as np
import matplotlib.pyplot as plt

def load_layout(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('stations', data.get('sensors'))

def plot_layout_comparison():
    # Load ocean mask from netCDF (73, 61)
    import xarray as xr
    nc_path = "../data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_105.5-110.5E_16.5-22.5N.nc"
    try:
        raw_ds = xr.open_dataset(nc_path)
    except:
        nc_path = "../data/real_data/copernicus_ssh_tonkin_jan2024.nc"
        raw_ds = xr.open_dataset(nc_path)
    if 'zos' in raw_ds:
        ssh = raw_ds['zos'].values
    elif 'sea_surface_height' in raw_ds:
        ssh = raw_ds['sea_surface_height'].values
    else:
        ssh = raw_ds['sla'].values
    Ny, Nx = 73, 61
    mask = (~np.isnan(ssh[0])).squeeze()
    if mask.ndim > 2:
        mask = mask[0]
    x_norm = np.linspace(-1, 1, Nx)
    y_norm = np.linspace(-1, 1, Ny)
    
    # Load layouts
    real_sensors = load_layout("sensors_real_stations.json")
    eq_sensors = load_layout("sensors_equispaced.json")
    rand_sensors = load_layout("sensors_random_seed0.json")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    
    # Custom colormap for background: light gray for land, very light blue for ocean
    import matplotlib.colors as colors
    cmap = colors.ListedColormap(['#e0e0e0', '#d4e6f1'])
    
    titles = ["A) Real-K12 Tide-Gauge Network", 
              "B) Equispaced-K12 Layout", 
              "C) Random-K12 Layout (Example)"]
    sensor_sets = [real_sensors, eq_sensors, rand_sensors]
    markers = [('^', 'red', 10), ('o', 'blue', 8), ('D', 'gray', 8)]
    
    for idx_p, ax in enumerate(axes):
        ax.imshow(mask, origin='lower', cmap=cmap)
        ax.set_title(titles[idx_p], fontsize=14, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot sensors
        mk, col, sz = markers[idx_p]
        for s in sensor_sets[idx_p]:
            if s.get('j') is None or s.get('i') is None:
                continue
            kx, ky = s['j'], s['i']
            ax.plot(kx, ky, marker=mk, color=col, markersize=sz, markeredgecolor='white', markeredgewidth=1.0)
            
            # Optional: Add text labels for Real stations if it's the first panel
            if idx_p == 0 and 'name' in s:
                # Add a small offset so text doesn't overlap marker
                ax.text(kx + 1.5, ky, s['name'], color='black', fontsize=9, 
                        fontweight='bold', verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
                
    plt.savefig("layout_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved layout_comparison.png")

if __name__ == "__main__":
    plot_layout_comparison()
