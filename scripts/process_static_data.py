import numpy as np
import xarray as xr
import os
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def process_gebco():
    gebco_file = "data/real_data/GEBCO_03_Mar_2026_c85fa4391b0e/gebco_2025_n22.5_s16.5_w105.5_e110.5.nc"
    print(f"Loading GEBCO data from {gebco_file}...")
    ds = xr.open_dataset(gebco_file)
    
    lon = ds['lon'].values
    lat = ds['lat'].values
    elev = ds['elevation'].values
    
    # Define our 2D Godunov Grid (approx 5km resolution)
    lon_min, lon_max = 105.5, 110.5
    lat_min, lat_max = 16.5, 22.5
    
    Nx = 110 # ~5km
    Ny = 130 # ~5km
    
    lon_grid = np.linspace(lon_min, lon_max, Nx)
    lat_grid = np.linspace(lat_min, lat_max, Ny)
    
    print(f"Interpolating to {Nx}x{Ny} grid...")
    interp = RegularGridInterpolator((lat, lon), elev, bounds_error=False, fill_value=0)
    
    LON, LAT = np.meshgrid(lon_grid, lat_grid) # shape (Ny, Nx)
    pts = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
    
    elev_grid = interp(pts).reshape((Ny, Nx))
    
    # Save
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/elev_grid.npy", elev_grid)
    np.save("data/processed/lon_grid.npy", lon_grid)
    np.save("data/processed/lat_grid.npy", lat_grid)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(elev_grid, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.contour(lon_grid, lat_grid, elev_grid, levels=[0], colors='red', linewidths=1.5, label='Coastline')
    plt.title("GEBCO Bathymetry - Interpolated to 5km Grid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("data/processed/gebco_5km.png")
    print("Process complete. Saved elev_grid.npy and a visualization map.")

if __name__ == "__main__":
    process_gebco()
