"""
Generate synthetic tidal SSH forcing for Gulf of Tonkin sensors.

Uses real tidal constituents for the region:
- M2 (principal lunar semidiurnal): period 12.42h, dominant in GoT
- S2 (principal solar semidiurnal): period 12.00h
- K1 (lunar diurnal): period 23.93h
- O1 (principal lunar diurnal): period 25.82h

Gulf of Tonkin has strong diurnal tides (K1, O1 dominant).
Amplitudes based on published tidal charts for the region.

Sensor placement: ocean cells near South and East open boundaries.
"""

import numpy as np
import os

def generate_tidal_ssh(n_sensors, n_hours, seed=42):
    """Generate realistic tidal SSH timeseries for boundary sensors."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours, dtype=np.float64)  # hours
    
    # Tidal constituents for Gulf of Tonkin
    # (name, period_hours, amplitude_range_m, phase_spread_rad)
    constituents = [
        ("K1",  23.93, (0.3, 0.6),  np.pi),     # dominant diurnal
        ("O1",  25.82, (0.3, 0.5),  np.pi),     # major diurnal
        ("M2",  12.42, (0.1, 0.3),  np.pi),     # semidiurnal
        ("S2",  12.00, (0.05, 0.15), np.pi),    # solar semidiurnal
        ("P1",  24.07, (0.05, 0.15), np.pi),    # solar diurnal
    ]
    
    ssh = np.zeros((n_hours, n_sensors), dtype=np.float64)
    
    for name, period, amp_range, phase_spread in constituents:
        omega = 2 * np.pi / period  # rad/hour
        
        # Each sensor gets slightly different amplitude and phase
        # (simulates spatial variation along boundary)
        amps = rng.uniform(amp_range[0], amp_range[1], size=n_sensors)
        phases = rng.uniform(0, phase_spread, size=n_sensors)
        
        # Add spatial coherence: neighboring sensors have similar phases
        # Smooth phases with running average
        for _ in range(3):
            phases_smooth = np.copy(phases)
            for i in range(1, n_sensors - 1):
                phases_smooth[i] = 0.5 * phases[i] + 0.25 * (phases[i-1] + phases[i+1])
            phases = phases_smooth
        
        for s in range(n_sensors):
            ssh[:, s] += amps[s] * np.cos(omega * t + phases[s])
    
    # Add small noise (measurement/interpolation error)
    ssh += rng.normal(0, 0.02, size=ssh.shape)
    
    # Remove mean to center around 0
    ssh -= ssh.mean(axis=0, keepdims=True)
    
    return ssh.astype(np.float32)


def place_sensors_in_ocean(bathymetry, n_south=10, n_east=10, min_depth=20.0):
    """
    Place sensors at ocean cells near South and East boundaries.
    Sensors placed a few rows/cols inward from the grid edge to be in ocean.
    """
    Ny, Nx = bathymetry.shape
    
    # South boundary: find ocean cells in rows 1-4
    south_sensors = []
    for target_row in [2, 3, 4, 5]:
        for x in range(Nx):
            if bathymetry[target_row, x] < -min_depth:
                south_sensors.append((target_row, x, bathymetry[target_row, x]))
        if len(south_sensors) >= n_south * 3:
            break
    
    # Select n_south evenly spaced from available
    if len(south_sensors) >= n_south:
        indices = np.linspace(0, len(south_sensors) - 1, n_south, dtype=int)
        south_sensors = [south_sensors[i] for i in indices]
    
    # East boundary: find ocean cells in cols Nx-3 to Nx-6
    east_sensors = []
    for target_col in [Nx - 2, Nx - 3, Nx - 4, Nx - 5]:
        for y in range(Ny):
            if bathymetry[y, target_col] < -min_depth:
                east_sensors.append((y, target_col, bathymetry[y, target_col]))
        if len(east_sensors) >= n_east * 3:
            break
    
    if len(east_sensors) >= n_east:
        indices = np.linspace(0, len(east_sensors) - 1, n_east, dtype=int)
        east_sensors = [east_sensors[i] for i in indices]
    
    return south_sensors, east_sensors


def main():
    bath = np.load("data/processed/elev_grid.npy")
    Ny, Nx = bath.shape
    print(f"Grid: {Ny}x{Nx}")
    
    # Place sensors in ocean
    south_s, east_s = place_sensors_in_ocean(bath, n_south=10, n_east=10, min_depth=20.0)
    
    print(f"\nSouth sensors: {len(south_s)}")
    for y, x, b in south_s:
        print(f"  (y={y}, x={x}) bath={b:.1f}m")
    
    print(f"\nEast sensors: {len(east_s)}")
    for y, x, b in east_s:
        print(f"  (y={y}, x={x}) bath={b:.1f}m")
    
    n_sensors = len(south_s) + len(east_s)
    
    # Generate 744 hours (31 days) of tidal SSH
    n_hours = 744
    ssh = generate_tidal_ssh(n_sensors, n_hours, seed=42)
    
    print(f"\nSSH shape: {ssh.shape}")
    print(f"SSH range: [{ssh.min():.4f}, {ssh.max():.4f}]")
    print(f"SSH std per sensor: min={ssh.std(axis=0).min():.4f}, max={ssh.std(axis=0).max():.4f}")
    
    # Verify oscillation
    print(f"\nFirst sensor timeseries (every 6h):")
    for ti in range(0, 48, 6):
        print(f"  t={ti}h: {ssh[ti, 0]:.4f}m")
    
    # Save
    np.save("data/processed/ssh_tidal_20s.npy", ssh)
    
    # Save sensor positions for use in solver boundary conditions
    sensor_positions = {
        "south": [(y, x) for y, x, b in south_s],
        "east": [(y, x) for y, x, b in east_s],
    }
    np.savez("data/processed/sensor_positions.npz",
             south_yx=np.array([(y, x) for y, x, b in south_s]),
             east_yx=np.array([(y, x) for y, x, b in east_s]),
             south_bath=np.array([b for y, x, b in south_s]),
             east_bath=np.array([b for y, x, b in east_s]))
    
    print(f"\nSaved: data/processed/ssh_tidal_20s.npy")
    print(f"Saved: data/processed/sensor_positions.npz")


if __name__ == "__main__":
    main()
