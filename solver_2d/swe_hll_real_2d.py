import numpy as np
import os

# ═══════════════════════════════════════════════════════════════
# Well-balanced HLL solver for 2D Shallow Water Equations
# with variable bathymetry via hydrostatic reconstruction.
#
# Key fix: HLL fluxes use reconstructed water depths that
# account for bed elevation differences at each interface,
# ensuring exact preservation of the lake-at-rest steady state
# (C-property).
# ═══════════════════════════════════════════════════════════════

def flux_x(U, g=9.81):
    h = U[..., 0]
    qx = U[..., 1]
    qy = U[..., 2]
    h_safe = np.maximum(h, 1e-12)
    u = qx / h_safe
    v = qy / h_safe
    F = np.zeros_like(U)
    F[..., 0] = qx
    F[..., 1] = qx * u + 0.5 * g * h**2
    F[..., 2] = qx * v
    return F

def flux_y(U, g=9.81):
    h = U[..., 0]
    qx = U[..., 1]
    qy = U[..., 2]
    h_safe = np.maximum(h, 1e-12)
    u = qx / h_safe
    v = qy / h_safe
    G = np.zeros_like(U)
    G[..., 0] = qy
    G[..., 1] = qy * u
    G[..., 2] = qy * v + 0.5 * g * h**2
    return G

def hll_flux_x(UL, UR, g=9.81):
    hL_safe = np.maximum(UL[..., 0], 1e-12)
    hR_safe = np.maximum(UR[..., 0], 1e-12)
    uL = UL[..., 1] / hL_safe
    uR = UR[..., 1] / hR_safe
    cL = np.sqrt(g * hL_safe)
    cR = np.sqrt(g * hR_safe)
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    SL = np.minimum(SL, 0.0)
    SR = np.maximum(SR, 0.0)
    FL = flux_x(UL, g)
    FR = flux_x(UR, g)
    denom = np.maximum(SR - SL, 1e-12)[..., np.newaxis]
    SL_ext = SL[..., np.newaxis]
    SR_ext = SR[..., np.newaxis]
    FHLL = (SR_ext * FL - SL_ext * FR + SL_ext * SR_ext * (UR - UL)) / denom
    return FHLL

def hll_flux_y(UD, UU, g=9.81):
    hD_safe = np.maximum(UD[..., 0], 1e-12)
    hU_safe = np.maximum(UU[..., 0], 1e-12)
    vD = UD[..., 2] / hD_safe
    vU = UU[..., 2] / hU_safe
    cD = np.sqrt(g * hD_safe)
    cU = np.sqrt(g * hU_safe)
    SD = np.minimum(vD - cD, vU - cU)
    SU = np.maximum(vD + cD, vU + cU)
    SD = np.minimum(SD, 0.0)
    SU = np.maximum(SU, 0.0)
    GD = flux_y(UD, g)
    GU = flux_y(UU, g)
    denom = np.maximum(SU - SD, 1e-12)[..., np.newaxis]
    SD_ext = SD[..., np.newaxis]
    SU_ext = SU[..., np.newaxis]
    GHLL = (SU_ext * GD - SD_ext * GU + SD_ext * SU_ext * (UU - UD)) / denom
    return GHLL

def update_cfl_dt(U, dx, dy, cfl=0.5, g=9.81):
    h = np.maximum(U[..., 0], 1e-12)
    u = np.abs(U[..., 1] / h)
    v = np.abs(U[..., 2] / h)
    c = np.sqrt(g * h)
    max_speed_x = np.max(u + c)
    max_speed_y = np.max(v + c)
    if max_speed_x < 1e-12 and max_speed_y < 1e-12:
        return cfl * min(dx, dy) / np.sqrt(9.81 * 1.0)  # fallback
    dt_x = dx / max(max_speed_x, 1e-12)
    dt_y = dy / max(max_speed_y, 1e-12)
    return cfl * min(dt_x, dt_y)

def apply_boundary_conditions_real(U, t, ssh_array, bathymetry, water_level_base=0.0,
                                    sensor_yx=None, base_h=None):
    """
    Apply boundary conditions with two modes:

    1. If sensor_yx is provided: nudge SSH at specific ocean sensor positions
       (sensors placed inside domain at ocean cells near boundary).
       SSH is interpolated along South and East open boundaries from sensor values.

    2. Fallback: old-style ghost cell BCs (for backward compatibility).

    ssh_array: shape (N_sensors,) containing SSH anomaly for current time.
               First half: South boundary sensors. Second half: East boundary sensors.
    sensor_yx: dict with 'south' and 'east' arrays of (y, x) positions, or None.
    base_h: array (Ny, Nx) of resting water depth, needed for sensor mode.
    """
    Ny, Nx = bathymetry.shape

    # 1. Closed Wall: West (x=0) and North (y=Ny+1) — always reflective
    U[:, 0, 0] = U[:, 1, 0]
    U[:, 0, 1] = -U[:, 1, 1]
    U[:, 0, 2] = U[:, 1, 2]

    U[-1, :, 0] = U[-2, :, 0]
    U[-1, :, 1] = U[-2, :, 1]
    U[-1, :, 2] = -U[-2, :, 2]

    half = len(ssh_array) // 2
    s_south = ssh_array[0:half]
    s_east = ssh_array[half:]

    if sensor_yx is not None and base_h is not None:
        # --- Mode A: Relaxation nudging at sensor positions ---
        # Problem: ghost cells are adjacent to land, causing huge h jumps.
        # Solution: gently nudge h at the sensor row/col toward SSH target.
        # Use relaxation: h_new = (1-alpha)*h_current + alpha*h_target
        # alpha controls nudging strength per call (called every timestep).
        alpha = 0.05  # gentle nudging — ~20 timesteps to reach target
        south_yx = sensor_yx['south']
        east_yx = sensor_yx['east']

        # South: nudge at sensor row
        south_row = south_yx[0][0]
        south_x_positions = np.array([x for y, x in south_yx], dtype=float)
        ssh_south_interp = np.interp(np.arange(Nx), south_x_positions, s_south)
        target_h_s = np.maximum(ssh_south_interp + base_h[south_row, :], 1e-6)
        ocean_s = base_h[south_row, :] > 1.0
        ey_s = south_row + 1  # extended grid row

        current_h_s = U[ey_s, 1:Nx+1, 0]
        nudged_h_s = (1 - alpha) * current_h_s + alpha * target_h_s
        U[ey_s, 1:Nx+1, 0] = np.where(ocean_s, nudged_h_s, current_h_s)

        # Ghost row: reflective wall (land boundary)
        U[0, :, 0] = U[1, :, 0]
        U[0, :, 1] = -U[1, :, 1]
        U[0, :, 2] = U[1, :, 2]

        # East: nudge at sensor col
        east_col = east_yx[0][1]
        east_y_positions = np.array([y for y, x in east_yx], dtype=float)
        ssh_east_interp = np.interp(np.arange(Ny), east_y_positions, s_east)
        target_h_e = np.maximum(ssh_east_interp + base_h[:, east_col], 1e-6)
        ocean_e = base_h[:, east_col] > 1.0
        ex_e = east_col + 1

        current_h_e = U[1:Ny+1, ex_e, 0]
        nudged_h_e = (1 - alpha) * current_h_e + alpha * target_h_e
        U[1:Ny+1, ex_e, 0] = np.where(ocean_e, nudged_h_e, current_h_e)

        # Ghost col: reflective wall
        U[:, -1, 0] = U[:, -2, 0]
        U[:, -1, 1] = -U[:, -2, 1]
        U[:, -1, 2] = U[:, -2, 2]
    else:
        # --- Mode B: Legacy ghost cell BCs ---
        # 2. Open Boundary: South (y=0)
        x_idx_sensors = np.linspace(0, Nx-1, half)
        ssh_south_interp = np.interp(np.arange(Nx), x_idx_sensors, s_south)
        base_h_south = np.maximum(-bathymetry[0, :], 0)
        h_south = np.maximum(ssh_south_interp + base_h_south, 1e-6)

        U[0, 1:-1, 0] = h_south
        U[0, 1:-1, 1] = U[1, 1:-1, 1] * (h_south / np.maximum(U[1, 1:-1, 0], 1e-6))
        U[0, 1:-1, 2] = U[1, 1:-1, 2] * (h_south / np.maximum(U[1, 1:-1, 0], 1e-6))

        # 3. Open Boundary: East (x=Nx+1)
        y_idx_sensors = np.linspace(0, Ny-1, half)
        ssh_east_interp = np.interp(np.arange(Ny), y_idx_sensors, s_east)
        base_h_east = np.maximum(-bathymetry[:, -1], 0)
        h_east = np.maximum(ssh_east_interp + base_h_east, 1e-6)

        U[1:-1, -1, 0] = h_east
        U[1:-1, -1, 1] = U[1:-1, -2, 1] * (h_east / np.maximum(U[1:-1, -2, 0], 1e-6))
        U[1:-1, -1, 2] = U[1:-1, -2, 2] * (h_east / np.maximum(U[1:-1, -2, 0], 1e-6))

def simulate_real_2d(
    bathymetry, ssh_timeseries, t_timeseries,
    Lx=555_000.0, Ly=666_000.0, cfl=0.4, g=9.81, save_every=50,
    sensor_yx=None
):
    Ny, Nx = bathymetry.shape
    dx = Lx / Nx
    dy = Ly / Ny
    
    U = np.zeros((Ny + 2, Nx + 2, 3), dtype=float)
    # Initial steady state h = max(-bathy, 0)
    base_h = np.maximum(-bathymetry, 0)
    base_h = np.maximum(base_h, 0.01)  # dry-cell minimum
    U[1:-1, 1:-1, 0] = base_h
    
    # ── Bed elevation on extended grid (for hydrostatic reconstruction) ──
    # b = bathymetry (negative underwater)
    b_ext = np.zeros((Ny + 2, Nx + 2), dtype=float)
    b_ext[1:-1, 1:-1] = bathymetry
    b_ext[0, :] = b_ext[1, :]
    b_ext[-1, :] = b_ext[-2, :]
    b_ext[:, 0] = b_ext[:, 1]
    b_ext[:, -1] = b_ext[:, -2]
    
    # Free surface elevation: eta = h + b (for underwater b < 0, so eta = h + b)
    # At rest: eta = 0 everywhere, h = -b for ocean cells
    
    t = 0.0
    t_end = (len(t_timeseries) - 1) * 3600.0
    step = 0
    next_save_t = 0.0
    
    out_t = []
    out_h = []
    
    t_hours = t_timeseries
    
    print(f"Starting simulate_real_2d: dx={dx:.2f}, dy={dy:.2f}, base_h max={base_h.max():.2f}")
    
    while t < t_end:
        if t >= next_save_t:
            out_t.append(t)
            out_h.append(U[1:-1, 1:-1, 0] - base_h)
            next_save_t += 3600.0
            print(f"\rTime: {t/3600:.2f}h / {t_end/3600:.2f}h", end="")
            
        t_hr = t / 3600.0
        idx = int(np.floor(t_hr))
        idx2 = min(idx + 1, len(t_hours) - 1)
        w2 = t_hr - idx
        w1 = 1.0 - w2
        current_ssh = ssh_timeseries[idx] * w1 + ssh_timeseries[idx2] * w2
        
        apply_boundary_conditions_real(U, t, current_ssh, bathymetry,
                                        sensor_yx=sensor_yx, base_h=base_h)
        
        dt = update_cfl_dt(U[1:-1, 1:-1, :], dx, dy, cfl, g)
        
        if t + dt > t_end:
            dt = t_end - t
        
        # ── Hydrostatic Reconstruction for x-direction ──
        # Free surface: eta = h + b
        eta = U[:, :, 0] + b_ext
        
        # At interface (i, i+1): reconstruct water depths
        # eta_L = eta at cell i, eta_R = eta at cell i+1
        # b_star = max(b[i], b[i+1]) — bed elevation at interface
        # h_L* = max(eta_L - b_star, 0)
        # h_R* = max(eta_R - b_star, 0)
        b_star_x = np.maximum(b_ext[:, :-1], b_ext[:, 1:])  # (Ny+2, Nx+1)
        hL_star = np.maximum(eta[:, :-1] - b_star_x, 0.0)
        hR_star = np.maximum(eta[:, 1:]  - b_star_x, 0.0)
        
        # Build reconstructed states for HLL
        UL_star = U[:, :-1, :].copy()
        UR_star = U[:, 1:, :].copy()
        
        # Scale momentum to match reconstructed depth (preserve velocity)
        hL_orig = np.maximum(U[:, :-1, 0], 1e-12)
        hR_orig = np.maximum(U[:, 1:, 0], 1e-12)
        ratio_L = hL_star / hL_orig
        ratio_R = hR_star / hR_orig
        
        UL_star[:, :, 0] = hL_star
        UL_star[:, :, 1] *= ratio_L
        UL_star[:, :, 2] *= ratio_L
        UR_star[:, :, 0] = hR_star
        UR_star[:, :, 1] *= ratio_R
        UR_star[:, :, 2] *= ratio_R
        
        Fx = hll_flux_x(UL_star, UR_star, g)
        
        # ── Hydrostatic Reconstruction for y-direction ──
        b_star_y = np.maximum(b_ext[:-1, :], b_ext[1:, :])
        hD_star = np.maximum(eta[:-1, :] - b_star_y, 0.0)
        hU_star = np.maximum(eta[1:, :]  - b_star_y, 0.0)
        
        UD_star = U[:-1, :, :].copy()
        UU_star = U[1:, :, :].copy()
        
        hD_orig = np.maximum(U[:-1, :, 0], 1e-12)
        hU_orig = np.maximum(U[1:, :, 0], 1e-12)
        ratio_D = hD_star / hD_orig
        ratio_U = hU_star / hU_orig
        
        UD_star[:, :, 0] = hD_star
        UD_star[:, :, 1] *= ratio_D
        UD_star[:, :, 2] *= ratio_D
        UU_star[:, :, 0] = hU_star
        UU_star[:, :, 1] *= ratio_U
        UU_star[:, :, 2] *= ratio_U
        
        Fy = hll_flux_y(UD_star, UU_star, g)
        
        # ── Source term from hydrostatic reconstruction ──
        # Pressure correction at each interface
        h_int = U[1:-1, 1:-1, 0]
        
        # x-direction pressure source (at left and right interfaces of each cell)
        # S_x = 0.5*g*(h^2 - hL_star^2) at left interface
        #     + 0.5*g*(h^2 - hR_star^2) at right interface
        Sx_left  = 0.5 * g * (h_int**2 - hL_star[1:-1, 1:, ]**2)  # right side of left interface
        Sx_right = 0.5 * g * (h_int**2 - hR_star[1:-1, :-1]**2)  # left side of right interface
        
        Sy_down  = 0.5 * g * (h_int**2 - hD_star[1:, 1:-1]**2)
        Sy_up    = 0.5 * g * (h_int**2 - hU_star[:-1, 1:-1]**2)
        
        # ── Update ──
        U[1:-1, 1:-1, :] -= (dt / dx) * (Fx[1:-1, 1:, :] - Fx[1:-1, :-1, :]) \
                          + (dt / dy) * (Fy[1:, 1:-1, :] - Fy[:-1, 1:-1, :])
        
        # Add hydrostatic pressure correction to momentum
        # Sign: += to CANCEL the flux imbalance (well-balanced / C-property)
        U[1:-1, 1:-1, 1] += (dt / dx) * (Sx_right - Sx_left)
        U[1:-1, 1:-1, 2] += (dt / dy) * (Sy_up - Sy_down)
        
        U[1:-1, 1:-1, 0] = np.maximum(U[1:-1, 1:-1, 0], 0.01)
        
        t += dt
        step += 1

    print("\nSimulation complete.")
    return np.array(out_t), np.array(out_h)
